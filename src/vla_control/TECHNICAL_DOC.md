# VLA Control 技术文档

## 概述

`src/vla_control` 负责 Franka 本地执行侧控制。

当前主链路由三部分组成：

- `franka_env.py`
  负责相机观测采集、机器人状态读取、dynamic pose skill 执行。
- `coordinator.py`
  负责状态机、推理客户端对接、遥测记录、WebSocket 推流。
- `src/data_collection/trajectory_replay_recorder.py`
  负责按历史 CSV 轨迹回放，并把 replay 时的 state/action 重新记录成新 CSV。

这套实现的目标不是把所有逻辑堆在控制循环里，而是把：

- 观测采集
- 机器人控制
- 推理编排
- 录制与推流

拆成稳定的调用边界。

---

## 模块关系

在线推理执行链路：

```text
Coordinator.run_control_loop()
  -> FrankaEnv.start_control()
  -> Coordinator._step()
      -> FrankaEnv.get_observation(prompt)
      -> websocket_client_policy.infer(obs)
      -> FrankaEnv.enqueue_action(action)
      -> FrankaEnv._skill_loop()
          -> FrankaArm.goto_pose(dynamic=True)
          -> FrankaArm.publish_sensor_data(...)
```

轨迹回放链路：

```text
TrajectoryReplayRecorder.run()
  -> FrankaEnv.start()
  -> FrankaEnv.reset_to_home()
  -> FrankaEnv.start_skill_thread()
  -> loop:
      -> FrankaEnv.get_observation(prompt)
      -> FrankaEnv.enqueue_action(action, transform=False)
```

---

## Coordinator

`Coordinator` 是顶层编排器，不直接做底层机器人控制。

它的职责是：

- 维护状态机：`IDLE` / `RUNNING` / `HOMING`
- 管理 prompt
- 在 `RUNNING` 时向推理服务请求 action chunk
- 将 action 送入 `FrankaEnv`
- 维护最新图像、状态、目标位姿、动作、推理耗时
- 自动录制双路视频和遥测
- 通过 WebSocket 向前端推流

### 状态机

- `IDLE`
  不推理，只调用 `env.hold_pose()` 保持当前位置。
- `RUNNING`
  当本地 `action_plan` 耗尽时触发一次推理，取前 `replan_steps` 个 action 执行。
- `HOMING`
  调用 `env.home_and_restart()`，回 home 后重新启动 dynamic skill，再切回 `IDLE`。

### 控制频率

`Coordinator` 本身按 `control_hz` 跑主循环，默认 10Hz。

这个频率决定：

- 观测采样节奏
- 推理节奏
- action 入队节奏
- 录屏和遥测采样节奏

它不直接决定底层 sensor message 发布频率；底层插值发布由 `FrankaEnv` 控制。

---

## FrankaEnv

`FrankaEnv` 是执行层抽象，负责把上层 action 转成机器人可执行的 dynamic pose 流。

### 公开接口

生命周期相关：

- `start()`
  启动双路相机。
- `stop()`
  停止控制线程并关闭相机。
- `start_control()`
  启动相机并启动 dynamic skill 线程。
- `stop_control()`
  停止 dynamic skill 线程并关闭相机。
- `reset_to_home()`
  停止 dynamic skill 并调用 `reset_joints()`。
- `home_and_restart()`
  先回 home，再重启 dynamic skill。

控制相关：

- `start_skill_thread()`
- `enqueue_action(action, transform=True)`
- `hold_pose()`

观测相关：

- `get_observation(prompt)`
- `commanded_pose_array`
- `ee_force_torque`

### 内部结构

这次重构后，`FrankaEnv` 内部按职责拆成了几类 helper：

- 观测 helper
  - `_resize_observation_image`
  - `_read_robot_state`
  - `_get_robot_state_for_observation`
  - `_build_observation_state`
- 控制 helper
  - `_start_dynamic_skill`
  - `_next_action`
  - `_resolve_goal_pose`
  - `_maybe_update_gripper`
  - `_record_control_trace`
  - `_publish_interp_pose`
  - `_publish_termination`
- 缓存与姿态 helper
  - `_pose_to_rotvec`
  - `_set_commanded_pose_array`
  - `_refresh_robot_cache`

这样 `get_observation()` 和 `_skill_loop()` 不再重复拼装状态和消息，主流程更容易读。

---

## Observation 格式

`FrankaEnv.get_observation(prompt)` 返回：

```python
{
    "observation/image": np.ndarray[224, 224, 3] uint8,
    "observation/wrist_image": np.ndarray[224, 224, 3] uint8,
    "observation/state": np.ndarray[8] float64,
    "observation/joints": np.ndarray[7] float64,
    "prompt": str,
}, img1_raw, img2_raw
```

其中：

- `observation/image`
  外部相机图像，resize + pad 到 `224x224`
- `observation/wrist_image`
  腕部相机图像，resize + pad 到 `224x224`
- `observation/state`
  `[eef_pos(3), eef_rotvec(3), finger1(1), finger2(1)]`
- `observation/joints`
  7 维关节角

当机器人不可用时：

- `state` 返回全零 `(8,)`
- `joints` 返回全零 `(7,)`

当相机不可用时：

- 对应相机返回全黑帧

---

## Action 格式

上游推理输出的 action 语义是：

```python
action = [delta_pos(3), delta_axisangle(3), gripper_cmd(1)]
```

`FrankaEnv.transform_action()` 会做两件事：

- 平移和旋转按配置做 clip 和 scale
- 夹爪做二值化

当前规则：

```python
ta[6] = 0.0 if ta[6] >= 0 else 0.08
```

也就是：

- `>= 0` 视为闭合
- `< 0` 视为打开

`enqueue_action(..., transform=False)` 用于 replay 等场景，表示 action 已经是执行尺度，不再重复缩放。

---

## Dynamic Skill 控制模型

`FrankaEnv._skill_loop()` 运行在后台线程。

### 启动阶段

启动时会：

1. 等待 1 秒让机械臂状态同步
2. 读取当前位姿
3. 调用 `FrankaArm.goto_pose(..., dynamic=True, block=False)`
4. 缓存初始 `commanded_pose`

### 循环阶段

每个控制周期会：

1. 刷新机器人缓存
2. 从 action 队列取一个动作
3. 计算新的目标位姿
4. 更新 `commanded_pose_array`
5. 将 `timestamp/state/joint_state/action/commanded_pose` 作为一条原子 control trace 写入缓存
6. 如有需要则异步控制夹爪
7. 在 `CONTROL_DT` 内按 `INTERP_DT` 做 min-jerk 插值并发布 sensor message

### 退出阶段

退出时会：

- 尝试发送 `ShouldTerminateSensorMessage`
- 清理线程引用

如果线程在异常路径退出，也会进入相同的终止清理。

---

## 控制参数

`franka_env.py` 当前关键参数：

```python
CONTROL_DT = 0.1
INTERP_DT = 0.02

POS_CLIP = 1.0
ROT_CLIP = 0.1
POS_MAX_VEL = 0.1 / 3
ROT_MAX_VEL = (math.pi / 4) / 3
GRIPPER_SPEED = 0.08
```

含义：

- 上层 action 消费频率：10Hz
- 底层插值发布频率：50Hz
- 最大线速度：约 `0.033 m/s`
- 最大角速度：约 `0.262 rad/s`
- 夹爪速度：`0.08 m/s`

阻抗参数默认与 `frankapy` 常量保持一致：

```python
TRANSLATIONAL_STIFFNESS = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES
ROTATIONAL_STIFFNESS = FC.DEFAULT_ROTATIONAL_STIFFNESSES
```

在 `frankapy` 不可用时退回到 `[600,600,600]` 和 `[50,50,50]`。

---

## 缓存与线程模型

`FrankaEnv` 维护了几类缓存：

- `_cached_pose`
- `_cached_joints`
- `_cached_gripper_width`
- `_cached_ee_force_torque`
- `_commanded_pose_array`
- `_control_trace_sample`
- `_control_trace_history`
- `_prev_rotvec`

设计目的：

- `get_observation()` 尽量读缓存，避免重复走 ROS API
- 前端可以读取最新目标位姿和末端力矩
- 旋转向量在跨帧时尽量保持连续
- 采集 / 回放 / 遥测都从统一的 control trace 获取已对齐底层数据

旋转向量相关 helper 的职责分工：

- `normalize_rotvec(rotvec)`
  只做单帧规范化，把角度折叠到 `[-pi, pi]`，不关心跨帧连续性
- `normalize_rotvec_with_history(rotvec, prev_rotvec)`
  在当前旋转的等价轴角表示中，选择与上一帧最接近的分支，只保证跨帧连续性
- `_set_commanded_pose_array()`
  用单帧规范化后的 rotvec 更新目标位姿显示
- `_refresh_robot_cache()`
  用带历史的规范化结果更新实际位姿缓存，并把该连续表示继续作为下一帧历史

线程同步使用：

- `_lock`
  保护机器人缓存和 skill 线程引用
- `_skill_stop`
  通知 skill 线程退出
- `_action_queue`
  上层 action 到底层控制线程的通道

---

## TrajectoryReplayRecorder

`TrajectoryReplayRecorder` 的目标不是调试动态控制，而是：

- 读取历史 `data.csv`
- 按给定频率把 action 送回 `FrankaEnv`
- 同时从 env control trace 重新记录 replay 时实际执行的 `timestamp/state/action`

当前拆分为四个步骤：

- `_prepare_env()`
  启动相机，必要时回 home，启动 skill 线程
- `_record_sample()`
  采样当前 observation，并送入一帧 action
- `_sleep_until()`
  按 replay 频率对齐时钟
- `_build_output_meta()`
  组织输出 CSV 的元信息

这条链路仍然复用 `FrankaEnv`，不自己实现另一套 dynamic pose 控制。

---

## 已删除脚本

`src/vla_control/trajectory_replay.py` 已移除。

原因不是功能无用，而是它维护了一套和 `FrankaEnv` 并行的直接控制实现，和当前代码结构重复，长期会带来：

- 控制参数漂移
- 终止逻辑不一致
- replay 语义不一致
- 维护成本增加

目前保留并建议使用的是：

- 在线执行：`coordinator.py`
- CSV 回放并重记：`src/data_collection/trajectory_replay_recorder.py`

---

## 当前限制

以下问题当前文档只记录，不在这次重构里改行为：

- action 队列仍然是无界队列
- action 还没有做 shape / `NaN` / `Inf` 校验
- `commanded_pose` 与 `actual_pose` 的偏差重同步逻辑当前未恢复

如果后续继续改，优先级建议是：

1. action 校验
2. 队列限长或只保留最新动作
3. 偏差检测与温和重同步

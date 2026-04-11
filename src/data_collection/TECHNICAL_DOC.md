# Data Collection 技术文档

## 概述

`src/data_collection` 负责录制 Franka 遥操作数据，输出双路视频和状态-动作对，供后续 VLA 训练使用。

当前实现包含两部分：

- `key_control.py`
  负责遥操作控制，支持 `keyboard` 和 `ps4` 两种输入设备。
- `data_recorder.py`
  负责双路 D435 取流、空动作过滤、episode 保存。

设计目标不是“录键盘”，而是生成和 `src/vla_control/franka_env.py` 执行侧兼容的 `state/action` 数据格式。

---

## 模块关系

启动链路：

```text
start_data_collector.sh
  -> python data_recorder.py
      -> KeyboardController(input_device=...)
      -> DataRecorder(...)
```

运行期职责：

- `KeyboardController`
  - 初始化 `FrankaArm(with_gripper=True)`
  - 维护 `commanded_pose / prev_commanded_pose / gripper_target`
  - 以 10Hz 采样输入
  - 以 50Hz 发布平滑插值后的 dynamic pose 指令
- `DataRecorder`
  - 以 10Hz 采样 `state/action`
  - 过滤空动作
  - 在录制开启时缓存图像和数据
  - 保存为 `cam1.mp4`、`cam2.mp4`、`data.csv`

---

## 输入设备

### 默认行为

默认输入设备是键盘：

```bash
./start_data_collector.sh
```

也可以显式指定：

```bash
./start_data_collector.sh --input_device keyboard
./start_data_collector.sh --input_device ps4
./start_data_collector.sh --input_device ps4 --joystick_index 0
```

对应参数来自 `data_recorder.py`：

- `--input_device {keyboard,ps4}`
- `--joystick_index <int>`

### 键盘映射

代码里的控制对如下：

- 平移：`W/S`、`A/D`、`I/K`
- 旋转：`Q/E`、`U/O`、`J/L`
- 夹爪：`G` 关闭，`H` 打开，`F` 半开
- 速度档位：`+` / `-`
- 复位：`R`
- 录制：`1` 开始，`2` 结束
- 退出：`ESC`

实际增量由下面的代码决定：

```python
dx = (int('s' in keys) - int('w' in keys)) * MAX_DELTA_POS * speed
dy = (int('d' in keys) - int('a' in keys)) * MAX_DELTA_POS * speed
dz = (int('i' in keys) - int('k' in keys)) * MAX_DELTA_POS * speed

droll = (int('q' in keys) - int('e' in keys)) * MAX_DELTA_ROT * speed
dpitch = (int('u' in keys) - int('o' in keys)) * MAX_DELTA_ROT * speed
dyaw = (int('l' in keys) - int('j' in keys)) * MAX_DELTA_ROT * speed
```

如果你关心正负方向，以这段代码为准，不要以旧文档或口头描述为准。

### PS4 二代手柄映射

当前 `pygame` 映射如下：

- 左摇杆：`X/Y` 平移
- 右摇杆：`Z` 平移和 `Yaw`
- `L1/R1`：`Roll`
- `L2/R2`：`Pitch`
- `方块`：关闭夹爪
- `圆圈`：打开夹爪
- `三角`：半开夹爪
- 十字键左右：速度档位减/加
- `OPTIONS`：复位
- `L3`：开始录制
- `R3`：结束录制
- `PS`：退出

代码里使用的轴/按钮编号：

```python
PS4_AXIS_LEFT_X = 0
PS4_AXIS_LEFT_Y = 1
PS4_AXIS_L2 = 2
PS4_AXIS_RIGHT_X = 3
PS4_AXIS_RIGHT_Y = 4
PS4_AXIS_R2 = 5

PS4_BUTTON_SQUARE = 0
PS4_BUTTON_CIRCLE = 2
PS4_BUTTON_TRIANGLE = 3
PS4_BUTTON_L1 = 4
PS4_BUTTON_R1 = 5
PS4_BUTTON_OPTIONS = 9
PS4_BUTTON_L3 = 10
PS4_BUTTON_R3 = 11
PS4_BUTTON_PS = 12
```

不同系统的 SDL 映射可能略有差异。如果实机按键不对，优先修改 `key_control.py` 顶部这些常量。

---

## 控制频率与速度

`key_control.py` 中的关键参数：

```python
INPUT_DT = 0.1
PUBLISH_DT = 0.02
MAX_LIN_VEL = 0.1
MAX_ROT_VEL = math.pi / 4
GRIPPER_SPEED = 0.08
```

含义：

- 输入采样频率：10Hz
- 控制发布频率：50Hz
- 最大线速度：`0.1 m/s`
- 最大角速度：`pi/4 rad/s`
- 夹爪速度：`0.08 m/s`

单次输入采样对应的最大增量：

```python
MAX_DELTA_POS = MAX_LIN_VEL * INPUT_DT   # 0.01 m
MAX_DELTA_ROT = MAX_ROT_VEL * INPUT_DT   # 0.0785 rad
```

速度倍率共三档：

```python
_speed_levels = [0.4, 0.7, 1.0]
```

默认是 `0.7` 档。

键盘和 PS4 共用同一套：

- `MAX_LIN_VEL`
- `MAX_ROT_VEL`
- `_speed_levels`

也就是说，手柄不是另一套速度体系，只是另一种输入源。

---

## State 与 Action 格式

### State

`get_state_and_action()` 返回的 `state` 为 `(8,)`：

```python
state = [pos(3), rotvec(3), finger1(1), finger2(1)]
```

具体含义：

- `state[0:3]`
  末端位置 `translation`
- `state[3:6]`
  末端姿态的旋转向量 `Rotation.from_matrix(...).as_rotvec()`
- `state[6]`
  当前夹爪宽度的一半
- `state[7]`
  当前夹爪宽度负的一半

代码实现：

```python
half = self._cached_gripper_width / 2.0
state = np.concatenate([pos, rotvec, [half], [-half]])
```

### Action

`action` 为 `(7,)`：

```python
action = [delta_pos(3), delta_axisangle(3), gripper_cmd(1)]
```

具体来源：

- `action[:3]`
  `commanded_pose.translation - prev_commanded_pose.translation`
- `action[3:6]`
  `commanded_pose.rotation @ prev_commanded_pose.rotation.T` 的轴角
- `action[6]`
  二值夹爪意图，不是物理宽度

夹爪编码：

```python
if self.gripper_target < 0.04:
    delta[6] = 1.0
else:
    delta[6] = -1.0
```

因此：

- `1.0` 表示闭合意图
- `-1.0` 表示打开意图

半开 `0.04m` 也会被编码为“打开意图”。

---

## 执行侧对齐

执行侧在 `src/vla_control/franka_env.py` 中使用：

```python
def transform_action(action):
    ta = action.copy()
    ta[:3] = np.clip(ta[:3], -POS_CLIP, POS_CLIP) * POS_SCALE
    ta[3:6] = np.clip(ta[3:6], -ROT_CLIP, ROT_CLIP) * ROT_SCALE
    ta[6] = 0.0 if ta[6] >= 0 else 0.08
    return ta
```

对应常量：

```python
CONTROL_DT = 0.1
POS_CLIP = 1.0
ROT_CLIP = 0.1
POS_MAX_VEL = 0.1 / 3
ROT_MAX_VEL = (math.pi / 4) / 3
POS_SCALE = POS_MAX_VEL * CONTROL_DT / POS_CLIP
ROT_SCALE = ROT_MAX_VEL * CONTROL_DT / ROT_CLIP
```

因此训练数据与执行侧的一致性是：

- 位置动作维度一致
- 旋转动作维度一致
- 夹爪动作语义一致
- 录制频率与执行 action 周期都为 10Hz

但也要注意：

- 采集端内部发布是 50Hz，为了让机械臂运动更平滑
- 执行侧真正消费模型 action 的周期仍是 10Hz

---

## 轨迹插值与发布

`key_control.py` 不是每次输入直接跳到新位姿，而是先在 10Hz 上生成一拍动作，再在 50Hz 上插值发布。

核心流程：

1. 每 `INPUT_DT=0.1s` 读取一次输入。
2. 若有动作，更新 `commanded_pose`。
3. 用 `min_jerk_weight` 在 `0 ~ INPUT_DT` 上生成插值轨迹。
4. 每 `PUBLISH_DT=0.02s` 发布一次目标 pose。

这样做的结果是：

- 用户输入采样率低，便于录制和对齐
- 机器人执行率高，避免明显顿挫

---

## 复位与退出

### 复位

调用 `_reset()` 时，当前实现会：

1. 打开夹爪
2. 终止当前 dynamic skill
3. 等待发布线程退出
4. `reset_joints()`
5. 重新读取当前 pose
6. 重新启动发布线程

### 退出

- 键盘模式下：`ESC`
- PS4 模式下：`PS`

退出时 `DataRecorder._on_exit()` 会尝试保存尚未结束但已有帧的轨迹。

---

## 录制逻辑

`DataRecorder` 默认参数：

```python
COLLECTION_DIR = repo_root / "collected"
TASK_NAME = "simple_pick_place"
COLLECT_HZ = 10.0
MAX_FRAMES = 1000
ACTION_THRESH_POS = 0.0005
ACTION_THRESH_ROT = 0.005
```

录制控制：

- 键盘模式：`1` 开始，`2` 结束
- PS4 模式：`L3` 开始，`R3` 结束

采集循环只在动作非空时记录：

```python
return (
    (np.abs(pos) < self.action_thresh_pos).all()
    and (np.abs(rot) < self.action_thresh_rot).all()
)
```

这意味着：

- 纯静止帧不会写入 `data.csv`
- 没有位移/旋转但只有夹爪变化时，也会被当作空动作跳过，因为当前过滤逻辑完全不检查 `action[6]`

---

## 数据保存格式

目录结构：

```text
collected/
  task_name/
    epo_1/
      cam1.mp4
      cam2.mp4
      data.csv
    epo_2/
      ...
```

episode 编号按 `epo_N` 自动递增。

### 视频

- `cam1.mp4`
- `cam2.mp4`

由 `imageio.mimwrite(..., codec="libx264", pixelformat="yuv420p")` 保存，帧率等于 `collect_hz`。

### CSV

`data.csv` 结构：

1. 第 1 行：单列 JSON 元数据
2. 第 2 行：表头
3. 第 3 行起：每行 `8 + 7 = 15` 个数值

第 1 行元数据字段：

```json
{
  "task_name": "...",
  "collect_hz": 10.0,
  "max_frames": 1000,
  "num_frames": 123
}
```

第 2 行列名：

```text
state_0 ... state_7 action_0 ... action_6
```

---

## 相机行为

`DataRecorder` 使用两路 D435：

- `CAM1_SERIAL = "346222072769"`
- `CAM2_SERIAL = "938422075745"`

当前实现特点：

- 某路相机启动失败时，记录 warning，并将该路视为不可用
- 不可用相机返回全黑帧
- 因此采集流程不因为单路相机失败而直接崩溃

---

## 依赖

与采集直接相关的 Python 依赖至少包括：

- `pyrealsense2`
- `imageio`
- `imageio-ffmpeg`
- `pynput`
- `pygame`，仅 PS4 模式必需

如果使用 PS4 模式但环境里没有 `pygame`，`KeyboardController(input_device="ps4")` 会直接抛错。

---

## 常用命令

默认启动：

```bash
cd /media/czl/sata/franka_my_code/vla4desk
./start_data_collector.sh
```

指定任务名：

```bash
./start_data_collector.sh --task_name pick_place
```

使用 PS4：

```bash
./start_data_collector.sh --input_device ps4
```

手动启动：

```bash
cd /root/Documents/my_code/vla4desk/src/data_collection
python data_recorder.py --task_name my_task --input_device ps4 --joystick_index 0
```

注意：`start_data_collector.sh` 是在容器里执行 `/root/Documents/my_code/vla4desk/...`，而仓库宿主机路径是 `/media/czl/sata/franka_my_code/vla4desk`，两者都是真实存在的启动路径，只是所处环境不同。

---

## 相关文件

- `src/data_collection/key_control.py`
- `src/data_collection/data_recorder.py`
- `src/vla_control/franka_env.py`
- `start_data_collector.sh`
- `requirements.txt`

---

最后更新：2026-04-12

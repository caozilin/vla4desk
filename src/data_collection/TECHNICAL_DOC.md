# Data Collection 技术文档

## 概述

`src/data_collection` 负责录制 Franka 遥操作数据，输出双路视频和状态-动作对，供后续 VLA 训练使用。

当前实现包含两部分：

- `key_control.py`
  负责遥操作控制，支持 `keyboard` 和 `ps4` 两种输入设备。
- `data_recorder.py`
  负责双路 D435 取流、基于 `action+state` 的去重过滤、episode 保存。

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
  - 以 10Hz 采样输入
  - 维护 `gripper_target` 与最近一拍 `state/action` 缓存
  - 将输入转换为 action，并通过 `FrankaEnv.enqueue_action(..., transform=False)` 送入执行侧
  - 不直接调用 dynamic skill 接口，避免和 env 的 skill 线程抢占控制面
- `DataRecorder`
  - 从 `FrankaEnv` 的 10Hz control trace 中提取已对齐的 `timestamp/state/joint_state/action/commanded_pose`
  - 仅在“action 近零且 state 与上一条已录制 state 基本一致”时跳过
  - 在录制开启时缓存图像和数据
  - 保存为 `cam1.mp4`、`cam2.mp4`、`data.json`

---

## 输入设备

### 默认行为

默认输入设备是键盘：

```bash
./start_data_collector.sh
```

默认 `task_name` 是 `default`。如果希望直接写到以前常用的任务目录名，可以显式传：

```bash
./start_data_collector.sh --task_name simple_pick_place
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
- 夹爪：`G` 关闭，`H` 打开
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
- `三角`：打开夹爪
- `圆圈`：关闭夹爪
- `叉 / Cross`：若正在录制则作废当前轨迹，不落盘；随后打开夹爪并复位
- `方块`：复位
- 十字键左右：速度档位减/加
- `OPTIONS`：退出
- `L3`：开始录制
- `R3`：结束录制

代码里使用的轴/按钮编号：

```python
PS4_AXIS_LEFT_X = 0
PS4_AXIS_LEFT_Y = 1
PS4_AXIS_L2 = 2
PS4_AXIS_RIGHT_X = 3
PS4_AXIS_RIGHT_Y = 4
PS4_AXIS_R2 = 5

PS4_BUTTON_CROSS = 0
PS4_BUTTON_CIRCLE = 1
PS4_BUTTON_TRIANGLE = 2
PS4_BUTTON_SQUARE = 3
PS4_BUTTON_L1 = 4
PS4_BUTTON_R1 = 5
PS4_BUTTON_OPTIONS = 9
PS4_BUTTON_L3 = 11
PS4_BUTTON_R3 = 12
```

不同系统的 SDL 映射可能略有差异。如果实机按键不对，优先修改 `key_control.py` 顶部这些常量。

---

## 控制频率与速度

`key_control.py` 中的关键参数：

```python
INPUT_DT = 0.1
MAX_LIN_VEL = 0.1
MAX_ROT_VEL = math.pi / 4
```

含义：

- 输入采样频率：10Hz
- 最大线速度：`0.1 m/s`
- 最大角速度：`pi/4 rad/s`

单次输入采样对应的最大增量：

```python
MAX_DELTA_POS = MAX_LIN_VEL * INPUT_DT   # 0.01 m
MAX_DELTA_ROT = MAX_ROT_VEL * INPUT_DT   # 0.0785 rad
```

速度倍率共三档：

```python
_speed_levels = [0.4, 0.7, 1.0]
```

默认是 `1.0` 档。

键盘和 PS4 共用同一套：

- `MAX_LIN_VEL`
- `MAX_ROT_VEL`
- `_speed_levels`

也就是说，手柄不是另一套速度体系，只是另一种输入源。

这里的速度上限是采集侧独立配置。

- `KeyboardController` 会直接按这套上限生成实际执行尺度的 action
- 送入 `FrankaEnv` 时使用 `transform=False`
- 因此不会复用 `FrankaEnv` 的推理侧 clip/scale 逻辑

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
  `finger1`，当前定义为夹爪宽度的一半
- `state[7]`
  `finger2`，当前定义为 `-finger1`

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
  当前 10Hz 输入采样拍上的目标位置增量，按基座坐标系表达
- `action[3:6]`
  当前 10Hz 输入采样拍上的目标旋转增量，输出时按基座坐标系 rotvec 表达
- `action[6]`
  二值夹爪意图，不是物理宽度

旋转的坐标系约定需要特别注意：

- 键盘和 PS4 的旋转输入，手感上按末端自身坐标系解释
- 但写入数据集和送入 `FrankaEnv` 的 `action[3:6]`，一定转换成基座坐标系 rotvec
- 这样可以同时满足“示教手感”与“数据/执行语义统一”

代码上，当前拍的局部旋转增量会按当前末端姿态做共轭变换：

```python
delta_local = Rotation.from_rotvec(local_rot_delta).as_matrix()
delta_base = current_rot @ delta_local @ current_rot.T
delta_rot_base = Rotation.from_matrix(delta_base).as_rotvec()
```

输入线程每个采样拍会：

1. 读取当前机器人 state
2. 读取键盘/PS4 输入
3. 将旋转输入从末端系转换到基座系
4. 生成 `(7,) action`
5. 调用 `self.env.enqueue_action(action, transform=False)`

因此：

- 本拍有输入：`action` 表示本拍输入增量
- 本拍没输入：`action[:6]` 为零
- `action` 不再通过 `commanded_pose - prev_commanded_pose` 反推

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

当前夹爪控制只有关闭/打开两档，不再保留半开语义。

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
- 录制使用 env skill 线程产出的 control trace，不再分别拼接 `state` 与 `action`

但也要注意：

- 采集端不会自己再开一套 50Hz 发布线程
- 真正的 dynamic skill 生命周期和 sensor message 发布都由 `FrankaEnv` 独占
- 采集端只负责以 10Hz 送 action，底层插值与发布由 `FrankaEnv` 内部处理

---

## 控制与线程模型

当前 `key_control.py` 的线程职责是：

- 输入线程：10Hz 读取输入，生成 action，送入 `FrankaEnv`
- env skill 线程：由 `FrankaEnv` 独占控制 dynamic pose skill 和底层插值发布

这样做的约束是：

- `key_control.py` 不允许直接调用 `goto_pose(dynamic=True)`
- `key_control.py` 不允许直接 `publish_sensor_data(...)`
- `key_control.py` 不允许自行发送 `ShouldTerminateSensorMessage`

这些接口都只应由 `FrankaEnv._skill_loop()` 使用，避免键控线程和循环线程抢占同一 interface skill。

---

## 复位与退出

### 复位

调用 `_reset()` 时，当前实现会：

1. 更新 `gripper_target`
2. 调用 `env.home_and_restart()`
3. 刷新本地缓存的 `state/action`

也就是说，复位也统一通过 `FrankaEnv` 串行处理，而不是采集侧自己终止/重启 skill。

### 退出

- 键盘模式下：`ESC`
- PS4 模式下：`PS`

退出时 `DataRecorder._on_exit()` 会尝试保存尚未结束但已有帧的轨迹。

---

## 录制逻辑

`DataRecorder` 默认参数：

```python
COLLECTION_DIR = repo_root / "collected"
TASK_NAME = "default"
COLLECT_HZ = 10.0
MAX_FRAMES = 1000
ACTION_THRESH_POS = 0.0005
ACTION_THRESH_ROT = 0.005
```

录制控制：

- 键盘模式：`1` 开始，`2` 结束
- PS4 模式：`L3` 开始，`R3` 结束

采集循环不是“只看 action 非空就记录”，而是先消费 env 输出的已对齐 control trace，再检查：

```python
if self._is_action_empty(action) and self._is_state_same_as_last_recorded(state):
    self.stats["skipped_frames"] += 1
    return
```

其中每条 control trace 至少包含：

- `seq_id`
- `timestamp`
- `state`
- `joint_state`
- `action`
- `commanded_pose`

这意味着：

- 纯静止帧不会写入 `data.json`
- 即使 `action` 近零，只要当前 `state` 与上一条已录制样本有明显差异，仍然会记录
- 当前 `_is_action_empty()` 仍然不检查 `action[6]`，所以夹爪-only 的过滤行为最终还取决于 `state` 是否变化
- `state/action/commanded_pose` 来自同一条底层 control trace，不再存在上层分别取样导致的一拍错位

---

## 数据保存格式

目录结构：

```text
collected/
  task_name/
    epo_1/
      cam1.mp4
      cam2.mp4
      data.json
    epo_2/
      ...
```

episode 编号按 `epo_N` 自动递增。

### 视频

- `cam1.mp4`
- `cam2.mp4`

由 `imageio.mimwrite(..., codec="libx264", pixelformat="yuv420p")` 保存，帧率等于 `collect_hz`。

### JSON

`data.json` 顶层结构：

```json
{
  "task_name": "...",
  "collect_hz": 10.0,
  "max_frames": 1000,
  "num_frames": 123,
  "action_scale": 100.0,
  "prompt": "",
  "frames": [
    {
      "timestamp": 0.1,
      "state": [...],
      "joint_state": [...],
      "action": [...],
      "commanded_pose": [...]
    }
  ]
}
```

逐帧字段：

- `timestamp`: 控制线程时间戳，单位秒
- `state`: `(8,)`
- `joint_state`: `(7,)`
- `action`: `(7,)`
- `commanded_pose`: `(6,)`

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
./start_data_collector.sh --task_name simple_pick_place
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

# Data Collection 技术文档

## 概述

数据采集模块 (`data_collection`) 用于录制键盘遥操作过程中的状态-动作对，生成可用于训练 VLA (Vision-Language-Action) 模型的数据集。

**关键设计目标**：与 `vla_control` 模块的 action 格式完全一致，确保采集的数据可以直接用于训练，且训练出的模型可以直接部署。

---

## Action 格式规范

### 整体结构

```python
action: np.ndarray  # shape: (7,)
# [delta_pos(3), delta_axisangle(3), gripper_cmd(1)]
```

- **`action[0:3]`**: 位置增量 `[dx, dy, dz]`（米），**基座坐标系**
- **`action[3:6]`**: 旋转增量 `[droll, dpitch, dyaw]`（弧度），**轴角表示，基座坐标系**
- **`action[6]`**: 夹爪控制信号（意图信号，非物理值）

---

## 夹爪信号映射

### 采集时 (key_control.py)

```python
if self.gripper_target < 0.04:
    delta[6] = 1.0   # 闭合意图
else:
    delta[6] = -1.0  # 打开意图
```

- **`gripper_target < 0.04`**（如 0.0 闭合）→ 输出 `1.0`（正数）
- **`gripper_target >= 0.04`**（如 0.04 半开、0.08 全开）→ 输出 `-1.0`（负数）

### 执行时 (vla_control/transform_action)

```python
ta[6] = 0.0 if ta[6] >= 0 else 0.08
```

- **`action[6] >= 0`** → 映射为 `0.0`（闭合，开口 0m）
- **`action[6] < 0`** → 映射为 `0.08`（打开，开口 0.08m）

### 为什么这样设计？

VLA 模型只需学习**符号信号**（正/负），不需要输出精确的物理值。这样简化了模型的学习任务，同时保证了控制的鲁棒性。

---

## 旋转坐标系与轴定义

### 坐标系：**基座坐标系（Base Frame / World Frame）**

旋转增量在基座坐标系中表达，使用**左乘**方式叠加：

```python
delta_rot = Rotation.from_rotvec(delta_axisangle).as_matrix()
target.rotation = delta_rot @ current.rotation  # 左乘
```

### 旋转轴与按键映射

| 轴 | 按键（正向） | 按键（负向） | 变量名 | 物理含义 |
|---|---|---|---|---|
| **X 轴** | `u` | `o` | `droll` | 绕基座 X 轴旋转 |
| **Y 轴** | `i` | `k` | `dpitch` | 绕基座 Y 轴旋转 |
| **Z 轴** | `l` | `j` | `dyaw` | 绕基座 Z 轴旋转 |

### 正方向定义

遵循**右手螺旋定则**：
- 右手拇指指向轴的正方向
- 四指弯曲的方向即为旋转的正方向

```python
droll  = (int('u' in keys) - int('o' in keys)) * MAX_DELTA_ROT * speed
dpitch = (int('i' in keys) - int('k' in keys)) * MAX_DELTA_ROT * speed
dyaw   = (int('l' in keys) - int('j' in keys)) * MAX_DELTA_ROT * speed

delta_rotvec = np.array([droll, dpitch, dyaw])
```

- 按 `u` → `droll > 0` → 绕 X 轴**正向**旋转
- 按 `o` → `droll < 0` → 绕 X 轴**负向**旋转
- 其他轴同理

---

## 位置控制

### 坐标系：**基座坐标系**

```python
dx = (int('w' in keys) - int('s' in keys)) * MAX_DELTA_POS * speed
dy = (int('a' in keys) - int('d' in keys)) * MAX_DELTA_POS * speed
dz = (int('q' in keys) - int('e' in keys)) * MAX_DELTA_POS * speed
```

| 方向 | 按键（正向） | 按键（负向） | 坐标轴 |
|---|---|---|---|
| **X 方向** | `w` | `s` | 基座 X 轴 |
| **Y 方向** | `a` | `d` | 基座 Y 轴 |
| **Z 方向** | `q` | `e` | 基座 Z 轴 |

---

## 完整键盘映射

### 位置控制
- `w` / `s`：沿基座 X 轴 正向/负向 移动
- `a` / `d`：沿基座 Y 轴 正向/负向 移动
- `q` / `e`：沿基座 Z 轴 正向/负向 移动

### 旋转控制
- `u` / `o`：绕基座 X 轴 正向/负向 旋转（roll）
- `i` / `k`：绕基座 Y 轴 正向/负向 旋转（pitch）
- `l` / `j`：绕基座 Z 轴 正向/负向 旋转（yaw）

### 夹爪控制
- `g`：闭合夹爪（gripper_target = 0.0）
- `h`：打开夹爪（gripper_target = 0.08）
- `f`：半开夹爪（gripper_target = 0.04）

### 其他控制
- `r`：复位到 home 位姿
- `+` / `=`：增加速度倍率
- `-` / `_`：减小速度倍率
- `ESC`：退出程序

### 录制控制
- `1`：开始录制一段轨迹
- `2`：结束当前轨迹（保存）

---

## 速度配置

### 三档速度控制

```python
_speed_levels = [0.4, 0.7, 1.0]  # 40%, 70%, 100%
```

- **默认档**：`0.7`（70%）
- **最大线速度**：`0.1 m/s`
- **最大角速度**：`π/4 rad/s`（45°/s）

### 每步最大增量（20Hz 控制频率下）

```python
MAX_DELTA_POS = 0.1 * 0.05 = 0.005 m      # 5mm
MAX_DELTA_ROT = (π/4) * 0.05 ≈ 0.039 rad  # ≈ 2.25°
```

实际增量 = 最大增量 × 速度倍率

---

## 控制频率

### key_control.py（采集端）
- **发布频率**：`20Hz`（`PUBLISH_DT = 0.05s`）
- **传感器消息发布**：每 50ms 发布一次 dynamic skill 控制指令

### data_recorder.py（录制端）
- **采集频率**：`10Hz`（`COLLECT_HZ = 10.0`，与 vla_control 保持一致）
- **空动作过滤**：位置和旋转增量小于阈值时不记录

```python
ACTION_THRESH_POS = 0.0005   # 位置阈值 0.5mm
ACTION_THRESH_ROT = 0.005    # 旋转阈值 0.005 rad ≈ 0.29°
```

---

## State 格式

```python
state: np.ndarray  # shape: (8,)
# [pos(3), rotvec(3), finger1(1), finger2(1)]
```

- **`state[0:3]`**: 末端位置 `[x, y, z]`（米），基座坐标系
- **`state[3:6]`**: 末端姿态轴角 `[ax, ay, az]`（弧度），基座坐标系
- **`state[6]`**: 手指1位置（半宽，米）
- **`state[7]`**: 手指2位置（负半宽，米）

---

## Transform Action 变换逻辑

在 vla_control 执行时，会应用以下变换：

```python
def transform_action(action: np.ndarray) -> np.ndarray:
    ta = action.copy()
    
    # 位置：裁剪 + 缩放
    ta[:3] = np.clip(ta[:3], -POS_CLIP, POS_CLIP) * POS_SCALE
    # POS_CLIP = 1.0, POS_SCALE = 0.1/3 * 0.1 / 1.0 ≈ 0.0033
    
    # 旋转：裁剪 + 缩放
    ta[3:6] = np.clip(ta[3:6], -ROT_CLIP, ROT_CLIP) * ROT_SCALE
    # ROT_CLIP = 0.1, ROT_SCALE = (π/2)/3 * 0.1 / 0.1 ≈ 0.524
    
    # 夹爪：二值化
    ta[6] = 0.0 if ta[6] >= 0 else 0.08
    
    return ta
```

### 参数说明

- **`POS_CLIP = 1.0`**：位置输入截断范围 `[-1.0, 1.0]` 米
- **`ROT_CLIP = 0.1`**：旋转输入截断范围 `[-0.1, 0.1]` 弧度
- **`POS_MAX_VEL = 0.1/3 m/s`**：末端位置最大速度
- **`ROT_MAX_VEL = π/2/3 rad/s`**：末端旋转最大速度
- **`CONTROL_DT = 0.1s`**：控制周期（10Hz）
- **`GRIPPER_SPEED = 0.08 m/s`**：夹爪运动速度

---

## 数据保存格式

### 目录结构

```
collected/
└── task_name/
    ├── epo_1/
    │   ├── cam1.mp4      # 外部相机视频
    │   ├── cam2.mp4      # 腕部相机视频
    │   └── data.csv      # 状态-动作数据
    ├── epo_2/
    │   └── ...
    └── ...
```

### CSV 格式

- **第1行**：超参数 JSON（任务名、频率、帧数等）
- **第2行**：列名 `state_0, state_1, ..., state_7, action_0, ..., action_6`
- **第3行起**：数据（每行 15 个值）

---

## 与 VLA Control 的对齐

### 完全一致的方面

1. **Action 格式**：`(7,)` 向量，含义完全相同
2. **旋转坐标系**：均为基座坐标系，左乘叠加
3. **旋转轴定义**：`[X, Y, Z]` 轴对应 `[droll, dpitch, dyaw]`
4. **夹爪逻辑**：通过 transform_action 统一处理
5. **State 格式**：`(8,)` 向量，包含位置、姿态、夹爪

### 差异（正常）

1. **控制频率**：采集端 20Hz，录制端 10Hz（与控制循环对齐）
2. **Action 来源**：采集端来自键盘，执行端来自 VLA 模型
3. **安全检查**：执行端有额外安全检查，采集端无

---

## 使用流程

### 1. 启动数据采集

```bash
cd /home/k324/franka_my_code/vla4desk
./start_data_collector.sh
```

或手动启动：

```bash
cd src/data_collection
python data_recorder.py --task_name my_task --collect_hz 10
```

### 2. 录制数据

1. 按 `1` 开始录制
2. 使用键盘控制机械臂完成任务
3. 按 `2` 结束录制（自动保存）
4. 重复步骤 1-3 收集多个 episode
5. 按 `ESC` 退出

### 3. 检查数据

```bash
ls collected/my_task/
# epo_1/  epo_2/  epo_3/  ...
```

---

## 常见问题

### Q: 为什么夹爪信号要用 -1.0/1.0 而不是 0.0/0.08？

A: 为了让 VLA 模型学习**意图**而非精确值。模型只需输出正负号，实际的物理值由 `transform_action` 统一映射。这简化了学习任务，提高了泛化能力。

### Q: 旋转为什么用基座坐标系而不是末端坐标系？

A: 基座坐标系是绝对坐标系，不随末端姿态变化，更容易学习和预测。VLA 模型输出的旋转增量可以直接在基座坐标系中应用，无需坐标变换。

### Q: 采集频率 10Hz 和控制频率 20Hz 不一致会有问题吗？

A: 不会。key_control 以 20Hz 发布控制指令，保证机械臂运动平滑；data_recorder 以 10Hz 采样录制，与 vla_control 的控制循环对齐，确保训练和部署的一致性。

---

## 相关文件

- `key_control.py`：键盘遥操作控制器，生成 state 和 action
- `data_recorder.py`：数据采集器，录制视频和 CSV
- `../vla_control/franka_env.py`：VLA 执行环境，包含 transform_action
- `../vla_control/coordinator.py`：主控制器，协调推理和执行

---

**最后更新**：2026-04-09

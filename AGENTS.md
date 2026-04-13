# vla4desk 项目规则

## 适用范围

- 本文件适用于整个仓库。
- 本项目中 `state` 和 `action` 是跨模块共享接口，不是某个文件内部的临时约定。
- `src/vla_control`、`src/data_collection`、`src/client`、回放链路、前端遥测、落盘数据都必须遵守同一套定义。

## 架构解耦原则

- 本项目以 `FrankaEnv` 作为核心运行时边界。
- `FrankaEnv` 统一负责两件事：观测采集与动作执行。
- 其他模块不得直接接管底层机器人执行细节，而应通过 `FrankaEnv` 的公开接口协作。
- `Coordinator` 只负责编排、状态机、推理调度、前端推流与遥测，不直接实现机器人控制。
- `data_collection` 只负责输入采样、录制、回放与落盘，不直接实现机器人控制。
- `client` 只负责推理通信与策略接口，不直接实现机器人控制。
- 模块之间的主要耦合面应保持为稳定的 `state` / `action` 定义和 `FrankaEnv` 公开接口，而不是共享底层控制实现。

## 一、`state` 的统一定义

- `state` 必须是 `float64` 类型、形状为 `(8,)` 的一维向量。
- `state` 的定义固定为：

```python
state = [pos(3), rotvec(3), finger1(1), finger2(1)]
```

- 各下标的含义固定如下：

```python
state[0] = eef_x
state[1] = eef_y
state[2] = eef_z
state[3] = rotvec_x
state[4] = rotvec_y
state[5] = rotvec_z
state[6] = finger1
state[7] = finger2
```

- 语义要求：
  - `state[0:3]` 表示末端执行器在基座坐标系下的位置，单位为米。
  - `state[3:6]` 表示末端执行器姿态的旋转向量（`rotvec`），单位为弧度。
  - `state[6]`、`state[7]` 表示夹爪两侧 finger 位置。
- `state` 的运行时标准来源是 `FrankaEnv.get_robot_state_vector()`。
- 不允许某个模块把同名 `state` 改成别的维度、别的顺序或别的姿态表示法。

## 二、`action` 的统一定义

- `action` 必须是 `float64` 类型、形状为 `(7,)` 的一维向量。
- `action` 的定义固定为：

```python
action = [delta_pos(3), delta_axisangle(3), gripper_cmd(1)]
```

- 各下标的含义固定如下：

```python
action[0] = delta_x
action[1] = delta_y
action[2] = delta_z
action[3] = delta_rotvec_x
action[4] = delta_rotvec_y
action[5] = delta_rotvec_z
action[6] = gripper_cmd
```

- 语义要求：
  - `action[0:3]` 表示本控制拍的末端平移增量，单位为米。
  - `action[3:6]` 表示本控制拍的末端旋转增量，表示形式为 `rotvec`，单位为弧度。
  - `action[6]` 表示夹爪命令语义，不表示物理宽度，不表示连续开合比例。
- 所有在线推理输出、手动采集输出、回放输入、落盘数据、遥测展示都必须保留这一维度和索引布局。

## 三、`action[6]` 的取值与意义

- `action[6]` 的仓库级语义是“夹爪开合意图”，不是夹爪宽度。
- 当前执行侧的判定规则固定为：

```python
if action[6] >= 0:
    target_gripper_width = 0.0
else:
    target_gripper_width = 0.08
```

- 因此当前项目中，`action[6]` 的意义必须解释为：
  - `action[6] >= 0`：表示“闭合夹爪”
  - `action[6] < 0`：表示“打开夹爪”
- 当前数据采集侧的标准发值是：
  - `1.0`：闭合夹爪
  - `-1.0`：打开夹爪
- 允许某些上游策略输出别的实数值，但只要进入本仓库执行链路，就必须按“符号”解释，而不是按大小解释：
  - 任意非负值都等价于“闭合”
  - 任意负值都等价于“打开”
- 当前项目中，`action[6]` 不支持三态、半开比例、连续宽度控制。
- 即使输入设备内部存在“半开”概念，只要编码到 `action[6]`，也仍然只能落到上述二值语义之一。
- 不允许在某处把 `action[6] = 0.04` 解释为“目标宽度 4cm”，在另一处又把它解释为“闭合命令”。

## 四、坐标系与表示法规则

- `state[3:6]` 和 `action[3:6]` 在整个仓库中必须统一使用 `rotvec` 表示。
- `action[3:6]` 的最终语义必须是“基座坐标系下的旋转增量 rotvec”。
- 输入设备可以为了操控手感先按末端局部坐标系理解旋转输入，但在生成最终 `action` 前必须转换到基座坐标系。
- 因此：
  - 写入数据集的 `action[3:6]` 必须是基座系 rotvec；
  - 发给 `FrankaEnv.enqueue_action()` 的 `action[3:6]` 必须是基座系 rotvec；
  - 回放读取并重新执行的 `action[3:6]` 必须按基座系 rotvec 解释。
- 不允许同样名为 `action` 的字段，在采集时表示末端系旋转、在执行时表示基座系旋转。

## 五、尺度与执行语义

- `action` 在逻辑上表达“本控制拍的目标增量”。
- 对于推理链路：
  - `FrankaEnv.transform_action()` 会对 `action[:3]`、`action[3:6]` 做裁剪和缩放；
  - 然后再把 `action[6]` 映射成夹爪目标宽度。
- 对于采集/回放链路：
  - 若调用 `enqueue_action(..., transform=False)`，表示该 `action[:6]` 已经是执行尺度，不再重复缩放；
  - 但 `action[6]` 仍然按“非负闭合、负值打开”的规则解释。
- 因此 `transform=False` 只影响位姿增量的缩放，不改变 `action[6]` 的二值语义。

## 六、文档与数据同步要求

- `README.md` 中对 `state` / `action` 的描述必须与本文件一致。
- `src/vla_control/TECHNICAL_DOC.md` 与 `src/data_collection/TECHNICAL_DOC.md` 中对 `state` / `action` 的描述必须与本文件一致。
- 任何涉及以下内容的修改，都必须同步更新代码和文档：
  - 维度
  - 下标定义
  - 坐标系
  - 单位
  - `action[6]` 的取值规则
  - 落盘数据语义

## 七、禁止事项

- 不允许把 `state` 从 `(8,)` 私自改成别的长度。
- 不允许把 `action` 从 `(7,)` 私自改成别的长度。
- 不允许在某一条链路中单独改变字段顺序。
- 不允许把 `state[3:6]` 或 `action[3:6]` 改成欧拉角、四元数或别的表示法，而不做仓库级迁移。
- 不允许把 `action[6]` 当作连续宽度值使用。
- 不允许让采集、推理、执行、回放对同一个 `action[6]` 使用不同解释。
- 不允许代码、落盘数据、遥测、文档之间出现静默漂移。

## 八、修改要求

- 如果任务没有明确要求修改 schema，则必须保持当前定义不变：
  - `state`: `(8,)`
  - `action`: `(7,)`
- 如果任务明确要求修改 `state` 或 `action`，提交内容里必须同时明确：
  - 新的 shape
  - 每个下标的意义
  - 坐标系
  - 单位
  - `action[6]` 的新语义
  - 对历史数据、回放链路、前端遥测、推理接口的影响

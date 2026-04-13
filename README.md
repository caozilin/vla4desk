# vla4desk

Franka Emika 机械臂 + openpi 云端推理的本地控制系统。

## 结构

```
vla4desk/
├── src/
│   ├── client/                # openpi-client（websocket 推理客户端）
│   ├── vla_control/
│   │   ├── coordinator.py      # 主控制器：状态机 + 推理编排 + Web 服务
│   │   ├── franka_env.py       # FrankaEnvironment：obs 采集 + action 执行
│   │   └── TECHNICAL_DOC.md    # 执行链技术文档
│   └── data_collection/
│       └── TECHNICAL_DOC.md    # 数据采集技术文档
├── static/                 # Web 前端静态文件
│   └── index.html
└── requirements.txt
```

## Docker 环境

本项目在 `franka` 容器内运行。当前仓库位于宿主机 `/media/czl/sata/franka_my_code/vla4desk`，并挂载到容器内 `/root/Documents/my_code/vla4desk`。

### 进入容器

```bash
docker exec -it franka bash
```

### 安装依赖（首次或更新后执行）

```bash
docker exec -it franka bash -c "cd /root/Documents/my_code/vla4desk && pip install -r requirements.txt"
```

> `pyrealsense2` 需要 D435 USB 设备已透传到容器，且宿主机已安装 RealSense SDK。
> `frankapy` / `autolab_core` 已在容器内预装，无需通过 pip 安装。

## 启动

### 本地控制器（在宿主机仓库目录下执行，实际命令会进入容器）

```bash
# 无机械臂模式（仅相机 + 前端）
docker exec -it franka bash -c "source /opt/ros/humble/setup.bash && source /root/Documents/franka-interface/ros2_ws/install/setup.bash && cd /root/Documents/my_code/vla4desk && python src/vla_control/coordinator.py --no_robot"

# 完整模式（机械臂 + 推理服务）
docker exec -it franka bash -c "source /opt/ros/humble/setup.bash && source /root/Documents/franka-interface/ros2_ws/install/setup.bash && cd /root/Documents/my_code/vla4desk && python src/vla_control/coordinator.py"
```

浏览器打开 `http://localhost:8080`

- 相机未连接：显示全黑帧，其余功能正常
- 机械臂未连接：加 `--no_robot` 参数跳过 FrankaArm 初始化
- 推理服务未开启：启动时重试 3 次后跳过，点击 Start 时自动回到 IDLE
- 传给 Docker 内 Python 的路径参数请使用仓库相对路径（如 `collected/...`）或容器内路径（`/root/Documents/my_code/vla4desk/...`），不要传宿主机绝对路径

## 文档

- 执行链路与控制说明：`src/vla_control/TECHNICAL_DOC.md`
- 数据采集说明：`src/data_collection/TECHNICAL_DOC.md`

### 可选参数

```bash
python src/vla_control/coordinator.py \
  --host <云端IP> \
  --port 8000 \
  --cam1_serial <D435序列号> \
  --cam2_serial <D435序列号> \
  --web_port 8080
```

### 云端推理服务（在 openpi 目录）

```bash
python scripts/serve_policy.py --env libero --policy.config <config> --policy.dir <ckpt_dir>
```

### CSV 回放与重记录

```bash
# 按 episode 目录回放，并将 replay 时的 state/action 重新保存为新 CSV
python src/data_collection/trajectory_replay_recorder.py \
  --episode collected/<task>/epo_<n> \
  --output_dir collected/<task>/replay_epo_<n>

# 无机械臂模式下联调 CSV 流程
python src/data_collection/trajectory_replay_recorder.py \
  --episode collected/<task>/epo_<n> \
  --no_robot
```

如果通过 `./start_trajectory_replay.sh` 在 Docker 中启动，`--episode` 同样应传 `collected/...` 这类仓库相对路径，或容器内路径 `/root/Documents/my_code/vla4desk/collected/...`。

### 数据采集

```bash
# 默认录到 collected/default
./start_data_collector.sh

# 显式录到常用任务目录名
./start_data_collector.sh --task_name simple_pick_place
```

## 轴角约束

- 已在 `FrankaEnv` 中实现 `state[3:6]` 的连续性约束：无历史帧时使用单帧 canonical rotvec；有历史帧时，在等价轴角表示中选择与上一帧最接近的分支，避免 `±pi` 附近的跳变。

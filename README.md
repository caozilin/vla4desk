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

本项目当前默认在 `franka_vla4desk` 容器内运行。宿主机代码根目录 `/home/k324/franka_my_code` 会挂载到容器内 `/workspace/franka_my_code`，其中本仓库路径为 `/workspace/franka_my_code/vla4desk`。

### 进入容器

```bash
/home/k324/franka_my_code/vla4desk/run_franka_vla4desk.sh
docker exec -it franka_vla4desk bash
# 或通过宿主机 uid/gid 身份进入
/home/k324/franka_my_code/vla4desk/exec_franka_vla4desk.sh 'bash'
```

### Python 依赖

```bash
/home/k324/franka_my_code/vla4desk/build_franka_vla4desk_image.sh
```

- `requirements.txt` 会在构建 `franka_emika_vla4desk:latest` 时自动安装进镜像。
- 容器启动后会自动执行 `setup_franka_vla4desk_env.sh`，对挂载进来的本地 `frankapy` 做 `pip install -e`，并检查 `frankapy/ros2_ws`、`franka-interface/ros2_ws` 是否已可 source。
- 如果你修改了 `requirements.txt`，重跑一次 `build_franka_vla4desk_image.sh`，然后重新启动容器即可。
- `pyrealsense2` 需要 D435 USB 设备已透传到容器，且宿主机已安装 RealSense SDK。
- `franka-interface/build` 这套 C++ 二进制当前仍然依赖宿主机现有构建产物；容器会自动把对应 `LD_LIBRARY_PATH` 接起来。

## 启动

### 本地控制器（在宿主机仓库目录下执行，实际命令会进入容器）

```bash
# 无机械臂模式（仅相机 + 前端）
/home/k324/franka_my_code/vla4desk/start_vla4desk.sh --no_robot

# 完整模式（机械臂 + 推理服务）
/home/k324/franka_my_code/vla4desk/start_vla4desk.sh
```

浏览器打开 `http://localhost:8080`

- 相机未连接：显示全黑帧，其余功能正常
- 机械臂未连接：加 `--no_robot` 参数跳过 FrankaArm 初始化
- 推理服务未开启：启动时重试 3 次后跳过，点击 Start 时自动回到 IDLE
- 传给 Docker 内 Python 的路径参数请使用仓库相对路径（如 `collected/...`）或容器内路径（`/workspace/franka_my_code/vla4desk/...`），不要传宿主机绝对路径

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

### JSON 回放与重记录

```bash
# 按 episode 目录回放，并将 replay 时的 state/action 重新保存为新 JSON
python src/data_collection/trajectory_replay_recorder.py \
  --episode collected/<task>/epo_<n> \
  --output_dir collected/<task>/replay_epo_<n>

# 无机械臂模式下联调 JSON 流程
python src/data_collection/trajectory_replay_recorder.py \
  --episode collected/<task>/epo_<n> \
  --no_robot
```

如果通过 `./start_trajectory_replay.sh` 在 Docker 中启动，`--episode` 同样应传 `collected/...` 这类仓库相对路径，或容器内路径 `/workspace/franka_my_code/vla4desk/collected/...`。

### 数据采集

```bash
# 默认录到 collected/default
/home/k324/franka_my_code/vla4desk/start_vla4desk_data_collector.sh

# 显式录到常用任务目录名
/home/k324/franka_my_code/vla4desk/start_vla4desk_data_collector.sh --task_name simple_pick_place
```

批量给已采集 JSON 写入 prompt：

```sh
/home/k324/franka_my_code/vla4desk/start_write_prompts_to_json.sh collected/strawberry_on_place
```

如果已经进入容器，也可以直接执行：

```bash
python -B src/data_collection/write_prompts_to_json.py collected/strawberry_on_place
```

- 只需要传任务目录
- 脚本会读取该目录里的 `prompts.txt`，例如 `collected/strawberry_on_place/prompts.txt`
- 脚本会递归写入该目录下各 episode 的 `data.json`
- Docker 脚本里推荐传 `collected/...`、`src/data_collection/...` 这种容器内仓库相对路径，不要传宿主机绝对路径

如果要遍历整个 `collected`，并且只给 `prompt` 为空的 JSON 补写：

```sh
/home/k324/franka_my_code/vla4desk/start_write_prompts_to_json.sh --fill-empty-under-collected collected
```

- 这个模式会遍历 `collected` 下所有带 `prompts.txt` 的任务目录
- 每个任务目录都使用自己的 `prompts.txt`
- 只会改 `prompt == ""` 的 `data.json`，已有 prompt 的不会覆盖

## 轴角约束

- 已在 `FrankaEnv` 中实现 `state[3:6]` 的连续性约束：无历史帧时使用单帧 canonical rotvec；有历史帧时，在等价轴角表示中选择与上一帧最接近的分支，避免 `±pi` 附近的跳变。

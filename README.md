# vla4desk

Franka Emika 机械臂 + openpi 云端推理的本地控制系统。

## 结构

```
vla4desk/
├── src/
│   ├── coordinator.py      # 主控制器：20Hz 控制循环 + FastAPI 服务
│   └── franka_env.py       # FrankaEnvironment：obs 采集 + action 执行
├── client/                 # openpi-client（websocket 推理客户端）
├── static/                 # Web 前端静态文件
│   └── index.html
└── requirements.txt
```

## Docker 环境

本项目在 `franka` 容器内运行。宿主机目录 `~/franka_my_code` 映射到容器内 `/root/Documents/my_code`。

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

### 本地控制器（在宿主机 vla4desk 目录下执行）

```bash
# 无机械臂模式（仅相机 + 前端）
docker exec -it franka bash -c "source /opt/ros/humble/setup.bash && source /root/Documents/franka-interface/ros2_ws/install/setup.bash && cd /root/Documents/my_code/vla4desk/src && python coordinator.py --no_robot"

# 完整模式（机械臂 + 推理服务）
docker exec -it franka bash -c "source /opt/ros/humble/setup.bash && source /root/Documents/franka-interface/ros2_ws/install/setup.bash && cd /root/Documents/my_code/vla4desk/src && python coordinator.py"
```

浏览器打开 `http://localhost:8080`

- 相机未连接：显示全黑帧，其余功能正常
- 机械臂未连接：加 `--no_robot` 参数跳过 FrankaArm 初始化
- 推理服务未开启：启动时重试 3 次后跳过，点击 Start 时自动回到 IDLE

### 可选参数

```bash
python coordinator.py \
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

# vla4desk

Franka Emika 机械臂 + openpi 云端推理的本地控制系统。

## 结构

```
vla4desk/
├── coordinator.py      # 主控制器：20Hz 控制循环 + FastAPI 服务
├── franka_env.py       # FrankaEnvironment：obs 采集 + action 执行
├── camera.py           # D435 双相机采集（pyrealsense2 直连）
├── static/             # Web 前端静态文件
│   └── index.html
└── requirements.txt
```

## 启动

```bash
# 云端（在 openpi 目录）
python scripts/serve_policy.py --env libero --policy.config <config> --policy.dir <ckpt_dir>

# 本地（在 vla4desk 目录）
python coordinator.py --host <cloud_ip> --port 8000
```

浏览器打开 `http://localhost:8080`

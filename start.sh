#!/bin/bash
# 用法：
#   ./start.sh          # 完整模式（机械臂 + 推理服务）
#   ./start.sh --no_robot  # 无机械臂模式（仅相机 + 前端）

ROS_SETUP="source /opt/ros/humble/setup.bash && source /root/Documents/franka-interface/ros2_ws/install/setup.bash"
CMD="cd /root/Documents/my_code/vla4desk/src && python coordinator.py $*"

docker exec -it franka bash -c "$ROS_SETUP && $CMD"

#!/bin/bash
# 轨迹复现 - Docker 版本
#
# 用法：
#   ./start_trajectory_replay.sh --episode collected/simple_pick_place/epo_9
#   ./start_trajectory_replay.sh --episode /root/Documents/my_code/vla4desk/collected/simple_pick_place/epo_9
#   ./start_trajectory_replay.sh --task default --epo 14
#   ./start_trajectory_replay.sh --task simple_pick_place --epo 1 --speed 2.0
#   ./start_trajectory_replay.sh --episode collected/simple_pick_place/epo_1 --no_robot
#
# 注意：
#   该脚本在 Docker 容器内执行，传给 --episode 的路径必须是容器可见路径，
#   或者直接使用相对仓库路径 collected/...；不要传宿主机绝对路径。

# ROS 环境配置（与项目其他脚本一致）
ROS_SETUP="source /opt/ros/humble/setup.bash && source /root/Documents/franka-interface/ros2_ws/install/setup.bash"

# 构建命令
CMD="cd /root/Documents/my_code/vla4desk/src/data_collection && python trajectory_replay_recorder.py $*"

# 在 Docker 容器中执行
docker exec -it franka bash -c "$ROS_SETUP && $CMD"

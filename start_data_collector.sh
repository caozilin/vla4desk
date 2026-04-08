#!/bin/bash
# 用法：
#   ./start_data_collector.sh                    # 默认参数
#   ./start_data_collector.sh --task_name pick_place  # 指定任务

ROS_SETUP="source /opt/ros/humble/setup.bash && source /root/Documents/franka-interface/ros2_ws/install/setup.bash"
CMD="cd /root/Documents/my_code/vla4desk/src/data_collection && python data_recorder.py $*"

docker exec -it franka bash -c "$ROS_SETUP && $CMD"

#!/bin/bash
# 启动 vla4desk 数据采集程序（遥操作）
#
# 用法：
#   xhost +SI:localuser:root
#   ./start_vla4desk_data_collector.sh --input_device keyboard
#   ./start_vla4desk_data_collector.sh --input_device ps4
#   ./start_vla4desk_data_collector.sh --task_name final_clean --input_device ps4
#   ./start_vla4desk_data_collector.sh --task_name simple_pick_place --input_device ps4 --joystick_index 0
#   ./start_vla4desk_data_collector.sh --task_name pick_place --input_device keyboard --rotation_input_frame base
#   ./start_vla4desk_data_collector.sh --input_device ps4 --resize_resolution 224
#
# 参数说明：
#   --task_name <name>           任务名称（默认: default）
#   --input_device <device>      输入设备: keyboard 或 ps4（默认: keyboard）
#   --joystick_index <index>     PS4 手柄索引（默认: 0）
#   --rotation_input_frame <frame> 旋转输入坐标系: base 或 end_effector（默认: end_effector）
#   --resize_resolution <res>    图像分辨率（默认: 224）
#
# 说明：
#   - 启动遥操作数据采集，支持键盘或 PS4 手柄控制
#   - 需要确保 franka 容器已启动并挂载了相机设备
#   - 需要先启动 Franka 三件套（start_interface_triplet.sh）
#   - 数据保存在 collected/<task_name>/ 目录下

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-franka}"
WORKSPACE_ROOT="/root/Documents/my_code"
FRANKA_INTERFACE_LD_PATH="${WORKSPACE_ROOT}/franka-interface/build/franka-interface:${WORKSPACE_ROOT}/franka-interface/build/franka-interface/proto:${WORKSPACE_ROOT}/franka-interface/build/libfranka:${WORKSPACE_ROOT}/franka-interface/build/franka-interface-common:/usr/local/lib"
FRANKAPY_PYTHONPATH="${WORKSPACE_ROOT}/frankapy"

# 获取宿主机用户ID和组ID
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

docker exec -it \
  -e WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
  -e FRANKAPY_PYTHONPATH="${FRANKAPY_PYTHONPATH}" \
  -e FRANKA_INTERFACE_LD_PATH="${FRANKA_INTERFACE_LD_PATH}" \
  -e DISPLAY="${DISPLAY:-:1}" \
  "${CONTAINER_NAME}" bash -lc "
    export HOME=/root
    export LD_LIBRARY_PATH=\"${FRANKA_INTERFACE_LD_PATH}:\${LD_LIBRARY_PATH:-}\"
    export PYTHONPATH=\"${FRANKAPY_PYTHONPATH}:\${PYTHONPATH:-}\"
    source /opt/ros/humble/setup.bash
    source ${WORKSPACE_ROOT}/franka-interface/install/setup.bash
    cd ${WORKSPACE_ROOT}/vla4desk/src/data_collection
    python data_recorder.py $*
  "

# 修复 collected 和 logs 文件夹权限
echo "[INFO] 修复文件夹权限..."
docker exec "${CONTAINER_NAME}" bash -lc "
  chown -R ${HOST_UID}:${HOST_GID} ${WORKSPACE_ROOT}/vla4desk/collected 2>/dev/null || true
  chown -R ${HOST_UID}:${HOST_GID} ${WORKSPACE_ROOT}/vla4desk/logs 2>/dev/null || true
  chmod -R u+rw ${WORKSPACE_ROOT}/vla4desk/collected 2>/dev/null || true
  chmod -R u+rw ${WORKSPACE_ROOT}/vla4desk/logs 2>/dev/null || true
" 2>/dev/null || true
echo "[OK] 权限修复完成"

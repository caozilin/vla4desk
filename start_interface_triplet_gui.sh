#!/bin/bash
# 启动 Franka 三件套（弹出3个终端窗口）
#
# 用法：
#   ./start_interface_triplet_gui.sh
#
# 说明：
#   - 在宿主机弹出3个终端窗口运行 franka 三件套
#   - 关闭终端窗口会自动结束对应进程
#   - 需要先确保 franka 容器正在运行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE_ROOT="${HOST_CODE_ROOT:-/home/k324/franka_my_code}"
CONTAINER_NAME="${CONTAINER_NAME:-franka}"
IMAGE_NAME="${IMAGE_NAME:-franka:vla4desk}"

# 固定参数（之前成功运行的配置）
ROBOT_IP="172.16.0.2"
ROBOT_NUMBER="1"
WITH_GRIPPER="0"  # 0=新夹爪模式
LOG_FRANKA_INTERFACE="0"
STOP_ON_ERROR="0"

WORKSPACE_ROOT="/root/Documents/my_code"
FRANKA_INTERFACE_ROOT="${WORKSPACE_ROOT}/franka-interface"
LOG_DIR="${WORKSPACE_ROOT}/vla4desk/logs/franka_backend"

# 检测终端
TERMINAL=""
if command -v gnome-terminal &> /dev/null; then
    TERMINAL="gnome-terminal"
elif command -v konsole &> /dev/null; then
    TERMINAL="konsole"
elif command -v xterm &> /dev/null; then
    TERMINAL="xterm"
elif command -v terminator &> /dev/null; then
    TERMINAL="terminator"
else
    echo "[ERROR] No supported terminal emulator found!"
    echo "Please install gnome-terminal, konsole, xterm, or terminator"
    exit 1
fi

# 确保容器运行
if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1 || [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "[INFO] starting container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -dit \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --network host \
    --cap-add IPC_LOCK \
    --cap-add SYS_NICE \
    --security-opt label=disable \
    -e DISPLAY="${DISPLAY:-:1}" \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME=/root \
    -v "${HOST_CODE_ROOT}:${WORKSPACE_ROOT}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/input:/dev/input \
    -v /run/udev:/run/udev:ro \
    -w ${WORKSPACE_ROOT}/vla4desk \
    "${IMAGE_NAME}" \
    bash >/dev/null
fi

echo "[INFO] 清理旧的 Franka 三件套进程..."
docker exec "${CONTAINER_NAME}" bash -lc "
set +e
LOG_DIR='${LOG_DIR}'

# 清理 FastDDS 残留 SHM 文件，避免 /dev/shm 被 fastrtps_* 撑满
rm -f /dev/shm/fastrtps_* /dev/shm/fastrtps_*_el >/dev/null 2>&1 || true

# 仅按命令行清理，且使用 [f] 写法避免命中当前清理命令自身
pkill -f '[f]ranka_interface --robot_ip' >/dev/null 2>&1 || true
pkill -f '[f]ranka_ros_interface.launch.py' >/dev/null 2>&1 || true
pkill -f '[f]ranka_ros_interface_execute_skill_action_server' >/dev/null 2>&1 || true
pkill -f '[f]ranka_gripper.launch.py' >/dev/null 2>&1 || true
pkill -f '[f]ranka_gripper_node_' >/dev/null 2>&1 || true

sleep 1
"
echo "[OK] 旧进程清理完成"

echo "=========================================="
echo "Starting Franka Interface Triplet (GUI)"
echo "=========================================="
echo "Terminal: ${TERMINAL}"
echo "Robot IP: ${ROBOT_IP}"
echo "Robot Number: ${ROBOT_NUMBER}"
echo "With Gripper: ${WITH_GRIPPER} (0=新夹爪, 1=旧夹爪)"
echo ""

# 环境变量设置
ENV_SETUP="source /opt/ros/humble/setup.bash && source ${FRANKA_INTERFACE_ROOT}/install/setup.bash && export LD_LIBRARY_PATH=${FRANKA_INTERFACE_ROOT}/build/franka-interface:${FRANKA_INTERFACE_ROOT}/build/franka-interface/proto:${FRANKA_INTERFACE_ROOT}/build/libfranka:${FRANKA_INTERFACE_ROOT}/build/franka-interface-common:/usr/local/lib:\$LD_LIBRARY_PATH"

# 终端1: franka_interface
echo "[1/3] Starting franka_interface..."
if [ "$TERMINAL" == "gnome-terminal" ]; then
    gnome-terminal --title="Franka Interface" -- bash -c "
        echo '=========================================='
        echo 'Franka Interface (底层接口)'
        echo 'Robot IP: ${ROBOT_IP}, With Gripper: ${WITH_GRIPPER}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && cd ${FRANKA_INTERFACE_ROOT}/build && ./franka_interface --robot_ip ${ROBOT_IP} --with_gripper ${WITH_GRIPPER} --log ${LOG_FRANKA_INTERFACE} --stop_on_error ${STOP_ON_ERROR}'
    "
elif [ "$TERMINAL" == "konsole" ]; then
    konsole --new-tab --title="Franka Interface" -e bash -c "
        echo '=========================================='
        echo 'Franka Interface (底层接口)'
        echo 'Robot IP: ${ROBOT_IP}, With Gripper: ${WITH_GRIPPER}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && cd ${FRANKA_INTERFACE_ROOT}/build && ./franka_interface --robot_ip ${ROBOT_IP} --with_gripper ${WITH_GRIPPER} --log ${LOG_FRANKA_INTERFACE} --stop_on_error ${STOP_ON_ERROR}'
    "
elif [ "$TERMINAL" == "xterm" ]; then
    xterm -title "Franka Interface" -e bash -c "
        echo '=========================================='
        echo 'Franka Interface (底层接口)'
        echo 'Robot IP: ${ROBOT_IP}, With Gripper: ${WITH_GRIPPER}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && cd ${FRANKA_INTERFACE_ROOT}/build && ./franka_interface --robot_ip ${ROBOT_IP} --with_gripper ${WITH_GRIPPER} --log ${LOG_FRANKA_INTERFACE} --stop_on_error ${STOP_ON_ERROR}'
    " &
fi

sleep 3

# 终端2: franka_ros_interface
echo "[2/3] Starting franka_ros_interface..."
if [ "$TERMINAL" == "gnome-terminal" ]; then
    gnome-terminal --title="Franka ROS Interface" -- bash -c "
        echo '=========================================='
        echo 'Franka ROS Interface (ROS Action Server)'
        echo 'Robot Number: ${ROBOT_NUMBER}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && ros2 launch franka_ros_interface franka_ros_interface.launch.py robot_num:=${ROBOT_NUMBER}'
    "
elif [ "$TERMINAL" == "konsole" ]; then
    konsole --new-tab --title="Franka ROS Interface" -e bash -c "
        echo '=========================================='
        echo 'Franka ROS Interface (ROS Action Server)'
        echo 'Robot Number: ${ROBOT_NUMBER}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && ros2 launch franka_ros_interface franka_ros_interface.launch.py robot_num:=${ROBOT_NUMBER}'
    "
elif [ "$TERMINAL" == "xterm" ]; then
    xterm -title "Franka ROS Interface" -e bash -c "
        echo '=========================================='
        echo 'Franka ROS Interface (ROS Action Server)'
        echo 'Robot Number: ${ROBOT_NUMBER}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && ros2 launch franka_ros_interface franka_ros_interface.launch.py robot_num:=${ROBOT_NUMBER}'
    " &
fi

sleep 3

# 终端3: franka_gripper
echo "[3/3] Starting franka_gripper..."
if [ "$TERMINAL" == "gnome-terminal" ]; then
    gnome-terminal --title="Franka Gripper" -- bash -c "
        echo '=========================================='
        echo 'Franka Gripper (夹爪节点)'
        echo 'Robot Number: ${ROBOT_NUMBER}, Robot IP: ${ROBOT_IP}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && ros2 launch franka_ros_interface franka_gripper.launch.py robot_num:=${ROBOT_NUMBER} robot_ip:=${ROBOT_IP}'
    "
elif [ "$TERMINAL" == "konsole" ]; then
    konsole --new-tab --title="Franka Gripper" -e bash -c "
        echo '=========================================='
        echo 'Franka Gripper (夹爪节点)'
        echo 'Robot Number: ${ROBOT_NUMBER}, Robot IP: ${ROBOT_IP}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && ros2 launch franka_ros_interface franka_gripper.launch.py robot_num:=${ROBOT_NUMBER} robot_ip:=${ROBOT_IP}'
    "
elif [ "$TERMINAL" == "xterm" ]; then
    xterm -title "Franka Gripper" -e bash -c "
        echo '=========================================='
        echo 'Franka Gripper (夹爪节点)'
        echo 'Robot Number: ${ROBOT_NUMBER}, Robot IP: ${ROBOT_IP}'
        echo '关闭此终端会自动结束进程'
        echo '=========================================='
        docker exec -it ${CONTAINER_NAME} bash -lc '${ENV_SETUP} && ros2 launch franka_ros_interface franka_gripper.launch.py robot_num:=${ROBOT_NUMBER} robot_ip:=${ROBOT_IP}'
    " &
fi

echo ""
echo "=========================================="
echo "All 3 terminals launched!"
echo "=========================================="
echo ""
echo "说明:"
echo "  - 终端1: Franka Interface (底层接口)"
echo "  - 终端2: Franka ROS Interface (ROS Action Server)"
echo "  - 终端3: Franka Gripper (夹爪节点)"
echo ""
echo "提示: 关闭终端窗口会自动结束对应进程"

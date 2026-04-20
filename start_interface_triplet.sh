#!/bin/bash
# 启动 Franka 三件套（在容器内后台运行）
#
# 用法：
#   ./start_interface_triplet.sh
#
# 说明：
#   - 在容器内后台启动 franka_interface、franka_ros_interface、franka_gripper
#   - 使用 nohup 运行，关闭终端不会停止进程
#   - 日志保存在 /root/Documents/my_code/vla4desk/logs/franka_backend/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE_ROOT="${HOST_CODE_ROOT:-/home/k324/franka_my_code}"
CONTAINER_NAME="${CONTAINER_NAME:-franka}"
IMAGE_NAME="${IMAGE_NAME:-franka:working}"

# 固定参数（之前成功运行的配置）
ROBOT_IP="172.16.0.2"
ROBOT_NUMBER="1"
WITH_GRIPPER="0"  # 0=新夹爪模式（使用 franka_gripper 节点）
LOG_FRANKA_INTERFACE="0"
STOP_ON_ERROR="0"

WORKSPACE_ROOT="/root/Documents/my_code"
LOG_DIR="${WORKSPACE_ROOT}/vla4desk/logs/franka_backend"
FRANKA_INTERFACE_ROOT="${WORKSPACE_ROOT}/franka-interface"

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

# 清理 FastDDS 残留 SHM/semaphore 文件，避免旧 DDS 会话影响新节点发现
rm -f /dev/shm/fastrtps_* /dev/shm/fastrtps_*_el /dev/shm/sem.fastrtps_* >/dev/null 2>&1 || true

# 仅按命令行清理，且使用 [x] 写法避免命中当前清理命令自身。
# ROS launch 崩溃后，子节点可能变成 PPID=1 的孤儿进程，所以父/子进程都要清。
patterns=(
  '[r]un_dynamic_pose.py'
  '[f]ranka_interface --robot_ip'
  '[r]os2 launch franka_ros_interface'
  '[f]ranka_ros_interface.launch.py'
  '[f]ranka_gripper.launch.py'
  '[f]ranka_ros_interface/'
  '[f]ranka_gripper/'
  '[e]xecute_skill_action_server_node_1'
  '[r]obot_state_publisher_node_1'
  '[f]ranka_interface_status_publisher_node_1'
  '[r]un_loop_process_info_state_publisher_node_1'
  '[g]et_current_robot_state_server_node_1'
  '[g]et_current_franka_interface_status_server_node_1'
  '[s]ensor_data_subscriber_node_1'
  '[f]ranka_gripper_1'
  '[g]et_current_gripper_state_server_node_1'
)

for pattern in \"\${patterns[@]}\"; do
  pkill -TERM -f \"\${pattern}\" >/dev/null 2>&1 || true
done

sleep 1

for pattern in \"\${patterns[@]}\"; do
  pkill -KILL -f \"\${pattern}\" >/dev/null 2>&1 || true
done

rm -f /dev/shm/fastrtps_* /dev/shm/fastrtps_*_el /dev/shm/sem.fastrtps_* >/dev/null 2>&1 || true
"
echo "[OK] 旧进程清理完成"

echo "=========================================="
echo "Starting Franka Interface Triplet"
echo "=========================================="
echo "Robot IP: ${ROBOT_IP}"
echo "Robot Number: ${ROBOT_NUMBER}"
echo "With Gripper: ${WITH_GRIPPER} (0=新夹爪, 1=旧夹爪)"
echo "Log: ${LOG_FRANKA_INTERFACE}"
echo "Stop on Error: ${STOP_ON_ERROR}"
echo ""

# 在容器内执行三件套启动
docker exec "${CONTAINER_NAME}" bash -lc "
set -e

export WORKSPACE_ROOT='${WORKSPACE_ROOT}'
export FRANKA_INTERFACE_ROOT='${FRANKA_INTERFACE_ROOT}'
export LOG_DIR='${LOG_DIR}'

mkdir -p \"\${LOG_DIR}\"

# 设置环境
source /opt/ros/humble/setup.bash
source \"\${FRANKA_INTERFACE_ROOT}/install/setup.bash\"
export LD_LIBRARY_PATH=\"\${FRANKA_INTERFACE_ROOT}/build/franka-interface:\${FRANKA_INTERFACE_ROOT}/build/franka-interface/proto:\${FRANKA_INTERFACE_ROOT}/build/libfranka:\${FRANKA_INTERFACE_ROOT}/build/franka-interface-common:/usr/local/lib:\${LD_LIBRARY_PATH:-}\"

# 检查 franka_interface 是否存在
if [[ ! -x \"\${FRANKA_INTERFACE_ROOT}/build/franka_interface\" ]]; then
  echo '[ERROR] franka_interface not found at \${FRANKA_INTERFACE_ROOT}/build/franka_interface'
  echo '[INFO] please build first: cd \${FRANKA_INTERFACE_ROOT} && bash ./bash_scripts/make_libfranka.sh && bash ./bash_scripts/make_franka_interface.sh'
  exit 1
fi

# 启动 franka_interface
cd \"\${FRANKA_INTERFACE_ROOT}/build\"
nohup ./franka_interface --robot_ip '${ROBOT_IP}' --with_gripper '${WITH_GRIPPER}' --log '${LOG_FRANKA_INTERFACE}' --stop_on_error '${STOP_ON_ERROR}' > \"\${LOG_DIR}/franka_interface.log\" 2>&1 &
echo \$! > \"\${LOG_DIR}/franka_interface.pid\"
echo \"[OK] franka_interface started (pid=\$!)\"

sleep 2

# 启动 franka_ros_interface
cd \"\${FRANKA_INTERFACE_ROOT}\"
nohup ros2 launch franka_ros_interface franka_ros_interface.launch.py robot_num:=${ROBOT_NUMBER} > \"\${LOG_DIR}/franka_ros_interface.launch.log\" 2>&1 &
echo \$! > \"\${LOG_DIR}/franka_ros_interface.pid\"
echo \"[OK] franka_ros_interface started (pid=\$!)\"

sleep 2

# 启动 franka_gripper（新夹爪模式，因为 with_gripper=0）
cd \"\${FRANKA_INTERFACE_ROOT}\"
nohup ros2 launch franka_ros_interface franka_gripper.launch.py robot_num:=${ROBOT_NUMBER} robot_ip:=${ROBOT_IP} > \"\${LOG_DIR}/franka_gripper.launch.log\" 2>&1 &
echo \$! > \"\${LOG_DIR}/franka_gripper.pid\"
echo \"[OK] franka_gripper started (pid=\$!)\"

echo ''
echo '[DONE] Interface triplet started successfully!'
echo \"Logs: \${LOG_DIR}\"
"

echo ""
echo "=========================================="
echo "Franka Interface Triplet Started"
echo "=========================================="
echo "Logs: ${LOG_DIR}"
echo ""
echo "查看日志:"
echo "  docker exec ${CONTAINER_NAME} tail -f ${LOG_DIR}/franka_interface.log"
echo "  docker exec ${CONTAINER_NAME} tail -f ${LOG_DIR}/franka_ros_interface.launch.log"
echo "  docker exec ${CONTAINER_NAME} tail -f ${LOG_DIR}/franka_gripper.launch.log"
echo ""
echo "停止三件套:"
echo "  docker exec ${CONTAINER_NAME} pkill -f franka_interface"
echo "  docker exec ${CONTAINER_NAME} pkill -f franka_ros_interface"
echo "  docker exec ${CONTAINER_NAME} pkill -f franka_gripper"

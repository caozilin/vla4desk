#!/bin/bash
# 启动 vla4desk 推理控制器
#
# 用法：
#   ./start_vla4desk.sh
#   ./start_vla4desk.sh --no_robot
#   ./start_vla4desk.sh --host 100.96.2.67 --port 8000
#   ./start_vla4desk.sh --web_port 8080
#   ./start_vla4desk.sh --host 100.96.2.67 --port 8000 --web_port 8081 --no_robot
#   ./start_vla4desk.sh --log_subdir eval/robu
#   ./start_vla4desk.sh --prompt "put the strawberry into the cardboard box" --log_subdir eval/robust/env_camera_all
#   ./start_vla4desk.sh --disable_async_chunk_replan
#
# 参数说明：
#   --no_robot             不连接真实机器人（仿真模式）
#   --host <ip>            VLA 服务器 IP 地址（默认: localhost）
#   --port <port>          VLA 服务器端口（默认: 8000）
#   --web_port <port>      Web 界面端口（默认: 8080）
#   --prompt <text>        初始语言指令（可在 Web 前端继续修改）
#   --log_subdir <path>    logs/ 下的可选子路径；非空时保存为 epo_1、epo_2...
#   --disable_async_chunk_replan        关闭默认启用的异步 chunk 重规划
#
# 说明：
#   - 启动 vla4desk 推理控制器，连接 VLA 模型进行视觉-语言-动作推理
#   - 需要确保 franka 容器已启动并挂载了相机设备
#   - 需要先启动 Franka 三件套（start_interface_triplet.sh）
#   - 需要先启动 VLA 服务器（如 pi0、openpi 等）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE_ROOT="${HOST_CODE_ROOT:-/home/k324/franka_my_code}"
CONTAINER_NAME="${CONTAINER_NAME:-franka}"
IMAGE_NAME="${IMAGE_NAME:-franka:working}"
DISPLAY_VALUE="${DISPLAY:-:1}"

# 获取宿主机用户ID和组ID
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

cleanup() {
    echo ""
    echo "[INFO] 收到终止信号，正在清理..."

    echo "[INFO] 终止容器内的 coordinator.py 进程..."
    docker exec "${CONTAINER_NAME}" bash -c '
        for pid in $(pgrep -f "coordinator.py"); do
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in $(pgrep -f "coordinator.py"); do
            kill -9 "$pid" 2>/dev/null || true
        done
    ' 2>/dev/null || true

    echo "[INFO] 修复文件夹权限..."
    docker exec "${CONTAINER_NAME}" bash -lc "
      chown -R ${HOST_UID}:${HOST_GID} ${WORKSPACE_ROOT}/vla4desk/collected 2>/dev/null || true
      chown -R ${HOST_UID}:${HOST_GID} ${WORKSPACE_ROOT}/vla4desk/logs 2>/dev/null || true
      chmod -R u+rw ${WORKSPACE_ROOT}/vla4desk/collected 2>/dev/null || true
      chmod -R u+rw ${WORKSPACE_ROOT}/vla4desk/logs 2>/dev/null || true
    " 2>/dev/null || true

    echo "[OK] 清理完成"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

WORKSPACE_ROOT="/root/Documents/my_code"

ensure_container_running() {
    if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1 || [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
      echo "[INFO] starting container ${CONTAINER_NAME}..."
      docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
      docker run -dit \
        --name "${CONTAINER_NAME}" \
        --privileged \
        --network host \
        --cap-add IPC_LOCK \
        --cap-add SYS_NICE \
        --security-opt label:disable \
        -e DISPLAY="${DISPLAY_VALUE}" \
        -e QT_X11_NO_MITSHM=1 \
        -e HOME=/root \
        -v "${HOST_CODE_ROOT}:/root/Documents/my_code" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v /dev/input:/dev/input \
        -v /run/udev:/run/udev:ro \
        -w /root/Documents/my_code/vla4desk \
        "${IMAGE_NAME}" \
        bash >/dev/null
    fi
}

fix_permissions() {
    echo "[INFO] 修复文件夹权限..."
    docker exec "${CONTAINER_NAME}" bash -lc "
      chown -R ${HOST_UID}:${HOST_GID} ${WORKSPACE_ROOT}/vla4desk/collected 2>/dev/null || true
      chown -R ${HOST_UID}:${HOST_GID} ${WORKSPACE_ROOT}/vla4desk/logs 2>/dev/null || true
      chmod -R u+rw ${WORKSPACE_ROOT}/vla4desk/collected 2>/dev/null || true
      chmod -R u+rw ${WORKSPACE_ROOT}/vla4desk/logs 2>/dev/null || true
    " 2>/dev/null || true
    echo "[OK] 权限修复完成"
}

ensure_container_running

FRANKA_INTERFACE_LD_PATH="${WORKSPACE_ROOT}/franka-interface/build/franka-interface:${WORKSPACE_ROOT}/franka-interface/build/franka-interface/proto:${WORKSPACE_ROOT}/franka-interface/build/libfranka:${WORKSPACE_ROOT}/franka-interface/build/franka-interface-common:/usr/local/lib"
FRANKAPY_PYTHONPATH="${WORKSPACE_ROOT}/frankapy"

echo "[INFO] 启动 coordinator.py..."
docker exec -it \
  -e WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
  -e FRANKAPY_PYTHONPATH="${FRANKAPY_PYTHONPATH}" \
  -e FRANKA_INTERFACE_LD_PATH="${FRANKA_INTERFACE_LD_PATH}" \
  "${CONTAINER_NAME}" bash -lc '
    export HOME=/root
    export LD_LIBRARY_PATH="${FRANKA_INTERFACE_LD_PATH}:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${FRANKAPY_PYTHONPATH}:${PYTHONPATH:-}"
    source /opt/ros/humble/setup.bash
    source "${WORKSPACE_ROOT}/franka-interface/install/setup.bash"
    cd "${WORKSPACE_ROOT}/vla4desk/src/vla_control"
    exec python coordinator.py "$@"
  ' _ "$@"

fix_permissions

#!/bin/bash
# 批量写入 prompt 到采集 JSON
#
# 用法：
#   ./start_write_prompts_to_json.sh collected/simple_pick_place
#   ./start_write_prompts_to_json.sh collected/strawberry_on_place
#   ./start_write_prompts_to_json.sh --fill-empty-under-collected collected
#   ./start_write_prompts_to_json.sh collected/simple_pick_place --dry-run
#   ./start_write_prompts_to_json.sh collected/simple_pick_place --prompt "pick up the banana and place it on the plate"
#
# 参数说明：
#   <path>                   采集数据目录路径（如: collected/simple_pick_place）
#   --fill-empty-under-collected <dir>  批量处理目录下的所有任务
#   --dry-run                试运行，不实际写入文件
#   --prompt <text>          指定 prompt 文本（覆盖现有 prompt）
#
# 说明：
#   - 批量给采集 JSON 写入或补写 prompt
#   - 支持从 prompts.txt 文件自动读取 prompt
#   - 使用 --dry-run 可以先预览修改内容
#   - 需要确保 franka 容器已启动

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE_ROOT="${HOST_CODE_ROOT:-/home/k324/franka_my_code}"
CONTAINER_NAME="${CONTAINER_NAME:-franka}"
IMAGE_NAME="${IMAGE_NAME:-franka:working}"
DISPLAY_VALUE="${DISPLAY:-:1}"

# 获取宿主机用户ID和组ID
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

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

# 容器内执行命令
WORKSPACE_ROOT="/root/Documents/my_code"
FRANKA_INTERFACE_LD_PATH="${WORKSPACE_ROOT}/franka-interface/build/franka-interface:${WORKSPACE_ROOT}/franka-interface/build/franka-interface/proto:${WORKSPACE_ROOT}/franka-interface/build/libfranka:${WORKSPACE_ROOT}/franka-interface/build/franka-interface-common:/usr/local/lib"
FRANKAPY_PYTHONPATH="${WORKSPACE_ROOT}/frankapy"

docker exec -it \
  -e WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
  -e FRANKAPY_PYTHONPATH="${FRANKAPY_PYTHONPATH}" \
  -e FRANKA_INTERFACE_LD_PATH="${FRANKA_INTERFACE_LD_PATH}" \
  "${CONTAINER_NAME}" bash -lc "
    export HOME=/root
    export LD_LIBRARY_PATH=\"${FRANKA_INTERFACE_LD_PATH}:\${LD_LIBRARY_PATH:-}\"
    export PYTHONPATH=\"${FRANKAPY_PYTHONPATH}:\${PYTHONPATH:-}\"
    source /opt/ros/humble/setup.bash
    source ${WORKSPACE_ROOT}/franka-interface/install/setup.bash
    cd ${WORKSPACE_ROOT}/vla4desk
    python -B src/data_collection/write_prompts_to_json.py $*
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

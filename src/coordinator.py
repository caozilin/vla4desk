"""主控制器：控制循环 + FastAPI WebSocket 服务。

状态机：
    IDLE    -> 保持当前位姿（静止）
    RUNNING -> 执行 openpi 推理 action
    HOMING  -> 停止推理，回到 home，完成后转 IDLE
"""
import asyncio
import base64
import collections
import dataclasses
import enum
import io
import logging
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "client"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cv2
import imageio
import numpy as np
import tyro
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import websocket_client_policy as _websocket_client_policy

from franka_env import FrankaEnv

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "videos"

logger = logging.getLogger(__name__)

DT = 0.05  # 推流间隔（秒），固定 20Hz


class State(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    HOMING = "homing"


@dataclasses.dataclass
class Args:
    host: str = "100.96.2.67"    # 云端推理服务 IP（openpi，Tailscale）
    port: int = 8000             # 云端推理服务端口
    web_port: int = 8080         # 本地 Web 前端端口
    cam1_serial: str | None = None   # 外部相机 serial（占位）
    cam2_serial: str | None = None   # 腕部相机 serial（占位）
    api_key: str | None = None   # 推理服务 API key（若有）
    replan_steps: int = 5        # 每次推理取几步 action 执行
    no_robot: bool = False       # 不启动 FrankaArm（机械臂未连接时使用）
    control_hz: float = 10.0     # 控制循环频率（Hz），与 skill loop 对齐


class Coordinator:
    def __init__(self, args: Args):
        self._args = args
        self._state = State.IDLE
        self._state_lock = threading.Lock()

        self._prompt = "pick up the object"  # 默认语言指令
        self._prompt_lock = threading.Lock()

        self._action_plan: collections.deque = collections.deque()

        # 录屏
        self._recording = False
        self._record_frames1: list[np.ndarray] = []
        self._record_frames2: list[np.ndarray] = []

        # 最新帧（供 WebSocket 推流）
        self._latest_img1: np.ndarray | None = None
        self._latest_img2: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # 最新 state / action（供前端显示）
        self._latest_state: list | None = None
        self._latest_action: list | None = None
        self._latest_infer_ms: float | None = None
        self._latest_prev_total_ms: float | None = None
        self._telemetry_lock = threading.Lock()

        # WebSocket 客户端集合
        self._ws_clients: set[WebSocket] = set()
        self._ws_lock = asyncio.Lock()

        self._env = FrankaEnv(args.cam1_serial, args.cam2_serial, no_robot=args.no_robot)
        try:
            self._client = _websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
            )
        except Exception as e:
            logger.warning(f"推理服务连接失败，推理功能不可用：{e}")
            self._client = None

    # ------------------------------------------------------------------
    # 状态机控制（供 REST API 调用）
    # ------------------------------------------------------------------

    def cmd_start(self):
        with self._state_lock:
            if self._state == State.IDLE:
                self._action_plan.clear()
                self._state = State.RUNNING
                logger.info("State -> RUNNING")

    def cmd_stop(self):
        with self._state_lock:
            if self._state == State.RUNNING:
                self._action_plan.clear()
                self._state = State.IDLE
                logger.info("State -> IDLE")

    def cmd_home(self):
        with self._state_lock:
            self._action_plan.clear()
            self._state = State.HOMING
            logger.info("State -> HOMING")

    def cmd_set_prompt(self, prompt: str):
        with self._prompt_lock:
            self._prompt = prompt
            logger.info(f"Prompt updated: {prompt}")

    def cmd_start_record(self):
        self._record_frames1.clear()
        self._record_frames2.clear()
        self._recording = True
        logger.info("Recording started")

    def cmd_stop_record(self):
        self._recording = False
        frames1 = self._record_frames1.copy()
        frames2 = self._record_frames2.copy()
        self._record_frames1.clear()
        self._record_frames2.clear()
        if not frames1:
            return None
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path1 = DATA_DIR / f"recording_{ts}_cam1.mp4"
        path2 = DATA_DIR / f"recording_{ts}_cam2.mp4"
        imageio.mimwrite(str(path1), frames1, fps=20, codec="libx264", pixelformat="yuv420p")
        imageio.mimwrite(str(path2), frames2, fps=20, codec="libx264", pixelformat="yuv420p")
        logger.info(f"Recording saved: {path1}, {path2} ({len(frames1)} frames)")
        return {"cam1": str(path1), "cam2": str(path2)}

    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------
    # 控制循环（在独立线程中运行）
    # ------------------------------------------------------------------

    def run_control_loop(self):
        dt = 1.0 / self._args.control_hz
        self._env.start()
        self._env.start_skill_thread()
        logger.info("Control loop started (skill thread running at 10Hz)")

        while True:
            t_start = time.time()

            try:
                self._step()
            except Exception as e:
                logger.error(f"Control loop error: {e}", exc_info=True)

            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Control loop overran by {-sleep_time*1000:.1f}ms")

    def _step(self):
        current_state = self.state

        with self._prompt_lock:
            prompt = self._prompt

        # 采集 obs（含原始帧）
        obs, img1_raw, img2_raw = self._env.get_observation(prompt)

        # 更新最新帧
        with self._frame_lock:
            self._latest_img1 = img1_raw
            self._latest_img2 = img2_raw

        # 缓存最新 state
        with self._telemetry_lock:
            self._latest_state = obs["observation/state"].tolist()

        # 录屏：分别保存双路图像帧
        if self._recording and img1_raw is not None and img2_raw is not None:
            self._record_frames1.append(img1_raw.copy())
            self._record_frames2.append(img2_raw.copy())

        if current_state == State.RUNNING:
            # action_plan 耗尽时推理
            if not self._action_plan:
                if self._client is None:
                    logger.warning("推理服务不可用，跳过推理")
                    with self._state_lock:
                        self._state = State.IDLE
                    return
                t0 = time.time()
                result = self._client.infer(obs)
                prev_total_ms = (time.time() - t0) * 1000
                action_chunk = result["actions"]
                timing = result.get("server_timing", {})
                infer_ms = timing.get("infer_ms", None)
                with self._telemetry_lock:
                    self._latest_infer_ms = infer_ms
                    self._latest_prev_total_ms = prev_total_ms
                self._action_plan.extend(action_chunk[:self._args.replan_steps])

            action = self._action_plan.popleft()  # (7,)
            self._env.enqueue_action(action)
            with self._telemetry_lock:
                self._latest_action = action.tolist()

        elif current_state == State.HOMING:
            self._env.reset_to_home()
            self._env.start_skill_thread()
            with self._state_lock:
                self._state = State.IDLE
            logger.info("Homing complete, State -> IDLE")

        else:  # IDLE
            self._env.hold_pose()

    # ------------------------------------------------------------------
    # 前端图像推流
    # ------------------------------------------------------------------

    async def stream_to_clients(self):
        """每 50ms 将最新双帧编码为 JPEG 推送给所有 WebSocket 客户端。"""
        while True:
            await asyncio.sleep(DT)

            with self._frame_lock:
                img1 = self._latest_img1
                img2 = self._latest_img2

            if img1 is None or img2 is None:
                continue

            img1_b64 = _encode_jpeg_b64(img1)
            img2_b64 = _encode_jpeg_b64(img2)

            payload = {"img1": img1_b64, "img2": img2_b64}
            import json
            msg = json.dumps(payload)

            async with self._ws_lock:
                dead = set()
                for ws in self._ws_clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                self._ws_clients -= dead


def _encode_jpeg_b64(img: np.ndarray) -> str:
    """RGB numpy → base64 JPEG string。"""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


# ------------------------------------------------------------------
# FastAPI 应用
# ------------------------------------------------------------------

def build_app(coordinator: Coordinator) -> FastAPI:
    STATIC_DIR = str(pathlib.Path(__file__).parent.parent / "static")
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open(pathlib.Path(STATIC_DIR) / "index.html") as f:
            return f.read()

    @app.post("/cmd/start")
    async def cmd_start():
        coordinator.cmd_start()
        return {"status": coordinator.state.value}

    @app.post("/cmd/stop")
    async def cmd_stop():
        coordinator.cmd_stop()
        return {"status": coordinator.state.value}

    @app.post("/cmd/home")
    async def cmd_home():
        coordinator.cmd_home()
        return {"status": coordinator.state.value}

    @app.post("/cmd/prompt")
    async def cmd_prompt(body: dict):
        coordinator.cmd_set_prompt(body.get("prompt", ""))
        return {"ok": True}

    @app.post("/cmd/record/start")
    async def cmd_record_start():
        coordinator.cmd_start_record()
        return {"recording": True}

    @app.post("/cmd/record/stop")
    async def cmd_record_stop():
        saved = coordinator.cmd_stop_record()
        return {"recording": False, "saved_to": saved}

    @app.get("/status")
    async def status():
        with coordinator._telemetry_lock:
            latest_state = coordinator._latest_state
            latest_action = coordinator._latest_action
            latest_infer_ms = coordinator._latest_infer_ms
            latest_prev_total_ms = coordinator._latest_prev_total_ms
        return {
            "state": coordinator.state.value,
            "recording": coordinator._recording,
            "prompt": coordinator._prompt,
            "robot_connected": coordinator._env._fa is not None,
            "server_connected": coordinator._client is not None,
            "latest_state": latest_state,
            "latest_action": latest_action,
            "infer_ms": latest_infer_ms,
            "prev_total_ms": latest_prev_total_ms,
        }

    @app.websocket("/ws/frames")
    async def ws_frames(websocket: WebSocket):
        await websocket.accept()
        async with coordinator._ws_lock:
            coordinator._ws_clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()  # 保持连接
        except WebSocketDisconnect:
            async with coordinator._ws_lock:
                coordinator._ws_clients.discard(websocket)

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(coordinator.stream_to_clients())

    return app


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main(args: Args):
    import uvicorn

    coordinator = Coordinator(args)

    # 控制循环在独立线程运行
    ctrl_thread = threading.Thread(target=coordinator.run_control_loop, daemon=True)
    ctrl_thread.start()

    app = build_app(coordinator)
    uvicorn.run(app, host="0.0.0.0", port=args.web_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))

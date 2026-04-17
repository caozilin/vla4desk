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
import json
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

from franka_env import FrankaEnv, transform_action

LOGS_BASE_DIR = pathlib.Path(__file__).parent.parent.parent / "logs"

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
    cam1_serial: str | None = "346222072769"   # 外部相机 serial（主视角）
    cam2_serial: str | None = "938422075745"   # 腕部相机 serial
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
        self._telemetry_log: collections.deque = collections.deque(maxlen=10800)
        self._session_dir: pathlib.Path | None = None
        self._record_start_time: float = 0.0
        self._record_lock = threading.Lock()

        # 最新帧（供 WebSocket 推流）
        self._latest_img1: np.ndarray | None = None
        self._latest_img2: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # 最新 state / action（供前端显示）
        self._latest_state: list | None = None
        self._latest_joints: list | None = None
        self._latest_target_pose: list | None = None
        self._latest_action: list | None = None
        self._latest_action_transformed: list | None = None
        self._latest_infer_ms: float | None = None
        self._latest_prev_total_ms: float | None = None
        self._latest_ee_force_torque: list | None = None
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

    def _clear_action_runtime(self):
        with self._telemetry_lock:
            self._latest_action = None
            self._latest_action_transformed = None
            self._latest_infer_ms = None
            self._latest_prev_total_ms = None

    def _reset_policy_client(self, reason: str):
        client = self._client
        if client is None:
            return
        try:
            logger.info("Resetting inference client: %s", reason)
            client.reset()
        except Exception as e:
            logger.warning("推理客户端 reset 失败（%s）：%s", reason, e)

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
        self._clear_action_runtime()
        self._reset_policy_client("stop")

    def cmd_home(self):
        with self._state_lock:
            self._action_plan.clear()
            self._state = State.HOMING
            logger.info("State -> HOMING")
        self._clear_action_runtime()
        self._reset_policy_client("home")

    def cmd_set_prompt(self, prompt: str):
        with self._prompt_lock:
            self._prompt = prompt
            logger.info(f"Prompt updated: {prompt}")

    def _start_session_logging(self):
        with self._record_lock:
            self._record_frames1.clear()
            self._record_frames2.clear()
            self._telemetry_log.clear()
            self._session_dir = LOGS_BASE_DIR / time.strftime("%Y%m%d_%H%M%S")
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._recording = True
            self._record_start_time = time.time()  # 记录开始时间，用于计算相对时间戳
        logger.info(f"Auto-recording started: {self._session_dir}")

    def _stop_session_logging(self):
        with self._record_lock:
            self._recording = False
            frames1 = self._record_frames1[:]
            frames2 = self._record_frames2[:]
            tele_log = list(self._telemetry_log)
            self._record_frames1.clear()
            self._record_frames2.clear()
            self._telemetry_log.clear()
            session_dir = self._session_dir

        if not frames1 or session_dir is None:
            return

        try:
            path1 = session_dir / "cam1.mp4"
            path2 = session_dir / "cam2.mp4"
            path_tele = session_dir / "telemetry.jsonl"

            fps = self._args.control_hz
            imageio.mimwrite(str(path1), frames1, fps=fps, codec="libx264", pixelformat="yuv420p")
            imageio.mimwrite(str(path2), frames2, fps=fps, codec="libx264", pixelformat="yuv420p")

            with open(path_tele, "w") as f:
                for entry in tele_log:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.info(f"Session saved to {session_dir}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------
    # 控制循环（在独立线程中运行）
    # ------------------------------------------------------------------

    def run_control_loop(self):
        """10Hz 控制循环。"""
        dt = 1.0 / self._args.control_hz
        self._env.start_control(home_first=True)
        logger.info("Control loop started after homing (skill thread running at 10Hz)")

        last_state = State.IDLE
        next_tick = time.perf_counter()
        while True:
            t_start = time.perf_counter()
            with self._state_lock:
                current_state = self._state

            self._sync_recording_state(current_state, last_state)
            last_state = current_state

            try:
                self._step()
            except Exception as e:
                logger.error(f"Control loop error: {e}", exc_info=True)

            next_tick += dt
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning("Control loop overran by %.1fms", -sleep_time * 1000.0)
                next_tick = now

    def _sync_recording_state(self, current_state: State, last_state: State):
        if current_state == State.RUNNING and last_state != State.RUNNING:
            self._start_session_logging()
        elif current_state != State.RUNNING and last_state == State.RUNNING:
            self._stop_session_logging()

    def _capture_observation(self, prompt: str):
        obs, img1_raw, img2_raw = self._env.get_observation(prompt)
        with self._frame_lock:
            self._latest_img1 = img1_raw
            self._latest_img2 = img2_raw
        return obs, img1_raw, img2_raw

    def _update_latest_telemetry(self, obs: dict):
        with self._telemetry_lock:
            self._latest_state = obs["observation/state"].tolist()
            self._latest_joints = obs.get("observation/joints", np.zeros(7)).tolist()
            self._latest_target_pose = self._env.commanded_pose_array.tolist()
            self._latest_ee_force_torque = self._env.ee_force_torque.tolist()

    def _record_frames(self, img1_raw: np.ndarray, img2_raw: np.ndarray):
        with self._record_lock:
            if self._recording and img1_raw is not None and img2_raw is not None:
                self._record_frames1.append(img1_raw.copy())
                self._record_frames2.append(img2_raw.copy())

    def _infer_action_plan_if_needed(self, obs: dict):
        if self._action_plan:
            return
        if self._client is None:
            logger.warning("推理服务不可用，跳过推理")
            with self._state_lock:
                self._state = State.IDLE
            return

        t0 = time.time()
        try:
            result = self._client.infer(obs)
        except Exception as e:
            logger.warning("推理请求失败，切回 IDLE：%s", e)
            self._clear_action_runtime()
            self._reset_policy_client("infer_error")
            with self._state_lock:
                self._state = State.IDLE
            return
        prev_total_ms = (time.time() - t0) * 1000
        action_chunk = result["actions"]
        timing = result.get("server_timing", {})
        infer_ms = timing.get("infer_ms", None)
        with self._telemetry_lock:
            self._latest_infer_ms = infer_ms
            self._latest_prev_total_ms = prev_total_ms
        self._action_plan.extend(action_chunk[:self._args.replan_steps])

    def _record_action_telemetry(self):
        with self._record_lock:
            if self._recording:
                relative_time = time.time() - self._record_start_time
                with self._prompt_lock:
                    prompt = self._prompt
                entry = {
                    "timestamp": round(relative_time, 3),
                    "controller_state": self._state.value,
                    "prompt": prompt,
                    "state": self._latest_state,
                    "joint_state": self._latest_joints,
                    "target_pose": self._latest_target_pose,
                    "ee_force_torque": self._latest_ee_force_torque,
                    "action_raw": self._latest_action,
                    "action_transformed": self._latest_action_transformed,
                    "inference_time_ms": self._latest_infer_ms,
                    "total_time_ms": self._latest_prev_total_ms,
                }
                self._telemetry_log.append(entry)

    def _run_action_step(self, obs: dict):
        self._infer_action_plan_if_needed(obs)
        if not self._action_plan:
            return

        action = self._action_plan.popleft()
        self._env.enqueue_action(action)
        ta = transform_action(action)

        with self._telemetry_lock:
            self._latest_action = action.tolist()
            self._latest_action_transformed = ta.tolist()

        self._record_action_telemetry()

    def _handle_homing(self):
        self._env.home_and_restart()
        self._clear_action_runtime()
        with self._state_lock:
            self._state = State.IDLE
        logger.info("Homing complete, State -> IDLE")

    def _build_stream_payload(self, img1_b64: str, img2_b64: str) -> str:
        with self._prompt_lock:
            prompt = self._prompt
        payload = {
            "controller_state": self._state.value,
            "recording": self._recording,
            "prompt": prompt,
            "img1": img1_b64,
            "img2": img2_b64,
            "state": self._latest_state,
            "joint_state": self._latest_joints,
            "target_pose": self._latest_target_pose,
            "ee_force_torque": self._latest_ee_force_torque,
            "action_raw": self._latest_action,
            "action_transformed": self._latest_action_transformed,
            "infer_ms": self._latest_infer_ms,
            "total_ms": self._latest_prev_total_ms,
            "robot_connected": self._env._fa is not None,
            "server_connected": self._client is not None,
        }
        return json.dumps(payload)

    def _step(self):
        current_state = self.state

        with self._prompt_lock:
            prompt = self._prompt

        obs, img1_raw, img2_raw = self._capture_observation(prompt)
        self._update_latest_telemetry(obs)
        self._record_frames(img1_raw, img2_raw)

        if current_state == State.RUNNING:
            self._run_action_step(obs)

        elif current_state == State.HOMING:
            self._handle_homing()

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

            msg = self._build_stream_payload(img1_b64, img2_b64)

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
    STATIC_DIR = str(pathlib.Path(__file__).parent.parent.parent / "static")
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        static_file = pathlib.Path(STATIC_DIR) / "index.html"
        with open(static_file, "r") as f:
            content = f.read()
        return HTMLResponse(content=content)

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

    @app.get("/status")
    async def status():
        with coordinator._telemetry_lock:
            latest_state = coordinator._latest_state
            latest_joints = coordinator._latest_joints
            latest_target_pose = coordinator._latest_target_pose
            latest_action = coordinator._latest_action
            latest_action_transformed = coordinator._latest_action_transformed
            latest_infer_ms = coordinator._latest_infer_ms
            latest_prev_total_ms = coordinator._latest_prev_total_ms
        return {
            "controller_state": coordinator.state.value,
            "recording": coordinator._recording,
            "prompt": coordinator._prompt,
            "robot_connected": coordinator._env._fa is not None,
            "server_connected": coordinator._client is not None,
            "state": latest_state,
            "joint_state": latest_joints,
            "target_pose": latest_target_pose,
            "action_raw": latest_action,
            "action_transformed": latest_action_transformed,
            "infer_ms": latest_infer_ms,
            "total_ms": latest_prev_total_ms,
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

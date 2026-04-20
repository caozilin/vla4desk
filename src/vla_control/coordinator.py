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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    prompt: str = "pick up the object"  # 初始语言指令
    log_subdir: str = ""         # logs/ 下的可选子路径，空字符串保持时间戳命名
    cam1_serial: str | None = "346222072769"   # 外部相机 serial（主视角）
    cam2_serial: str | None = "938422075745"   # 腕部相机 serial
    api_key: str | None = None   # 推理服务 API key（若有）
    replan_steps: int = 5        # 每次推理取几步 action 执行
    no_robot: bool = False       # 不启动 FrankaArm（机械臂未连接时使用）
    control_hz: float = 10.0     # 控制循环频率（Hz），与 skill loop 对齐
    disable_async_chunk_replan: bool = False  # 默认启用；传入该开关则关闭异步 action chunk 重规划
    async_launch_after_steps: int = 2
    async_delay_steps_default: int = 3
    async_s_min: int = 2
    async_blend_enabled: bool = True
    async_blend_weights: str = "1.0,1.0,0.6,0.25,0.0"
    async_latest_only: bool = True
    async_use_measured_delay: bool = True
    async_delay_history_len: int = 20
    async_swap_immediately_when_ready: bool = True


class Coordinator:
    def __init__(self, args: Args):
        self._args = args
        self._state = State.IDLE
        self._state_lock = threading.Lock()

        self._prompt = args.prompt  # 默认语言指令
        self._prompt_lock = threading.Lock()

        self._log_subdir = _normalize_log_subdir(args.log_subdir)
        self._log_config_lock = threading.Lock()

        self._action_plan: collections.deque = collections.deque()
        self._client_lock = threading.Lock()
        self._async_chunk_enabled = not args.disable_async_chunk_replan

        # 异步 chunk 重规划运行时状态
        self._current_chunk: list[np.ndarray] | None = None
        self._next_chunk: list[np.ndarray] | None = None
        self._chunk_exec_idx = 0
        self._async_infer_running = False
        self._async_infer_thread: threading.Thread | None = None
        self._async_generation = 0
        self._async_lock = threading.Lock()
        self._async_delay_history: collections.deque = collections.deque(
            maxlen=max(1, args.async_delay_history_len)
        )
        self._latest_delay_steps = int(max(1, args.async_delay_steps_default))
        self._blend_weights = _parse_blend_weights(args.async_blend_weights)

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
            with self._client_lock:
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
                self._reset_async_chunk_runtime()
                self._state = State.RUNNING
                logger.info("State -> RUNNING")

    def cmd_stop(self):
        with self._state_lock:
            if self._state == State.RUNNING:
                self._action_plan.clear()
                self._reset_async_chunk_runtime()
                self._state = State.IDLE
                logger.info("State -> IDLE")
        self._clear_action_runtime()
        self._reset_policy_client("stop")

    def cmd_home(self):
        with self._state_lock:
            self._action_plan.clear()
            self._reset_async_chunk_runtime()
            self._state = State.HOMING
            logger.info("State -> HOMING")
        self._clear_action_runtime()
        self._reset_policy_client("home")

    def cmd_set_prompt(self, prompt: str):
        with self._prompt_lock:
            self._prompt = prompt
            logger.info(f"Prompt updated: {prompt}")

    def cmd_set_log_subdir(self, log_subdir: str) -> str:
        normalized = _normalize_log_subdir(log_subdir)
        with self._log_config_lock:
            self._log_subdir = normalized
        logger.info("Log subdir updated: %s", normalized or "<default>")
        return normalized

    @property
    def log_subdir(self) -> str:
        with self._log_config_lock:
            return self._log_subdir

    def _allocate_session_dir(self) -> pathlib.Path:
        log_subdir = self.log_subdir
        if not log_subdir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for idx in range(100):
                suffix = "" if idx == 0 else f"_{idx:02d}"
                session_dir = LOGS_BASE_DIR / f"{stamp}{suffix}"
                try:
                    session_dir.mkdir(parents=True, exist_ok=False)
                    return session_dir
                except FileExistsError:
                    continue
            session_dir = LOGS_BASE_DIR / f"{stamp}_{time.time_ns()}"
            session_dir.mkdir(parents=True, exist_ok=False)
            return session_dir

        base_dir = LOGS_BASE_DIR / log_subdir
        base_dir.mkdir(parents=True, exist_ok=True)

        nums = []
        for path in base_dir.iterdir():
            if path.is_dir() and path.name.startswith("epo_"):
                try:
                    nums.append(int(path.name[4:]))
                except ValueError:
                    pass

        next_epo = (max(nums) + 1) if nums else 1
        while True:
            session_dir = base_dir / f"epo_{next_epo}"
            try:
                session_dir.mkdir(parents=True, exist_ok=False)
                return session_dir
            except FileExistsError:
                next_epo += 1

    def _start_session_logging(self):
        with self._record_lock:
            self._record_frames1.clear()
            self._record_frames2.clear()
            self._telemetry_log.clear()
            self._session_dir = self._allocate_session_dir()
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
            self._session_dir = None

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

        try:
            chunk, infer_ms, prev_total_ms = self._infer_chunk(obs)
        except Exception as e:
            logger.warning("推理请求失败，切回 IDLE：%s", e)
            self._clear_action_runtime()
            self._reset_policy_client("infer_error")
            with self._state_lock:
                self._state = State.IDLE
            return
        with self._telemetry_lock:
            self._latest_infer_ms = infer_ms
            self._latest_prev_total_ms = prev_total_ms
        self._action_plan.extend(chunk)

    def _infer_chunk(
        self,
        obs: dict,
        *,
        delay_steps: int = 0,
        pad_short: bool = False,
    ) -> tuple[list[np.ndarray], float | None, float]:
        if self._client is None:
            raise RuntimeError("推理服务不可用")

        t0 = time.time()
        with self._client_lock:
            result = self._client.infer(obs)
        prev_total_ms = (time.time() - t0) * 1000
        timing = result.get("server_timing", {})
        infer_ms = timing.get("infer_ms", None)
        chunk = self._coerce_action_chunk(
            result["actions"],
            delay_steps=delay_steps,
            pad_short=pad_short,
        )
        return chunk, infer_ms, prev_total_ms

    def _coerce_action_chunk(
        self,
        actions,
        *,
        delay_steps: int = 0,
        pad_short: bool = False,
    ) -> list[np.ndarray]:
        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(f"Expected actions with shape (H, 7), got {arr.shape}")

        target_len = max(1, int(self._args.replan_steps))
        start = max(0, int(delay_steps))
        if start >= len(arr):
            start = max(0, len(arr) - 1)
        sliced = arr[start:start + target_len]
        if len(sliced) == 0:
            return []
        if pad_short and len(sliced) < target_len:
            pad = np.repeat(sliced[-1:], target_len - len(sliced), axis=0)
            sliced = np.concatenate([sliced, pad], axis=0)
        return [row.astype(np.float64).copy() for row in sliced]

    def _copy_observation(self, obs: dict) -> dict:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in obs.items()
        }

    def _estimate_delay_steps(self, total_ms: float) -> int:
        if not self._args.async_use_measured_delay:
            return int(max(1, self._args.async_delay_steps_default))
        step_ms = 1000.0 / max(1e-6, float(self._args.control_hz))
        return int(max(1, round(float(total_ms) / step_ms)))

    def _bootstrap_chunk(self, obs: dict) -> bool:
        if self._client is None:
            logger.warning("推理服务不可用，跳过推理")
            with self._state_lock:
                self._state = State.IDLE
            return False

        try:
            chunk, infer_ms, prev_total_ms = self._infer_chunk(obs)
        except Exception as e:
            logger.warning("推理请求失败，切回 IDLE：%s", e)
            self._clear_action_runtime()
            self._reset_policy_client("async_bootstrap_error")
            with self._state_lock:
                self._state = State.IDLE
            return False

        delay_steps = self._estimate_delay_steps(prev_total_ms)
        with self._async_lock:
            self._current_chunk = chunk
            self._next_chunk = None
            self._chunk_exec_idx = 0
            self._latest_delay_steps = delay_steps
            self._async_delay_history.append(delay_steps)
        with self._telemetry_lock:
            self._latest_infer_ms = infer_ms
            self._latest_prev_total_ms = prev_total_ms
        return bool(chunk)

    def _maybe_launch_async_infer(self, obs: dict):
        if not self._async_chunk_enabled or self._client is None:
            return

        with self._async_lock:
            thread_alive = self._async_infer_thread is not None and self._async_infer_thread.is_alive()
            if self._async_infer_running or thread_alive:
                return
            if self._current_chunk is None:
                return
            if self._chunk_exec_idx < max(0, int(self._args.async_launch_after_steps)):
                return

            self._async_infer_running = True
            generation = self._async_generation

        obs_snapshot = self._copy_observation(obs)
        thread = threading.Thread(
            target=self._async_infer_worker,
            args=(obs_snapshot, generation),
            daemon=True,
        )
        with self._async_lock:
            self._async_infer_thread = thread
        thread.start()

    def _async_infer_worker(self, obs_snapshot: dict, generation: int):
        total_ms = 0.0
        infer_ms = None
        try:
            if self._client is None:
                raise RuntimeError("推理服务不可用")

            t0 = time.time()
            with self._client_lock:
                result = self._client.infer(obs_snapshot)
            total_ms = (time.time() - t0) * 1000
            timing = result.get("server_timing", {})
            infer_ms = timing.get("infer_ms", None)
            delay_steps = self._estimate_delay_steps(total_ms)
            start_offset = max(int(self._args.async_s_min), delay_steps)
            new_chunk = self._coerce_action_chunk(
                result["actions"],
                delay_steps=start_offset,
                pad_short=True,
            )

            with self._async_lock:
                if generation != self._async_generation:
                    return
                if self._args.async_blend_enabled and self._args.async_swap_immediately_when_ready:
                    next_chunk = self._blend_chunks(
                        self._current_chunk,
                        self._chunk_exec_idx,
                        new_chunk,
                    )
                else:
                    next_chunk = new_chunk
                self._next_chunk = next_chunk
                self._latest_delay_steps = delay_steps
                self._async_delay_history.append(delay_steps)

            with self._telemetry_lock:
                self._latest_infer_ms = infer_ms
                self._latest_prev_total_ms = total_ms
        except Exception as e:
            logger.warning("异步推理失败，继续执行当前 chunk：%s", e)
        finally:
            with self._async_lock:
                if generation == self._async_generation:
                    self._async_infer_running = False

    def _maybe_swap_next_chunk(self, *, force: bool = False) -> bool:
        with self._async_lock:
            if self._next_chunk is None:
                return False
            current_exhausted = (
                self._current_chunk is None
                or self._chunk_exec_idx >= len(self._current_chunk)
            )
            if not force and not self._args.async_swap_immediately_when_ready and not current_exhausted:
                return False

            self._current_chunk = self._next_chunk
            self._next_chunk = None
            self._chunk_exec_idx = 0
            return True

    def _blend_single_action(self, old_action: np.ndarray, new_action: np.ndarray, weight: float) -> np.ndarray:
        w = float(np.clip(weight, 0.0, 1.0))
        old_arr = np.asarray(old_action, dtype=np.float64)
        new_arr = np.asarray(new_action, dtype=np.float64)
        blended = new_arr.copy()
        blended[:6] = w * old_arr[:6] + (1.0 - w) * new_arr[:6]
        blended[6] = old_arr[6] if w >= 0.5 else new_arr[6]
        return blended.astype(np.float64)

    def _blend_chunks(
        self,
        old_chunk: list[np.ndarray] | None,
        old_start_idx: int,
        new_chunk: list[np.ndarray],
    ) -> list[np.ndarray]:
        if not old_chunk:
            return [np.asarray(action, dtype=np.float64).copy() for action in new_chunk]

        old_remaining = old_chunk[max(0, int(old_start_idx)):]
        blended: list[np.ndarray] = []
        for idx, new_action in enumerate(new_chunk):
            if idx < len(old_remaining):
                weight = self._blend_weights[idx] if idx < len(self._blend_weights) else 0.0
                blended.append(self._blend_single_action(old_remaining[idx], new_action, weight))
            else:
                blended.append(np.asarray(new_action, dtype=np.float64).copy())
        return blended

    def _reset_async_chunk_runtime(self):
        with self._async_lock:
            self._current_chunk = None
            self._next_chunk = None
            self._chunk_exec_idx = 0
            self._async_generation += 1
            self._async_infer_running = False
            self._async_delay_history.clear()
            self._latest_delay_steps = int(max(1, self._args.async_delay_steps_default))

    def _async_status_fields(self) -> dict:
        with self._async_lock:
            current_chunk_len = len(self._current_chunk) if self._current_chunk is not None else 0
            chunk_exec_idx = self._chunk_exec_idx
            delay_steps = self._latest_delay_steps
            async_infer_running = self._async_infer_running

        return {
            "delay_steps": delay_steps,
            "chunk_exec_idx": chunk_exec_idx,
            "current_chunk_len": current_chunk_len,
            "pending_action_count": self._env.get_pending_action_count(),
            "async_chunk_enabled": bool(self._async_chunk_enabled),
            "async_infer_running": async_infer_running,
            "blend_enabled": bool(
                self._async_chunk_enabled and self._args.async_blend_enabled
            ),
            "blend_weights": list(self._blend_weights),
        }

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
                entry.update(self._async_status_fields())
                self._telemetry_log.append(entry)

    def _run_sync_action_step(self, obs: dict):
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

    def _run_async_action_step(self, obs: dict):
        self._maybe_swap_next_chunk()

        with self._async_lock:
            needs_bootstrap = self._current_chunk is None
        if needs_bootstrap and not self._bootstrap_chunk(obs):
            return

        self._maybe_launch_async_infer(obs)
        self._maybe_swap_next_chunk()

        with self._async_lock:
            if self._current_chunk is None:
                return
            if self._chunk_exec_idx >= len(self._current_chunk):
                current_exhausted = True
                infer_running = self._async_infer_running
            else:
                current_exhausted = False
                infer_running = self._async_infer_running

        if current_exhausted:
            if self._maybe_swap_next_chunk(force=True):
                with self._async_lock:
                    action = self._current_chunk[self._chunk_exec_idx]
                    self._chunk_exec_idx += 1
            else:
                if not infer_running:
                    logger.warning("异步 chunk 已耗尽且没有可用 next_chunk，切回 IDLE")
                    with self._state_lock:
                        self._state = State.IDLE
                return
        else:
            with self._async_lock:
                action = self._current_chunk[self._chunk_exec_idx]
                self._chunk_exec_idx += 1

        self._env.enqueue_action(action, latest_only=self._args.async_latest_only)
        ta = transform_action(action)

        with self._telemetry_lock:
            self._latest_action = action.tolist()
            self._latest_action_transformed = ta.tolist()

        self._record_action_telemetry()

    def _run_action_step(self, obs: dict):
        if self._async_chunk_enabled:
            self._run_async_action_step(obs)
        else:
            self._run_sync_action_step(obs)

    def _handle_homing(self):
        self._env.home_and_restart()
        self._clear_action_runtime()
        self._reset_async_chunk_runtime()
        with self._state_lock:
            self._state = State.IDLE
        logger.info("Homing complete, State -> IDLE")

    def _build_stream_payload(self, img1_b64: str, img2_b64: str) -> str:
        with self._prompt_lock:
            prompt = self._prompt
        with self._record_lock:
            session_dir = self._session_dir
        payload = {
            "controller_state": self._state.value,
            "recording": self._recording,
            "prompt": prompt,
            "log_subdir": self.log_subdir,
            "log_session_dir": _display_log_path(session_dir),
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
        payload.update(self._async_status_fields())
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


def _parse_blend_weights(value: str) -> list[float]:
    weights: list[float] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        weights.append(float(item))
    return weights or [0.0]


def _normalize_log_subdir(value: str | None) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if pathlib.PurePosixPath(raw).is_absolute():
        raise ValueError("log_subdir must be relative to logs/")
    parts = [part for part in raw.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ValueError("log_subdir cannot contain '.' or '..'")
    return "/".join(parts)


def _display_log_path(path: pathlib.Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(LOGS_BASE_DIR.parent))
    except ValueError:
        return str(path)


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

    @app.post("/cmd/log_subdir")
    async def cmd_log_subdir(body: dict):
        try:
            log_subdir = coordinator.cmd_set_log_subdir(body.get("log_subdir", ""))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "ok": True,
            "log_subdir": log_subdir,
            "log_session_dir": "",
        }

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
            "log_subdir": coordinator.log_subdir,
            "log_session_dir": _display_log_path(coordinator._session_dir),
            "robot_connected": coordinator._env._fa is not None,
            "server_connected": coordinator._client is not None,
            "state": latest_state,
            "joint_state": latest_joints,
            "target_pose": latest_target_pose,
            "action_raw": latest_action,
            "action_transformed": latest_action_transformed,
            "infer_ms": latest_infer_ms,
            "total_ms": latest_prev_total_ms,
            **coordinator._async_status_fields(),
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

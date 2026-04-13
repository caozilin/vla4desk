#!/usr/bin/env python3
"""
Franka Panda 键盘遥操作控制器
=============================
提供：
  - 10Hz 输入采样：读取按键并计算当前这拍的 action 增量
  - 通过 FrankaEnv.enqueue_action() 将动作送入唯一的 dynamic skill 执行线程
  - 按键 → 位姿增量的映射（平移按基座系，旋转输入可选基座系或末端系）
  - 输出 action 统一使用基座坐标系 rotvec 语义
  - 保持采集侧独立的最大速度配置，不复用 env 的动作缩放

本文件不涉及录制逻辑，供 data_recorder.py 等模块引用。
"""

import logging
import math
import os
import pathlib
import sys
import threading
import time

import numpy as np

from pynput import keyboard
from scipy.spatial.transform import Rotation

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    import pygame
except ModuleNotFoundError:
    pygame = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "vla_control"))

from franka_env import FrankaEnv


# ==================================================================
# 控制参数
# ==================================================================

INPUT_DT = 0.1               # 输入采样间隔 100ms (10Hz)
MAX_LIN_VEL = 0.1             # 最大线速度 0.1 m/s
MAX_ROT_VEL = math.pi / 4     # 最大角速度 45°/s
MAX_DELTA_POS = MAX_LIN_VEL * INPUT_DT     # 0.01 m
MAX_DELTA_ROT = MAX_ROT_VEL * INPUT_DT     # ≈ 0.0785 rad
JOYSTICK_DEADZONE = 0.12

PS4_AXIS_LEFT_X = 0
PS4_AXIS_LEFT_Y = 1
PS4_AXIS_L2 = 2
PS4_AXIS_RIGHT_X = 3
PS4_AXIS_RIGHT_Y = 4
PS4_AXIS_R2 = 5

PS4_BUTTON_SQUARE = 0
PS4_BUTTON_CROSS = 1
PS4_BUTTON_CIRCLE = 2
PS4_BUTTON_TRIANGLE = 3
PS4_BUTTON_L1 = 4
PS4_BUTTON_R1 = 5
PS4_BUTTON_SHARE = 8
PS4_BUTTON_OPTIONS = 9
PS4_BUTTON_L3 = 10
PS4_BUTTON_R3 = 11
PS4_BUTTON_PS = 12


# ==================================================================
# 键盘控制器
# ==================================================================

class KeyboardController:
    """输入控制器。

    使用方式（推荐）：
        kc = KeyboardController()
        kc.start()                # 启动 env skill + 输入线程（非阻塞）
        # 自行管理 keyboard.Listener
        with keyboard.Listener(...) as listener:
            listener.join()
        kc.stop()

    公开属性（线程安全）：
        gripper_target     float 夹爪目标宽度 (m)
        step_size          float 当前速度倍率，取值来自 [0.4, 0.7, 1.0]
        running            bool  程序是否在运行
    """

    def __init__(
        self,
        input_device: str = "keyboard",
        joystick_index: int = 0,
        rotation_input_frame: str = "eef",
    ):
        if input_device not in ("keyboard", "ps4"):
            raise ValueError(f"不支持的输入设备: {input_device}")
        if rotation_input_frame not in ("eef", "base"):
            raise ValueError(f"不支持的旋转输入坐标系: {rotation_input_frame}")

        self.input_device = input_device
        self.joystick_index = joystick_index
        self.rotation_input_frame = rotation_input_frame
        self.env = FrankaEnv()

        self.gripper_target = 0.04   # 默认半开
        self.step_size = 0.7          # 速度倍率：3档 [0.4, 0.7, 1.0]
        self._speed_levels = [0.4, 0.7, 1.0]  # 3档速度：40%, 70%, 100%
        self._speed_index = 1                  # 默认第2档（70%）
        self.running = True

        # 按键状态
        self.keys_pressed: set = set()
        self._keys_lock = threading.Lock()
        self._motion_keys = {
            "w", "s", "a", "d",
            "i", "k",
            "q", "e",
            "u", "o",
            "j", "l",
        }

        # 线程管理
        self._input_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._reset_lock = threading.Lock()  # 防止并发复位

        # 状态缓存
        self._latest_state = np.zeros(8, dtype=np.float64)
        self._latest_action = np.zeros(7, dtype=np.float64)

        # 手柄事件
        self._event_callbacks: dict[str, callable] = {}
        self._pygame_ready = False
        self._joystick = None
        self._prev_buttons: dict[int, bool] = {}
        self._prev_hat = (0, 0)
        if self.input_device == "ps4":
            self._init_ps4()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self):
        """启动 env 与输入线程（daemon），不阻塞。"""
        self._print_controls()
        self.env.start_control(home_first=True)
        self._latest_state = self.env.get_robot_state_vector()
        self._latest_action = np.zeros(7, dtype=np.float64)
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def stop(self):
        """停止控制线程，等待其退出。"""
        self.running = False
        self._stop_event.set()
        if (
            self._input_thread
            and self._input_thread.is_alive()
            and threading.current_thread() is not self._input_thread
        ):
            self._input_thread.join(timeout=2.0)
        self.env.stop_control()
        self._shutdown_ps4()

    def bind_event(self, event_name: str, callback):
        """绑定离散事件回调，例如 record_start / record_stop。"""
        self._event_callbacks[event_name] = callback

    # ------------------------------------------------------------------
    # 键盘回调（供外部 Listener 注入）
    # ------------------------------------------------------------------

    def _on_key_press(self, key):
        if self.input_device != "keyboard":
            return
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()

        with self._keys_lock:
            if char == 'escape':
                self.stop()
                return False

            if char == 'r':
                self._reset()
                return

            if char in ('+', '='):
                self._speed_index = min(len(self._speed_levels) - 1, self._speed_index + 1)
                self.step_size = self._speed_levels[self._speed_index]
                print(f"  [速度] 倍率: {self.step_size*100:.0f}%")
                return

            if char in ('-', '_'):
                self._speed_index = max(0, self._speed_index - 1)
                self.step_size = self._speed_levels[self._speed_index]
                print(f"  [速度] 倍率: {self.step_size*100:.0f}%")
                return

            if char == 'g':
                self._close_gripper(); return
            if char == 'h':
                self._open_gripper(); return
            if char == 'f':
                self._half_gripper(); return

            self.keys_pressed.add(char)
            if char in self._motion_keys:
                logging.info("键盘按下: key=%s active=%s", char, "".join(sorted(self.keys_pressed)))

    def _on_key_release(self, key):
        if self.input_device != "keyboard":
            return
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()
        with self._keys_lock:
            self.keys_pressed.discard(char)
            if char in self._motion_keys:
                logging.info("键盘释放: key=%s active=%s", char, "".join(sorted(self.keys_pressed)))

    def _init_ps4(self):
        if pygame is None:
            raise ModuleNotFoundError(
                "PS4 手柄模式需要安装 pygame，请先执行 `pip install pygame`。"
            )

        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count <= self.joystick_index:
            raise RuntimeError(
                f"未找到索尼 PS4 手柄。当前检测到 {count} 个手柄，"
                f"但请求使用索引 {self.joystick_index}。"
            )

        self._joystick = pygame.joystick.Joystick(self.joystick_index)
        self._joystick.init()
        self._pygame_ready = True
        self._prev_hat = (0, 0)
        self._prev_buttons = {
            idx: bool(self._joystick.get_button(idx))
            for idx in range(self._joystick.get_numbuttons())
        }

        print(f"  [手柄] 已连接: {self._joystick.get_name()} (index={self.joystick_index})")

    def _shutdown_ps4(self):
        if not self._pygame_ready or pygame is None:
            return

        if self._joystick is not None and self._joystick.get_init():
            self._joystick.quit()
        pygame.joystick.quit()
        pygame.quit()
        self._pygame_ready = False

    def _print_controls(self):
        if self.input_device == "keyboard":
            print("  [控制] 键盘模式")
            print("  [控制] W/S:X  A/D:Y  I/K:Z  Q/E:Roll  U/O:Pitch  J/L:Yaw")
            print(f"  [控制] 旋转输入坐标系: {self.rotation_input_frame}")
            print("  [控制] G/H/F:夹爪  +/-:速度档位  R:复位  1/2:开始/结束录制  ESC:退出")
            return

        print("  [控制] PS4 手柄模式")
        print("  [控制] 左摇杆:X/Y平移  右摇杆:Z平移/Yaw  L1/R1:Roll  L2/R2:Pitch")
        print(f"  [控制] 旋转输入坐标系: {self.rotation_input_frame}")
        print("  [控制] 方块/圆圈/三角:关闭/打开/半开夹爪  十字键左右:速度档位")
        print("  [控制] L3/R3:开始/结束录制  OPTIONS:复位  PS:退出")

    def _emit_event(self, event_name: str):
        callback = self._event_callbacks.get(event_name)
        if callback is not None:
            callback()

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < JOYSTICK_DEADZONE else float(value)

    def _read_axis(self, axis_index: int) -> float:
        if self._joystick is None or axis_index >= self._joystick.get_numaxes():
            return 0.0
        return self._apply_deadzone(self._joystick.get_axis(axis_index))

    def _read_trigger(self, axis_index: int, fallback_button: int | None = None) -> float:
        if self._joystick is not None and axis_index < self._joystick.get_numaxes():
            raw = self._joystick.get_axis(axis_index)
            normalized = (raw + 1.0) / 2.0
            if normalized < JOYSTICK_DEADZONE:
                return 0.0
            return float(np.clip(normalized, 0.0, 1.0))

        if fallback_button is not None and self._joystick is not None:
            if fallback_button < self._joystick.get_numbuttons():
                return float(self._joystick.get_button(fallback_button))
        return 0.0

    def _handle_ps4_buttons(self):
        if self._joystick is None:
            return

        for button_idx in range(self._joystick.get_numbuttons()):
            pressed = bool(self._joystick.get_button(button_idx))
            prev_pressed = self._prev_buttons.get(button_idx, False)
            if pressed and not prev_pressed:
                if button_idx == PS4_BUTTON_PS:
                    self.stop()
                    return
                if button_idx == PS4_BUTTON_OPTIONS:
                    self._reset()
                    return
                if button_idx == PS4_BUTTON_L3:
                    self._emit_event("record_start")
                elif button_idx == PS4_BUTTON_R3:
                    self._emit_event("record_stop")
                elif button_idx == PS4_BUTTON_SQUARE:
                    self._close_gripper()
                elif button_idx == PS4_BUTTON_CIRCLE:
                    self._open_gripper()
                elif button_idx == PS4_BUTTON_TRIANGLE:
                    self._half_gripper()
            self._prev_buttons[button_idx] = pressed

        hat = self._joystick.get_hat(0) if self._joystick.get_numhats() > 0 else (0, 0)
        prev_hat_x, _ = self._prev_hat
        hat_x, _ = hat
        if hat_x == 1 and prev_hat_x != 1:
            self._speed_index = min(len(self._speed_levels) - 1, self._speed_index + 1)
            self.step_size = self._speed_levels[self._speed_index]
            print(f"  [速度] 倍率: {self.step_size*100:.0f}%")
        elif hat_x == -1 and prev_hat_x != -1:
            self._speed_index = max(0, self._speed_index - 1)
            self.step_size = self._speed_levels[self._speed_index]
            print(f"  [速度] 倍率: {self.step_size*100:.0f}%")
        self._prev_hat = hat

    def _get_keyboard_delta(self) -> tuple[float, float, float, float, float, float]:
        with self._keys_lock:
            keys = set(self.keys_pressed)

        speed = self.step_size
        dx = (int('s' in keys) - int('w' in keys)) * MAX_DELTA_POS * speed
        dy = (int('d' in keys) - int('a' in keys)) * MAX_DELTA_POS * speed
        dz = (int('i' in keys) - int('k' in keys)) * MAX_DELTA_POS * speed
        droll = (int('q' in keys) - int('e' in keys)) * MAX_DELTA_ROT * speed
        dpitch = (int('u' in keys) - int('o' in keys)) * MAX_DELTA_ROT * speed
        dyaw = (int('l' in keys) - int('j' in keys)) * MAX_DELTA_ROT * speed
        return dx, dy, dz, droll, dpitch, dyaw

    def _get_ps4_delta(self) -> tuple[float, float, float, float, float, float]:
        pygame.event.pump()
        self._handle_ps4_buttons()
        speed = self.step_size

        left_x = self._read_axis(PS4_AXIS_LEFT_X)
        left_y = self._read_axis(PS4_AXIS_LEFT_Y)
        right_x = self._read_axis(PS4_AXIS_RIGHT_X)
        right_y = self._read_axis(PS4_AXIS_RIGHT_Y)
        l2 = self._read_trigger(PS4_AXIS_L2, fallback_button=6)
        r2 = self._read_trigger(PS4_AXIS_R2, fallback_button=7)
        l1 = float(self._joystick.get_button(PS4_BUTTON_L1))
        r1 = float(self._joystick.get_button(PS4_BUTTON_R1))

        dx = left_y * MAX_DELTA_POS * speed
        dy = left_x * MAX_DELTA_POS * speed
        dz = -right_y * MAX_DELTA_POS * speed
        droll = (l1 - r1) * MAX_DELTA_ROT * speed
        dpitch = (l2 - r2) * MAX_DELTA_ROT * speed
        dyaw = right_x * MAX_DELTA_ROT * speed
        return dx, dy, dz, droll, dpitch, dyaw

    def _get_input_delta(self) -> tuple[float, float, float, float, float, float]:
        if self.input_device == "ps4":
            return self._get_ps4_delta()
        return self._get_keyboard_delta()

    def _convert_local_rot_delta_to_base(
        self,
        local_rot_delta: np.ndarray,
        state: np.ndarray | None,
    ) -> np.ndarray:
        """将末端系旋转增量转换为基座系 rotvec。

        输入设备的旋转操控按末端自身坐标系解释，但输出 action 语义必须与
        vla_control 保持一致，即 rotvec 在基座坐标系表达。
        """
        local_rot_delta = np.asarray(local_rot_delta, dtype=np.float64)
        if np.linalg.norm(local_rot_delta) < 1e-12:
            return local_rot_delta.copy()

        if state is None or state.shape[0] < 6:
            current_rot = np.eye(3, dtype=np.float64)
        else:
            current_rot = Rotation.from_rotvec(state[3:6]).as_matrix()

        delta_local = Rotation.from_rotvec(local_rot_delta).as_matrix()
        delta_base = current_rot @ delta_local @ current_rot.T
        return Rotation.from_matrix(delta_base).as_rotvec()

    def _convert_rot_delta_to_base(
        self,
        rot_delta: np.ndarray,
        state: np.ndarray | None,
    ) -> np.ndarray:
        """将输入设备旋转增量统一转换为基座系 rotvec。"""
        rot_delta = np.asarray(rot_delta, dtype=np.float64)
        if self.rotation_input_frame == "base":
            return rot_delta.copy()
        return self._convert_local_rot_delta_to_base(rot_delta, state)

    # ------------------------------------------------------------------
    # 输入线程
    # ------------------------------------------------------------------

    def _build_action(
        self,
        dx: float,
        dy: float,
        dz: float,
        droll: float,
        dpitch: float,
        dyaw: float,
        state: np.ndarray | None,
    ) -> np.ndarray:
        delta_rot_base = self._convert_rot_delta_to_base(
            np.array([droll, dpitch, dyaw], dtype=np.float64),
            state,
        )
        action = np.array(
            [dx, dy, dz, *delta_rot_base.tolist(), 0.0],
            dtype=np.float64,
        )
        action[6] = 1.0 if self.gripper_target < 0.04 else -1.0
        return action

    def _input_loop(self):
        """10Hz 输入采样线程。

        这个线程只负责：
        - 读取输入设备
        - 生成当前拍 action
        - 通过 FrankaEnv.enqueue_action() 送给唯一的 skill loop

        它不直接调用 dynamic skill 接口，避免与 env 的执行线程抢占控制面。
        """
        next_tick = time.perf_counter()
        while self.running and not self._stop_event.is_set():
            loop_start = time.perf_counter()
            state = self.env.get_robot_state_vector()
            dx, dy, dz, droll, dpitch, dyaw = self._get_input_delta()
            action = self._build_action(dx, dy, dz, droll, dpitch, dyaw, state)
            self._latest_action = action.copy()
            self.env.enqueue_action(
                action,
                transform=False,
                latest_only=True,
            )
            self._latest_state = state.copy()

            next_tick += INPUT_DT
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
                continue

            # 超时后按当前时刻重置 deadline，避免长期累积漂移。
            if now - loop_start > INPUT_DT:
                logging.warning("Keyboard input loop overran by %.1fms", (now - next_tick) * 1000.0)
            next_tick = now

    # ------------------------------------------------------------------
    # 夹爪 & 复位
    # ------------------------------------------------------------------

    def _close_gripper(self):
        print("  [夹爪] 关闭")
        self.gripper_target = 0.0

    def _open_gripper(self):
        print("  [夹爪] 打开")
        self.gripper_target = 0.08

    def _half_gripper(self):
        print("  [夹爪] 半开 (0.04m)")
        self.gripper_target = 0.04

    def _reset(self):
        """通过 FrankaEnv 串行执行 home + restart，避免和 skill loop 抢接口。"""
        if not self._reset_lock.acquire(blocking=False):
            print("  [复位] 正在复位中，请勿重复操作")
            return

        try:
            print("  [复位] 回到初始位姿...")
            self.gripper_target = 0.08
            self.env.home_and_restart()
            self._latest_state = self.env.get_robot_state_vector()
            self._latest_action = self._build_action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._latest_state)
            print("  [复位] 完成")
        except Exception:
            logging.exception("复位失败")
            print("  [复位] 失败，请查看日志")
        finally:
            self._reset_lock.release()

    # ------------------------------------------------------------------
    # 外部访问接口
    # ------------------------------------------------------------------

    def get_state_and_action(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (state, action)，供数据采集模块调用。

        state  : (8,) [pos(3), rotvec(3), finger1(1), finger2(1)]
        action : (7,) [delta_pos(3), delta_axisangle(3), gripper_cmd(1)]

        action 为当前 10Hz 输入采样拍送入 FrankaEnv 的动作。
        """
        return self._latest_state.copy(), self._latest_action.copy()

#!/usr/bin/env python3
"""
Franka Panda 数据采集框架
=========================
录制状态-动作对，保存为视频 + JSON。

依赖 key_control.py 提供 state/action 数据源。

命令行参数：
    --collection_dir  数据根目录 (默认: 仓库根目录/collected)
    --task_name       任务子目录名   (默认: default)
    --collect_hz      采集频率 Hz     (默认: 10)
    --max_frames      每段轨迹最大帧数 (默认: 1000)

保存格式：
    collection_dir/task_name/
        epo_1/
            cam1.mp4   外部相机视频
            cam2.mp4   腕部相机视频
            data.json  元信息 + 逐帧 state/joint_state/action
        epo_2/
            ...

按键：
    1   开始录制一段轨迹
    2   结束当前轨迹（保存）
    ESC 退出（自动保存未完成轨迹）
"""

import argparse
import json
import logging
import pathlib
import threading
import time
from collections import deque

import imageio
import numpy as np
import pyrealsense2 as rs
from pynput import keyboard


# ==================================================================
# 相机封装（与 franka_env.py 保持一致）
# ==================================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

class D435Camera:
    def __init__(
        self,
        serial_number: str | None = None,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ):
        self._serial = serial_number
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None
        self._latest_frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None

    def start(self):
        try:
            cfg = rs.config()
            if self._serial:
                cfg.enable_device(self._serial)
            cfg.enable_stream(rs.stream.color, self._width, self._height,
                              rs.format.rgb8, self._fps)
            self._pipeline = rs.pipeline()
            self._pipeline.start(cfg)
            self._stop_event.clear()
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
        except RuntimeError as e:
            logging.warning(f"相机 {self._serial} 启动失败: {e}")
            self._pipeline = None

    def _reader_loop(self):
        while not self._stop_event.is_set() and self._pipeline is not None:
            try:
                frames = self._pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
                with self._frame_lock:
                    self._latest_frame = frame.copy()
            except RuntimeError as e:
                logging.warning(f"相机 {self._serial} 读帧失败，将保持上一帧缓存: {e}")
                break

    def get_frame(self) -> np.ndarray:
        with self._frame_lock:
            return self._latest_frame.copy()

    def stop(self):
        self._stop_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None


class DualD435:
    def __init__(self, cam1_serial: str | None = None, cam2_serial: str | None = None):
        self._cam1 = D435Camera(cam1_serial)
        self._cam2 = D435Camera(cam2_serial)

    def start(self):
        self._cam1.start()
        self._cam2.start()

    def get_frames(self):
        return self._cam1.get_frame(), self._cam2.get_frame()

    def stop(self):
        self._cam1.stop()
        self._cam2.stop()


# ==================================================================
# 相机参数（与 vla4desk/coordinator.py 保持一致）
# ==================================================================

CAM1_SERIAL = "346222072769"
CAM2_SERIAL = "938422075745"


# ==================================================================
# 数据录制器
# ==================================================================

class DataRecorder:
    """数据采集框架。

    Args:
        key_control: KeyboardController 实例，提供 get_state_and_action() 方法。
        collection_dir: 数据保存根目录。
        task_name: 任务子目录名。
        collect_hz: 采集频率 Hz。
        max_frames: 每段轨迹最大帧数。
        action_thresh_pos: 位置空动作阈值 (m)，小于此值不记录。
        action_thresh_rot: 旋转空动作阈值 (rad)，小于此值不记录。
        cam1_serial / cam2_serial: 相机序列号。
    """

    # 默认超参数
    COLLECTION_DIR = pathlib.Path(__file__).parent.parent.parent / "collected"
    TASK_NAME = "default"
    COLLECT_HZ = 10.0  # 与 vla_control 保持一致
    MAX_FRAMES = 1000
    ACTION_THRESH_POS = 0.0005    # m
    ACTION_THRESH_ROT = 0.005    # rad
    STATE_THRESH_POS = 0.0005    # m
    STATE_THRESH_ROT = 0.005     # rad
    STATE_THRESH_GRIPPER = 0.0005  # m
    TRAILING_REPEAT_FRAMES = 10
    ACTION_SCALE = 100.0
    PROMPT = ""

    def __init__(
        self,
        key_control,
        collection_dir=None,
        task_name=None,
        collect_hz=None,
        max_frames=None,
        action_thresh_pos=None,
        action_thresh_rot=None,
        cam1_serial=None,
        cam2_serial=None,
        action_scale=None,
        prompt=None,
    ):
        self.kc = key_control

        self.collection_dir = (
            pathlib.Path(collection_dir) if collection_dir else self.COLLECTION_DIR
        )
        self.task_name = task_name or self.TASK_NAME
        self.collect_hz = collect_hz or self.COLLECT_HZ
        self.max_frames = max_frames if max_frames is not None else self.MAX_FRAMES
        self.dt = 1.0 / self.collect_hz
        self.action_scale = float(action_scale) if action_scale is not None else self.ACTION_SCALE
        self.prompt = self.PROMPT if prompt is None else prompt

        self.action_thresh_pos = (
            action_thresh_pos if action_thresh_pos is not None
            else self.ACTION_THRESH_POS
        )
        self.action_thresh_rot = (
            action_thresh_rot if action_thresh_rot is not None
            else self.ACTION_THRESH_ROT
        )

        # 相机
        self.cameras = DualD435(
            cam1_serial or CAM1_SERIAL,
            cam2_serial or CAM2_SERIAL
        )
        self.cameras.start()

        # 录制状态
        self.is_recording = False
        self.recording_frames1: deque = deque(maxlen=self.max_frames)
        self.recording_frames2: deque = deque(maxlen=self.max_frames)
        self.recording_data: list = []
        self.record_lock = threading.Lock()

        # 统计
        self.stats = dict(trajectories_saved=0, total_frames=0, skipped_frames=0)

        self.running = True
        self._exited = False  # 防止 _on_exit 被多次调用
        self._last_recorded_state: np.ndarray | None = None

    def _scale_action_for_storage(self, action: np.ndarray) -> list[float]:
        """仅缩放位姿维度，夹爪命令保持 -1/1 原值。"""
        scaled_action = np.asarray(action, dtype=np.float64).copy()
        scaled_action[:6] *= self.action_scale
        return scaled_action.tolist()

    # ------------------------------------------------------------------
    # 录制控制
    # ------------------------------------------------------------------

    def start_recording(self):
        """开始录制当前轨迹。"""
        if self.is_recording:
            print("  [录制] 已在录制中")
            return
        with self.record_lock:
            self.is_recording = True
            self.recording_frames1.clear()
            self.recording_frames2.clear()
            self.recording_data.clear()
            self._last_recorded_state = None
        print("  [录制] 开始 → 按 2 结束当前轨迹")

    def stop_recording(self):
        """结束并保存当前轨迹。"""
        if not self.is_recording:
            print("  [录制] 当前没有在录制")
            return

        with self.record_lock:
            self.is_recording = False
            frames1 = list(self.recording_frames1)
            frames2 = list(self.recording_frames2)
            data = list(self.recording_data)

        if frames1:
            self._save_trajectory(frames1, frames2, data)

    def _save_trajectory(self, frames1, frames2, data):
        """将一段轨迹写入磁盘"""
        if frames1 and data:
            last_state = np.asarray(data[-1]["state"], dtype=np.float64)
            last_joint_state = np.asarray(data[-1]["joint_state"], dtype=np.float64)
            last_action = np.asarray(data[-1]["action"], dtype=np.float64)
            padded_action = np.zeros_like(last_action)
            padded_action[6] = last_action[6]

            last_frame1 = frames1[-1]
            last_frame2 = frames2[-1]
            for _ in range(self.TRAILING_REPEAT_FRAMES):
                frames1.append(last_frame1.copy())
                frames2.append(last_frame2.copy())
                data.append({
                    "state": last_state.tolist(),
                    "joint_state": last_joint_state.tolist(),
                    "action": padded_action.tolist(),
                })

        task_dir = self.collection_dir / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # 自动递增 episode 编号
        nums = []
        for name in task_dir.iterdir():
            if name.is_dir() and name.name.startswith("epo_"):
                try:
                    nums.append(int(name.name[4:]))
                except ValueError:
                    pass
        next_epo = (max(nums) + 1) if nums else 1
        epo_dir = task_dir / f"epo_{next_epo}"
        epo_dir.mkdir(parents=True, exist_ok=True)

        meta = dict(
            task_name=self.task_name,
            collect_hz=self.collect_hz,
            max_frames=self.max_frames,
            num_frames=len(frames1),
            action_scale=self.action_scale,
            prompt=self.prompt,
            frames=data,
        )

        # 保存视频
        try:
            imageio.mimwrite(str(epo_dir / "cam1.mp4"), frames1,
                             fps=self.collect_hz, codec="libx264",
                             pixelformat="yuv420p")
            imageio.mimwrite(str(epo_dir / "cam2.mp4"), frames2,
                             fps=self.collect_hz, codec="libx264",
                             pixelformat="yuv420p")
            print(f"  [保存] 视频: {epo_dir}")
        except Exception as e:
            logging.error(f"视频保存失败: {e}")

        path_json = epo_dir / "data.json"
        with open(path_json, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  [保存] JSON: {path_json} ({len(data)} 帧)")
        self.stats["trajectories_saved"] += 1
        self.stats["total_frames"] += len(data)
        print(f"  [统计] 已保存 {self.stats['trajectories_saved']} 段, "
              f"总帧 {self.stats['total_frames']}, 跳过 {self.stats['skipped_frames']} 帧")

    def _is_action_empty(self, action: np.ndarray) -> bool:
        """判断是否为空动作（不记录）"""
        pos = action[:3]
        rot = action[3:6]
        return (
            (np.abs(pos) < self.action_thresh_pos).all()
            and (np.abs(rot) < self.action_thresh_rot).all()
        )

    def _is_state_same_as_last_recorded(self, state: np.ndarray) -> bool:
        """判断当前 state 是否与上一条已录制 state 基本一致。"""
        if self._last_recorded_state is None:
            return False

        pos_same = np.all(
            np.abs(state[:3] - self._last_recorded_state[:3]) < self.STATE_THRESH_POS
        )
        rot_same = np.all(
            np.abs(state[3:6] - self._last_recorded_state[3:6]) < self.STATE_THRESH_ROT
        )
        gripper_same = np.all(
            np.abs(state[6:8] - self._last_recorded_state[6:8]) < self.STATE_THRESH_GRIPPER
        )
        return pos_same and rot_same and gripper_same

    # ------------------------------------------------------------------
    # 采集循环（在独立线程运行）
    # ------------------------------------------------------------------

    def run(self):
        """主采集循环，在独立线程运行。"""
        self.running = True
        control_hint = (
            "  1: 开始录制   2: 结束录制   ESC: 退出"
            if self.kc.input_device == "keyboard"
            else "  L3: 开始录制   R3: 结束录制   PS: 退出"
        )

        print("\n" + "=" * 60)
        print("  数据录制器已启动")
        print(f"  保存路径: {self.collection_dir / self.task_name}")
        print(f"  采集频率: {self.collect_hz}Hz  每段最大帧: {self.max_frames}")
        print("=" * 60)
        print(control_hint)
        print("=" * 60 + "\n")

        last_t = time.time()
        frame_count = 0

        while self.kc.running:
            elapsed = time.time() - last_t
            if elapsed >= self.dt:
                last_t += self.dt
                self._collect_frame()
                frame_count += 1
                if frame_count % int(self.collect_hz) == 0:
                    self._print_status()

            time.sleep(0.005)

        self._on_exit()

    def _collect_frame(self):
        state, action = self.kc.get_state_and_action()
        joint_state = self.kc.env.get_joint_state_vector()

        if self._is_action_empty(action) and self._is_state_same_as_last_recorded(state):
            self.stats["skipped_frames"] += 1
            return

        img1, img2 = self.cameras.get_frames()

        recording = False
        with self.record_lock:
            recording = self.is_recording

        if recording:
            self.recording_frames1.append(img1.copy())
            self.recording_frames2.append(img2.copy())
            self.recording_data.append({
                "state": state.tolist(),
                "joint_state": joint_state.tolist(),
                "action": self._scale_action_for_storage(action),
            })
            self._last_recorded_state = np.asarray(state, dtype=np.float64).copy()

            if len(self.recording_frames1) >= self.max_frames:
                print(f"  [录制] 达到最大帧数 {self.max_frames}，自动结束")
                self.stop_recording()

    def _print_status(self):
        state, _ = self.kc.get_state_and_action()
        pos = state[:3]
        gripper = state[6] - state[7]
        status = "REC" if self.is_recording else "IDLE"
        frames = len(self.recording_data) if self.is_recording else 0

        print(f"  [{status}] pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  "
              f"gripper={gripper:.3f}m  epi={self.stats['trajectories_saved']}  "
              f"cur_frames={frames}")

    def _on_exit(self):
        if getattr(self, "_exited", False):
            return
        self._exited = True

        print("\n  [退出] 保存未完成轨迹...")
        with self.record_lock:
            if self.is_recording and self.recording_frames1:
                frames1 = list(self.recording_frames1)
                frames2 = list(self.recording_frames2)
                data = list(self.recording_data)
                self.is_recording = False
                if frames1:
                    self._save_trajectory(frames1, frames2, data)

        self.cameras.stop()
        print("  [退出] 完成")


# ==================================================================
# 入口
# ==================================================================

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Franka 数据采集器")
    parser.add_argument("--collection_dir", default=None)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--collect_hz", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--input_device",
        choices=("keyboard", "ps4"),
        default="keyboard",
        help="控制输入设备，默认 keyboard，可选 ps4。",
    )
    parser.add_argument(
        "--joystick_index",
        type=int,
        default=0,
        help="PS4 手柄索引，默认 0。",
    )
    parser.add_argument(
        "--rotation_input_frame",
        choices=("eef", "base"),
        default="eef",
        help="旋转输入按哪个坐标系解释，默认 eef，可选 base。",
    )
    args = parser.parse_args()

    from key_control import KeyboardController
    print(f"[DEBUG] 创建控制器... input_device={args.input_device}")
    kc = KeyboardController(
        input_device=args.input_device,
        joystick_index=args.joystick_index,
        rotation_input_frame=args.rotation_input_frame,
    )
    print("[DEBUG] 启动控制线程...")
    kc.start()
    time.sleep(0.5)
    print(f"[DEBUG] kc.running={kc.running}, input_thread={kc._input_thread}, "
          f"alive={kc._input_thread.is_alive() if kc._input_thread else None}")

    recorder = DataRecorder(
        key_control=kc,
        collection_dir=args.collection_dir,
        task_name=args.task_name,
        collect_hz=args.collect_hz,
        max_frames=args.max_frames,
        action_scale=args.action_scale,
        prompt=args.prompt,
    )

    # 采集循环独立线程
    rec_thread = threading.Thread(target=recorder.run, daemon=True)
    rec_thread.start()

    if args.input_device == "keyboard":
        _orig_on_press = kc._on_key_press

        def wrapped_on_press(key):
            try:
                char = key.char.lower()
            except AttributeError:
                char = key.name.lower()
            if char == '1':
                recorder.start_recording()
                return
            if char == '2':
                recorder.stop_recording()
                return
            _orig_on_press(key)

        print("[DEBUG] 启动键盘监听...")
        with keyboard.Listener(
            on_press=wrapped_on_press,
            on_release=kc._on_key_release,
        ) as listener:
            listener.join()
    else:
        kc.bind_event("record_start", recorder.start_recording)
        kc.bind_event("record_stop", recorder.stop_recording)
        print("[DEBUG] PS4 手柄模式运行中...")
        try:
            while kc.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            kc.stop()

    kc.stop()
    recorder._on_exit()
    print("程序已退出。")


if __name__ == "__main__":
    main()

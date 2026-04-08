#!/usr/bin/env python3
"""
Franka Panda 键盘遥操作控制器
=============================
提供：
  - 20Hz dynamic skill 控制循环（发布 sensor message）
  - 按键 → 位姿增量的映射（末端坐标系旋转，世界坐标系位置）
  - 命令位姿追踪（commanded_pose / prev_commanded_pose）

本文件不涉及录制逻辑，供 data_recorder.py 等模块引用。
"""

import logging
import math
import threading
import time

import numpy as np

from pynput import keyboard
from scipy.spatial.transform import Rotation

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType, FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import (
    PosePositionSensorMessage,
    ShouldTerminateSensorMessage,
    CartesianImpedanceSensorMessage,
)


# ==================================================================
# 控制参数
# ==================================================================

PUBLISH_DT = 0.05           # 发布间隔 50ms (20Hz)
MAX_LIN_VEL = 0.1          # 最大线速度 0.1 m/s
MAX_ROT_VEL = math.pi / 4  # 最大角速度 45°/s

DEFAULT_TRANS_STIFF = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES
DEFAULT_ROT_STIFF = FC.DEFAULT_ROTATIONAL_STIFFNESSES

# 每步最大增量（由 PUBLISH_DT 决定）
MAX_DELTA_POS = MAX_LIN_VEL * PUBLISH_DT   # 0.005 m
MAX_DELTA_ROT = MAX_ROT_VEL * PUBLISH_DT   # ≈ 0.039 rad


# ==================================================================
# 键盘控制器
# ==================================================================

class KeyboardController:
    """独立的键盘遥操作控制类。

    使用方式（推荐）：
        kc = KeyboardController()
        kc.start()                # 启动控制线程（非阻塞）
        # 自行管理 keyboard.Listener
        with keyboard.Listener(...) as listener:
            listener.join()
        kc.stop()

    公开属性（线程安全）：
        commanded_pose      RigidTransform  当前目标位姿
        prev_commanded_pose RigidTransform 上一拍目标位姿（用于计算 action）
        gripper_target     float          夹爪目标宽度 (m)
        step_size          float          速度倍率 [0.1, 1.0]
        running            bool           程序是否在运行
    """

    def __init__(self):
        self.fa = FrankaArm(with_gripper=True)
        self.fa.reset_joints()

        self.gripper_target = 0.04   # 默认半开
        self.step_size = 0.7          # 速度倍率：3档 [0.4, 0.7, 1.0]
        self._speed_levels = [0.4, 0.7, 1.0]  # 3档速度：40%, 70%, 100%
        self._speed_index = 1                  # 默认第2档（70%）
        self.running = True

        # 按键状态
        self.keys_pressed: set = set()
        self._keys_lock = threading.Lock()

        # 命令位姿（由 _publish_loop 更新，外部读取）
        self._pose_lock = threading.Lock()
        self._current_pose: RigidTransform | None = None
        self._commanded_pose: RigidTransform | None = None
        self._prev_commanded_pose: RigidTransform | None = None

        # 线程管理
        self._pub_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # 状态缓存
        self._cached_gripper_width = self.fa.get_gripper_width()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self):
        """启动控制线程（daemon），不阻塞。"""
        if self._pub_thread is None or not self._pub_thread.is_alive():
            self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._pub_thread.start()

    def stop(self):
        """停止控制线程，等待其退出。"""
        self.running = False
        self._stop_event.set()
        if self._pub_thread and self._pub_thread.is_alive():
            self._pub_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # 键盘回调（供外部 Listener 注入）
    # ------------------------------------------------------------------

    def _on_key_press(self, key):
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

    def _on_key_release(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()
        with self._keys_lock:
            self.keys_pressed.discard(char)

    # ------------------------------------------------------------------
    # 控制线程
    # ------------------------------------------------------------------

    def _publish_loop(self):
        """20Hz 动态控制循环：读取按键 → 更新 commanded_pose → 发布 sensor message"""
        fa = self.fa
        seq_id = 0

        # 初始化位姿
        self._current_pose = fa.get_pose()
        self._commanded_pose = self._current_pose.copy()
        self._prev_commanded_pose = self._current_pose.copy()

        # 启动 dynamic skill
        fa.goto_pose(
            self._commanded_pose,
            duration=1000.0,
            dynamic=True,
            buffer_time=20.0,
            cartesian_impedances=DEFAULT_TRANS_STIFF + DEFAULT_ROT_STIFF,
            block=False
        )
        init_time = fa.get_time()

        while self.running and not self._stop_event.is_set():
            t_loop_start = time.time()
            timestamp = fa.get_time() - init_time

            # 读取按键状态（安全复制）
            with self._keys_lock:
                keys = set(self.keys_pressed)
            speed = self.step_size

            # 计算位姿增量
            dx = (int('w' in keys) - int('s' in keys)) * MAX_DELTA_POS * speed
            dy = (int('a' in keys) - int('d' in keys)) * MAX_DELTA_POS * speed
            dz = (int('q' in keys) - int('e' in keys)) * MAX_DELTA_POS * speed
            droll  = (int('u' in keys) - int('o' in keys)) * MAX_DELTA_ROT * speed
            dpitch = (int('i' in keys) - int('k' in keys)) * MAX_DELTA_ROT * speed
            dyaw   = (int('l' in keys) - int('j' in keys)) * MAX_DELTA_ROT * speed

            moved = dx or dy or dz or droll or dpitch or dyaw

            with self._pose_lock:
                if moved:
                    self._prev_commanded_pose = self._commanded_pose.copy()

                    # 位置增量：直接累加（基座坐标系）
                    self._commanded_pose.translation[0] += dx
                    self._commanded_pose.translation[1] += dy
                    self._commanded_pose.translation[2] += dz

                    # 旋转增量：基座坐标系，左乘（与 vla_control 一致）
                    if droll or dpitch or dyaw:
                        delta_rotvec = np.array([droll, dpitch, dyaw])
                        angle = np.linalg.norm(delta_rotvec)
                        if angle > 1e-6:
                            delta_rot = Rotation.from_rotvec(delta_rotvec).as_matrix()
                            # 左乘：delta 在基座坐标系表达
                            self._commanded_pose.rotation = delta_rot @ self._commanded_pose.rotation

                # 更新缓存
                self._current_pose = fa.get_pose()
                self._cached_gripper_width = fa.get_gripper_width()

            # 发布 sensor message
            msg_pose = PosePositionSensorMessage(
                id=seq_id, timestamp=timestamp,
                position=self._commanded_pose.translation,
                quaternion=self._commanded_pose.quaternion
            )
            msg_imp = CartesianImpedanceSensorMessage(
                id=seq_id, timestamp=timestamp,
                translational_stiffnesses=DEFAULT_TRANS_STIFF,
                rotational_stiffnesses=DEFAULT_ROT_STIFF
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    msg_pose, SensorDataMessageType.POSE_POSITION),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    msg_imp, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )
            fa.publish_sensor_data(ros_msg)
            seq_id += 1

            # 精确延时
            elapsed = time.time() - t_loop_start
            time.sleep(max(0, PUBLISH_DT - elapsed))

        # 停止 dynamic skill
        term_msg = ShouldTerminateSensorMessage(
            timestamp=fa.get_time() - init_time, should_terminate=True
        )
        term_ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
        fa.publish_sensor_data(term_ros_msg)
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # 夹爪 & 复位
    # ------------------------------------------------------------------

    def _close_gripper(self):
        print("  [夹爪] 关闭")
        self.gripper_target = 0.0
        self.fa.goto_gripper(width=0.0, grasp=True, block=False)

    def _open_gripper(self):
        print("  [夹爪] 打开")
        self.gripper_target = 0.08
        self.fa.open_gripper(block=False)

    def _half_gripper(self):
        print("  [夹爪] 半开 (0.04m)")
        self.gripper_target = 0.04
        self.fa.goto_gripper(width=0.04, block=False)

    def _reset(self):
        """平滑回到初始位姿（home）并打开夹爪。"""
        print("  [复位] 回到初始位姿...")
        
        # 1. 打开夹爪
        self.gripper_target = 0.08
        self.fa.open_gripper(block=False)
        
        # 2. 通过 goto_pose 平滑移动到初始位姿
        # 先停止当前的 dynamic skill
        fa = self.fa
        term_msg = ShouldTerminateSensorMessage(
            timestamp=fa.get_time(), should_terminate=True
        )
        term_ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
        fa.publish_sensor_data(term_ros_msg)
        time.sleep(0.5)
        
        # 回到 home 位姿（使用 FrankaConstants 的默认位姿）
        home_pose = fa.get_pose()  # 获取当前位姿
        # 使用 reset_joints 获取标准 home 位姿，但不阻塞
        try:
            fa.reset_joints()
        except Exception as e:
            print(f"  [复位] 警告: reset_joints 异常: {e}")
            # 如果失败，尝试使用 goto_joints
            from frankapy import FrankaConstants as FC
            fa.goto_joints(FC.DEFAULT_JOINT_POSITIONS, duration=3.0, block=True)
        
        time.sleep(0.5)
        
        # 3. 更新 commanded_pose 为新的 home 位姿
        self._current_pose = self.fa.get_pose()
        with self._pose_lock:
            self._commanded_pose = self._current_pose.copy()
            self._prev_commanded_pose = self._current_pose.copy()
        
        # 4. 重新启动 dynamic skill
        self._stop_event.clear()
        self.running = True
        if self._pub_thread is None or not self._pub_thread.is_alive():
            self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._pub_thread.start()
        
        print("  [复位] 完成")

    # ------------------------------------------------------------------
    # 外部访问接口
    # ------------------------------------------------------------------

    def get_state_and_action(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (state, action)，供数据采集模块调用。

        state  : (8,) [pos(3), rotvec(3), finger1(1), finger2(1)]
        action : (7,) [delta_pos(3), delta_axisangle(3), gripper_cmd(1)]

        其中 action = commanded_pose - prev_commanded_pose（用户输入的位姿增量）。
        旋转增量在基座坐标系中表达（与 vla_control 一致）。
        """
        with self._pose_lock:
            pos = self._current_pose.translation
            rotvec = Rotation.from_matrix(
                self._current_pose.rotation
            ).as_rotvec()

            half = self._cached_gripper_width / 2.0
            state = np.concatenate([pos, rotvec, [half], [-half]])

            delta = np.zeros(7)
            if self._prev_commanded_pose is not None:
                # 位置增量：基座坐标系
                delta[:3] = (
                    self._commanded_pose.translation
                    - self._prev_commanded_pose.translation
                )
                # 旋转增量：基座坐标系（与 vla_control 的 _compute_target_pose 一致）
                # diff_rot = R_current @ R_prev.T 表示从 prev 到 current 的旋转（在基座坐标系）
                diff_rot = (
                    self._commanded_pose.rotation
                    @ self._prev_commanded_pose.rotation.T
                )
                delta[3:6] = Rotation.from_matrix(diff_rot).as_rotvec()
            
            # 夹爪目标：二值化（与 vla_control 的 transform_action 一致）
            # vla_control: 0.0 if ta[6] >= 0 else 0.08（>=0 为闭合，<0 为打开）
            # 但 data_collection 中 gripper_target 始终 >= 0，所以调整为：
            # >= 0.04 为打开 (0.08)，< 0.04 为闭合 (0.0)
            delta[6] = 0.08 if self.gripper_target >= 0.04 else 0.0

            return state, delta

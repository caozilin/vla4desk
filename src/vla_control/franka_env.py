"""FrankaEnvironment：obs 采集 + action 执行，基于 libero 的字段格式。

obs 格式（发给 openpi 服务端）：
    observation/image       : (224, 224, 3) uint8  外部相机
    observation/wrist_image : (224, 224, 3) uint8  腕部相机
    observation/state       : (8,) float64  [eef_pos(3), eef_axisangle(3), finger1_qpos(1), finger2_qpos(1)]
    observation/joints      : (7,) float64  机械臂 7 维关节角
    prompt                  : str

action 格式（从 openpi 服务端接收）：
    actions : (H, 7) float64  [delta_pos(3), delta_axisangle(3), gripper(1)]
"""
import logging
import math
import pathlib
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "client"))

import image_tools
import numpy as np

try:
    from autolab_core import RigidTransform
    from frankapy import FrankaArm, SensorDataMessageType
    from frankapy import FrankaConstants as FC
    from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
    from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
    from frankapy.utils import min_jerk_weight
    HAS_FRANKA = True
except ImportError:
    HAS_FRANKA = False

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


from scipy.spatial.transform import Rotation

CONTROL_DT = 0.1   # 每个 action 的执行周期（s），决定控制频率（10Hz）
INTERP_DT = 0.02   # 50Hz：与 key_control 的发布频率一致
INTERP_STEPS = int(CONTROL_DT / INTERP_DT)  # = 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
DEFAULT_CAM1_SERIAL = "346222072769"
DEFAULT_CAM2_SERIAL = "938422075745"
D435_START_RETRIES = 5
D435_START_RETRY_DELAY = 0.5

POS_CLIP = 1.0          # 位置 action 输入截断范围 [-POS_CLIP, POS_CLIP]
ROT_CLIP = 6.0          # 旋转 action 输入截断范围 [-ROT_CLIP, ROT_CLIP]
POS_MAX_VEL = 0.1       # 末端位置最大速度 (m/s)
ROT_MAX_VEL = (math.pi / 4)    # 末端旋转最大速度 (rad/s)
GRIPPER_SPEED = 0.08    # 夹爪运动速度 (m/s)，1s 走完全程 0.08m
GRIPPER_WIDTH_MAX = 0.08
POS_SCALE = POS_MAX_VEL * CONTROL_DT / POS_CLIP
ROT_SCALE = ROT_MAX_VEL * CONTROL_DT / ROT_CLIP
MAX_TRANSLATION_ERROR = 0.03  # m
MAX_ROTATION_ERROR = 0.35 # rad

# 阻抗控制刚度参数，与 key_control 保持一致
TRANSLATIONAL_STIFFNESS = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES if HAS_FRANKA else [600.0, 600.0, 600.0]
ROTATIONAL_STIFFNESS = FC.DEFAULT_ROTATIONAL_STIFFNESSES if HAS_FRANKA else [50.0, 50.0, 50.0]


def transform_action(action: np.ndarray) -> np.ndarray:
    """将原始 action [7] 裁剪并缩放为机器人可执行的指令。"""
    ta = action.copy()
    ta[:3] = np.clip(ta[:3], -POS_CLIP, POS_CLIP) * POS_SCALE
    ta[3:6] = np.clip(ta[3:6], -ROT_CLIP, ROT_CLIP) * ROT_SCALE
    ta[6] = 0.0 if ta[6] >= 0 else GRIPPER_WIDTH_MAX
    return ta


@dataclass
class ActionTransformConfig:
    pos_clip: float = POS_CLIP
    rot_clip: float = ROT_CLIP
    pos_max_vel: float = POS_MAX_VEL
    rot_max_vel: float = ROT_MAX_VEL


@dataclass
class ControlTraceSample:
    seq_id: int
    timestamp: float
    state: np.ndarray
    joint_state: np.ndarray
    action: np.ndarray
    commanded_pose: np.ndarray


def _apply_action_transform(
    action: np.ndarray,
    config: ActionTransformConfig,
    *,
    scale_motion: bool = True,
) -> np.ndarray:
    ta = np.asarray(action, dtype=np.float64).copy()
    if scale_motion:
        pos_scale = config.pos_max_vel * CONTROL_DT / config.pos_clip
        rot_scale = config.rot_max_vel * CONTROL_DT / config.rot_clip
        ta[:3] = np.clip(ta[:3], -config.pos_clip, config.pos_clip) * pos_scale
        ta[3:6] = np.clip(ta[3:6], -config.rot_clip, config.rot_clip) * rot_scale
    ta[6] = 0.0 if ta[6] >= 0 else GRIPPER_WIDTH_MAX
    return ta


class D435Camera:
    """单路 D435，pyrealsense2 直连 USB。serial_number=None 时自动选第一个设备。"""

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

    def _release_pipeline(self):
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def start(self):
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 未安装，无法启动相机")
        if self._pipeline is not None:
            self.stop()

        last_error: Exception | None = None
        for attempt in range(1, D435_START_RETRIES + 1):
            pipeline = rs.pipeline()
            try:
                config = rs.config()
                if self._serial:
                    config.enable_device(self._serial)
                config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
                pipeline.start(config)
                self._pipeline = pipeline
                self._stop_event.clear()
                self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._reader_thread.start()
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "相机 %s 启动失败（第 %d/%d 次）: %s",
                    self._serial or "<auto>",
                    attempt,
                    D435_START_RETRIES,
                    e,
                )
                self._release_pipeline()
                if attempt < D435_START_RETRIES:
                    time.sleep(D435_START_RETRY_DELAY)

        raise RuntimeError(
            f"相机 {self._serial or '<auto>'} 启动失败，已重试 {D435_START_RETRIES} 次: {last_error}"
        )

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
            except Exception as e:
                logger.warning(f"相机读帧失败，将保持上一帧缓存：{e}")
                self._stop_event.set()
                self._release_pipeline()
                break

    def get_frame(self) -> np.ndarray:
        with self._frame_lock:
            return self._latest_frame.copy()

    def stop(self):
        self._stop_event.set()
        if (
            self._reader_thread is not None
            and self._reader_thread.is_alive()
            and threading.current_thread() is not self._reader_thread
        ):
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None
        self._release_pipeline()


class DualD435:
    """双路 D435 封装，某路 serial 为 None 或启动失败时返回全黑帧。"""

    def __init__(self, cam1_serial: str | None = None, cam2_serial: str | None = None):
        self._cam1 = D435Camera(cam1_serial)
        self._cam2 = D435Camera(cam2_serial)
        self._black = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    def start(self):
        for cam in (self._cam1, self._cam2):
            try:
                cam.start()
            except Exception as e:
                logger.warning(f"相机启动失败，将使用全黑帧：{e}")
            time.sleep(0.2)

    def stop(self):
        for cam in (self._cam1, self._cam2):
            try:
                cam.stop()
            except Exception:
                pass

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (frame1, frame2)，某路不可用时返回全黑帧。"""
        frames = []
        for cam in (self._cam1, self._cam2):
            if cam._pipeline is None:
                frames.append(self._black.copy())
            else:
                try:
                    frames.append(cam.get_frame())
                except Exception as e:
                    logger.warning(f"相机读帧失败，返回全黑帧：{e}")
                    cam._pipeline = None
                    frames.append(self._black.copy())
        return frames[0], frames[1]


RESIZE = 224


def normalize_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """对单帧 rotvec 做规范化。

    这个函数只做单帧 canonicalization：
    - 保持旋转轴方向不变
    - 将旋转角度折叠到 [-pi, pi]

    它不依赖历史帧，因此不负责跨帧连续性。若需要尽量避免相邻帧在
    ±pi 附近发生跳变，应使用 normalize_rotvec_with_history()。
    """
    rotvec = np.asarray(rotvec, dtype=np.float64).copy()

    angle = np.linalg.norm(rotvec)

    if angle < 1e-6:
        return rotvec

    axis = rotvec / angle
    angle = np.angle(np.exp(1j * angle))
    return axis * angle


def normalize_rotvec_with_history(rotvec: np.ndarray, prev_rotvec: np.ndarray | None) -> np.ndarray:
    """在等价轴角表示中选取与上一帧最接近的分支。

    只保证时间连续性，不引入任何分布偏置：
    - 无历史帧时，返回单帧 canonical rotvec
    - 有历史帧时，在当前旋转的等价 rotvec 中选择离上一帧最近的表示

    返回值允许超出 [-pi, pi]，这是连续性的代价。
    """
    rotvec = np.asarray(rotvec, dtype=np.float64).copy()

    if prev_rotvec is None:
        return normalize_rotvec(rotvec)

    base = normalize_rotvec(rotvec)
    angle = np.linalg.norm(base)
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float64)

    axis = base / angle
    candidates: list[np.ndarray] = []
    for wrap in range(-2, 3):
        two_pi = 2.0 * np.pi * wrap
        candidates.append(axis * (angle + two_pi))
        candidates.append(-axis * ((2.0 * np.pi - angle) + two_pi))

    prev_rotvec = np.asarray(prev_rotvec, dtype=np.float64)
    return min(candidates, key=lambda candidate: np.linalg.norm(candidate - prev_rotvec))


class FrankaEnv:
    """封装机械臂状态读取与 dynamic pose 控制。"""

    def __init__(
        self,
        cam1_serial: str | None = DEFAULT_CAM1_SERIAL,
        cam2_serial: str | None = DEFAULT_CAM2_SERIAL,
        dynamic_duration: float = 3600.0,
        no_robot: bool = False,
        pos_clip: float = POS_CLIP,
        rot_clip: float = ROT_CLIP,
        pos_max_vel: float = POS_MAX_VEL,
        rot_max_vel: float = ROT_MAX_VEL,
    ):
        if no_robot or not HAS_FRANKA:
            if not no_robot:
                logger.warning("frankapy 未安装")
            self._fa = None
        else:
            self._fa = FrankaArm()

        self._cameras = DualD435(cam1_serial, cam2_serial)
        self._dynamic_duration = dynamic_duration
        self._action_transform = ActionTransformConfig(
            pos_clip=pos_clip,
            rot_clip=rot_clip,
            pos_max_vel=pos_max_vel,
            rot_max_vel=rot_max_vel,
        )

        # 动作队列：coordinator 往里放 action，_skill_thread 消费
        self._action_queue: queue.Queue = queue.Queue()
        self._skill_thread: threading.Thread | None = None
        self._skill_stop = threading.Event()
        self._lock = threading.Lock()
        self._fa_api_lock = threading.RLock()
        self._commanded_pose_array = np.zeros(6)
        
        # 实际位姿缓存：_skill_loop 定期更新，get_observation 读取，避免重复调用 ROS API
        self._cached_pose: np.ndarray | None = None  # [pos(3), rot_axisangle(3)]
        self._cached_joints: np.ndarray | None = None  # (7,)
        self._cached_gripper_width: float | None = None  # 单指开口(m)

        # 末端力矩缓存：[force(3), torque(3)]，单位 [N, Nm]
        self._cached_ee_force_torque: np.ndarray | None = None  # (6,)

        # 最近一次被执行线程真正采用的控制快照
        self._executed_action = np.zeros(7, dtype=np.float64)
        self._executed_commanded_pose = np.zeros(6, dtype=np.float64)
        self._control_trace_seq = 0
        self._control_trace_history: deque[ControlTraceSample] = deque(maxlen=4096)
        self._control_trace_sample = ControlTraceSample(
            seq_id=0,
            timestamp=0.0,
            state=np.zeros(8, dtype=np.float64),
            joint_state=np.zeros(7, dtype=np.float64),
            action=np.zeros(7, dtype=np.float64),
            commanded_pose=np.zeros(6, dtype=np.float64),
        )

        # 旋转矢量历史缓存：存原始 rotvec，供 normalize_rotvec_with_history 做跨帧连续化
        self._prev_rotvec: np.ndarray | None = None
        self._prev_commanded_rotvec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self):
        self._cameras.start()

    def stop(self):
        self.stop_control()

    def start_control(self, *, home_first: bool = False):
        self.start()
        if self._fa is not None:
            if home_first:
                logger.info("启动控制前回到 home...")
                with self._fa_api_lock:
                    self._fa.reset_joints()
            self._refresh_robot_cache()
        self.start_skill_thread()

    def stop_control(self):
        self._stop_skill_thread()
        self._cameras.stop()

    def reset_to_home(self):
        """结束 dynamic skill 线程，回到 home 位姿。"""
        if self._fa is None:
            logger.warning("机械臂不可用，跳过 reset_to_home")
            return
        self._stop_skill_thread()
        self._wait_for_skill_idle()
        with self._fa_api_lock:
            self._fa.reset_joints()
        self._refresh_robot_cache()

    def home_and_restart(self):
        self.reset_to_home()
        self.start_skill_thread()

    def set_action_transform(
        self,
        *,
        pos_clip: float | None = None,
        rot_clip: float | None = None,
        pos_max_vel: float | None = None,
        rot_max_vel: float | None = None,
    ):
        if pos_clip is not None:
            self._action_transform.pos_clip = pos_clip
        if rot_clip is not None:
            self._action_transform.rot_clip = rot_clip
        if pos_max_vel is not None:
            self._action_transform.pos_max_vel = pos_max_vel
        if rot_max_vel is not None:
            self._action_transform.rot_max_vel = rot_max_vel

    def transform_action(
        self,
        action: np.ndarray,
        *,
        scale_motion: bool = True,
    ) -> np.ndarray:
        return _apply_action_transform(
            action,
            self._action_transform,
            scale_motion=scale_motion,
        )

    def _resize_observation_image(self, image: np.ndarray) -> np.ndarray:
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, RESIZE, RESIZE))

    def _pose_to_rotvec(self, pose: "RigidTransform") -> np.ndarray:
        return Rotation.from_matrix(pose.rotation).as_rotvec()

    def _set_commanded_pose_array(self, pose: "RigidTransform"):
        raw_rot_vec = self._pose_to_rotvec(pose)
        rot_vec = normalize_rotvec_with_history(raw_rot_vec, self._prev_commanded_rotvec)
        self._commanded_pose_array = np.concatenate([pose.translation, rot_vec])
        self._prev_commanded_rotvec = rot_vec.copy()

    def _read_robot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with self._fa_api_lock:
            pose_rt = self._fa.get_pose()
            joints = self._fa.get_joints()
            half = self._fa.get_gripper_width() / 2.0
        return pose_rt.translation, self._pose_to_rotvec(pose_rt), joints, half

    def _get_robot_state_for_observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with self._lock:
            if (
                self._cached_pose is not None
                and self._cached_joints is not None
                and self._cached_gripper_width is not None
            ):
                return (
                    self._cached_pose[:3],
                    self._cached_pose[3:6],
                    self._cached_joints,
                    self._cached_gripper_width,
                )
        return self._read_robot_state()

    def _build_observation_state(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        half_gripper_width: float,
    ) -> np.ndarray:
        return np.concatenate([
            pos,
            rot,
            [+half_gripper_width],
            [-half_gripper_width],
        ])

    def _build_control_trace_sample(
        self,
        *,
        timestamp: float,
        state: np.ndarray,
        joint_state: np.ndarray,
        action: np.ndarray,
        commanded_pose: np.ndarray,
    ) -> ControlTraceSample:
        return ControlTraceSample(
            seq_id=0,
            timestamp=float(timestamp),
            state=np.asarray(state, dtype=np.float64).copy(),
            joint_state=np.asarray(joint_state, dtype=np.float64).copy(),
            action=np.asarray(action, dtype=np.float64).copy(),
            commanded_pose=np.asarray(commanded_pose, dtype=np.float64).copy(),
        )

    def _record_control_trace(
        self,
        *,
        timestamp: float,
        state: np.ndarray,
        joint_state: np.ndarray,
        action: np.ndarray,
        commanded_pose: np.ndarray,
    ):
        sample = self._build_control_trace_sample(
            timestamp=timestamp,
            state=state,
            joint_state=joint_state,
            action=action,
            commanded_pose=commanded_pose,
        )
        with self._lock:
            self._control_trace_seq += 1
            sample.seq_id = self._control_trace_seq
            self._control_trace_sample = sample
            self._control_trace_history.append(sample)

    def _snapshot_cached_robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        pos, rot, joints, half = self._get_robot_state_for_observation()
        return (
            self._build_observation_state(pos, rot, half).astype(np.float64),
            np.asarray(joints, dtype=np.float64).copy(),
        )

    def _apply_delta_to_pose_array(self, pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
        next_pose = np.asarray(pose, dtype=np.float64).copy()
        delta = np.asarray(delta, dtype=np.float64)
        next_pose[:3] += delta[:3]

        angle = np.linalg.norm(delta[3:6])
        if angle > 1e-6:
            delta_rot = Rotation.from_rotvec(delta[3:6]).as_matrix()
            current_rot = Rotation.from_rotvec(next_pose[3:6]).as_matrix()
            next_pose[3:6] = Rotation.from_matrix(delta_rot @ current_rot).as_rotvec()
        return next_pose

    def _compute_translation_error_scale(
        self,
        actual_pos: np.ndarray,
        commanded_pose: "RigidTransform",
        transformed_action: np.ndarray,
    ) -> float:
        current_error = np.asarray(commanded_pose.translation, dtype=np.float64) - np.asarray(actual_pos, dtype=np.float64)
        delta_pos = np.asarray(transformed_action[:3], dtype=np.float64)
        predicted_error = current_error + delta_pos

        predicted_norm = float(np.linalg.norm(predicted_error))
        if predicted_norm <= MAX_TRANSLATION_ERROR:
            return 1.0

        delta_norm_sq = float(np.dot(delta_pos, delta_pos))
        if delta_norm_sq <= 1e-12:
            return 1.0

        current_norm = float(np.linalg.norm(current_error))
        if current_norm >= MAX_TRANSLATION_ERROR:
            # 当前误差已经超限时，只允许继续执行能减小误差的平移分量。
            if float(np.dot(current_error, delta_pos)) >= 0.0:
                return 0.0

        a = delta_norm_sq
        b = 2.0 * float(np.dot(current_error, delta_pos))
        c = float(np.dot(current_error, current_error)) - MAX_TRANSLATION_ERROR ** 2
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return 0.0

        sqrt_disc = math.sqrt(max(discriminant, 0.0))
        roots = [
            (-b - sqrt_disc) / (2.0 * a),
            (-b + sqrt_disc) / (2.0 * a),
        ]
        valid_roots = [root for root in roots if 0.0 <= root <= 1.0]
        if not valid_roots:
            return 0.0
        return float(max(valid_roots))

    def _compute_rotation_error_norm(
        self,
        actual_rotvec: np.ndarray,
        commanded_pose: "RigidTransform",
        delta_rotvec: np.ndarray | None = None,
        scale: float = 1.0,
    ) -> float:
        actual_rot = Rotation.from_rotvec(np.asarray(actual_rotvec, dtype=np.float64))
        commanded_rot = Rotation.from_matrix(commanded_pose.rotation)
        if delta_rotvec is not None:
            commanded_rot = Rotation.from_rotvec(np.asarray(delta_rotvec, dtype=np.float64) * scale) * commanded_rot
        error_rot = commanded_rot * actual_rot.inv()
        return float(np.linalg.norm(error_rot.as_rotvec()))

    def _compute_rotation_error_scale(
        self,
        actual_rotvec: np.ndarray,
        commanded_pose: "RigidTransform",
        transformed_action: np.ndarray,
    ) -> float:
        delta_rotvec = np.asarray(transformed_action[3:6], dtype=np.float64)
        if float(np.dot(delta_rotvec, delta_rotvec)) <= 1e-12:
            return 1.0

        current_norm = self._compute_rotation_error_norm(actual_rotvec, commanded_pose)
        predicted_norm = self._compute_rotation_error_norm(
            actual_rotvec,
            commanded_pose,
            delta_rotvec,
            scale=1.0,
        )
        if predicted_norm <= MAX_ROTATION_ERROR:
            return 1.0
        if current_norm >= MAX_ROTATION_ERROR and predicted_norm >= current_norm:
            return 0.0

        sample_count = 64
        scales = np.linspace(0.0, 1.0, sample_count + 1)
        norms = [
            self._compute_rotation_error_norm(
                actual_rotvec,
                commanded_pose,
                delta_rotvec,
                scale=float(scale),
            )
            for scale in scales
        ]
        feasible_indices = [idx for idx, norm in enumerate(norms) if norm <= MAX_ROTATION_ERROR]
        if not feasible_indices:
            return 0.0

        best_idx = max(feasible_indices)
        best_scale = float(scales[best_idx])
        if best_idx == sample_count:
            return 1.0

        low = best_scale
        high = float(scales[best_idx + 1])
        for _ in range(20):
            mid = 0.5 * (low + high)
            mid_norm = self._compute_rotation_error_norm(
                actual_rotvec,
                commanded_pose,
                delta_rotvec,
                scale=mid,
            )
            if mid_norm <= MAX_ROTATION_ERROR:
                low = mid
            else:
                high = mid
        return float(low)

    def _limit_action_by_pose_error(
        self,
        actual_pos: np.ndarray,
        actual_rotvec: np.ndarray,
        commanded_pose: "RigidTransform",
        transformed_action: np.ndarray,
        raw_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        translation_scale = self._compute_translation_error_scale(actual_pos, commanded_pose, transformed_action)
        rotation_scale = self._compute_rotation_error_scale(actual_rotvec, commanded_pose, transformed_action)
        scale = min(translation_scale, rotation_scale)
        if scale >= 1.0:
            return transformed_action, raw_action

        limited_transformed_action = np.asarray(transformed_action, dtype=np.float64).copy()
        limited_raw_action = np.asarray(raw_action, dtype=np.float64).copy()
        limited_transformed_action[:6] *= scale
        limited_raw_action[:6] *= scale
        return limited_transformed_action, limited_raw_action

    def get_robot_state_vector(self) -> np.ndarray:
        """返回当前机器人 state 向量，格式与 observation/state 一致。"""
        if self._fa is None:
            return np.zeros(8, dtype=np.float64)
        pos, rot, _joints, half = self._get_robot_state_for_observation()
        return self._build_observation_state(pos, rot, half).astype(np.float64)

    def get_joint_state_vector(self) -> np.ndarray:
        """返回当前机器人关节状态向量，格式与 observation/joints 一致。"""
        if self._fa is None:
            return np.zeros(7, dtype=np.float64)
        _pos, _rot, joints, _half = self._get_robot_state_for_observation()
        return np.asarray(joints, dtype=np.float64).copy()

    def _refresh_robot_cache(self):
        with self._fa_api_lock:
            actual_pose = self._fa.get_pose()
            actual_joints = self._fa.get_joints()
            actual_gripper_half = self._fa.get_gripper_width() / 2.0
            ee_force_torque = self._fa.get_ee_force_torque()

        with self._lock:
            raw_rot_vec = self._pose_to_rotvec(actual_pose)
            rot_vec = normalize_rotvec_with_history(raw_rot_vec, self._prev_rotvec)

            self._cached_pose = np.concatenate([actual_pose.translation, rot_vec])
            self._cached_joints = actual_joints
            self._cached_gripper_width = actual_gripper_half
            self._cached_ee_force_torque = ee_force_torque
            self._prev_rotvec = rot_vec.copy()

    def _start_dynamic_skill(self, fa: "FrankaArm") -> tuple["RigidTransform", float]:
        logger.info("等待机械臂状态同步...")
        time.sleep(1.0)
        with self._fa_api_lock:
            current_pose = fa.get_pose()

        logger.info("启动 dynamic skill...")
        with self._fa_api_lock:
            fa.goto_pose(
                current_pose,
                duration=self._dynamic_duration,
                dynamic=True,
                buffer_time=20.0,
                cartesian_impedances=TRANSLATIONAL_STIFFNESS + ROTATIONAL_STIFFNESS,
                block=False,
            )
            init_time = fa.get_time()
        self._set_commanded_pose_array(current_pose)
        with self._lock:
            self._executed_commanded_pose = self._commanded_pose_array.copy()
            self._executed_action = np.zeros(7, dtype=np.float64)
        state, joints = self._snapshot_cached_robot_state()
        self._record_control_trace(
            timestamp=0.0,
            state=state,
            joint_state=joints,
            action=np.zeros(7, dtype=np.float64),
            commanded_pose=self._commanded_pose_array,
        )
        logger.info("dynamic skill 就绪，开始接收 action")
        return current_pose.copy(), init_time

    def _next_action(self):
        try:
            return self._action_queue.get(timeout=CONTROL_DT * 2)
        except queue.Empty:
            return None

    def _resolve_goal_pose(
        self,
        commanded_pose: "RigidTransform",
        actual_pos: np.ndarray,
        actual_rotvec: np.ndarray,
        action,
    ):
        if action is None:
            return commanded_pose, commanded_pose, None, None

        should_transform = True
        if isinstance(action, tuple):
            action, should_transform = action

        raw_action = np.asarray(action, dtype=np.float64).copy()
        transformed_action = self.transform_action(raw_action, scale_motion=should_transform)
        transformed_action, raw_action = self._limit_action_by_pose_error(
            actual_pos,
            actual_rotvec,
            commanded_pose,
            transformed_action,
            raw_action,
        )
        goal_pose = self._compute_target_pose(commanded_pose, transformed_action[:6])
        return commanded_pose, goal_pose, transformed_action, raw_action

    def _maybe_update_gripper(
        self,
        fa: "FrankaArm",
        action: np.ndarray,
        last_gripper: float | None,
    ) -> float:
        gripper_target = action[6]
        if last_gripper is None or gripper_target != last_gripper:
            with self._fa_api_lock:
                if gripper_target <= 0.0:
                    fa.close_gripper(grasp=True, block=False)
                else:
                    fa.open_gripper(block=False)
            return gripper_target
        return last_gripper

    def _set_executed_control_snapshot(self, raw_action: np.ndarray):
        with self._lock:
            self._executed_action = np.asarray(raw_action, dtype=np.float64).copy()
            self._executed_commanded_pose = self._commanded_pose_array.copy()

    def _publish_interp_pose(
        self,
        fa: "FrankaArm",
        start_pose: "RigidTransform",
        goal_pose: "RigidTransform",
        init_time: float,
        msg_id: int,
    ) -> int:
        cycle_start = time.perf_counter()
        ts = np.arange(INTERP_DT, CONTROL_DT + INTERP_DT * 0.5, INTERP_DT)
        for idx, t in enumerate(ts, start=1):
            if self._skill_stop.is_set():
                break
            w = min_jerk_weight(t, CONTROL_DT)
            interp_pose = goal_pose.interpolate_with(start_pose, 1.0 - w)
            msg_id += 1
            with self._fa_api_lock:
                timestamp = fa.get_time() - init_time
            traj_proto = PosePositionSensorMessage(
                id=msg_id,
                timestamp=timestamp,
                position=interp_pose.translation,
                quaternion=interp_pose.quaternion,
            )
            impedance_proto = CartesianImpedanceSensorMessage(
                id=msg_id,
                timestamp=timestamp,
                translational_stiffnesses=TRANSLATIONAL_STIFFNESS,
                rotational_stiffnesses=ROTATIONAL_STIFFNESS,
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_proto, SensorDataMessageType.POSE_POSITION
                ),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    impedance_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE
                ),
            )
            with self._fa_api_lock:
                fa.publish_sensor_data(ros_msg)
            deadline = cycle_start + idx * INTERP_DT
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        cycle_deadline = cycle_start + CONTROL_DT
        remaining = cycle_deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        return msg_id

    def _publish_termination(self, fa: "FrankaArm", init_time: float):
        with self._fa_api_lock:
            timestamp = fa.get_time() - init_time
        term_msg = ShouldTerminateSensorMessage(
            timestamp=timestamp, should_terminate=True
        )
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_msg, SensorDataMessageType.SHOULD_TERMINATE
            )
        )
        with self._fa_api_lock:
            fa.publish_sensor_data(ros_msg)
        logger.info("dynamic skill 已终止")

    def _wait_for_skill_idle(self, timeout: float = 5.0):
        """等待底层 FrankaArm skill 完全退出，避免后续命令被判定为 active skill。"""
        if self._fa is None:
            return

        start = time.perf_counter()
        try:
            with self._fa_api_lock:
                self._fa.wait_for_skill()
        except Exception:
            logger.exception("等待 dynamic skill 退出失败")
            raise
        finally:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                logger.warning("等待 dynamic skill 退出耗时 %.2fs", elapsed)

    # ------------------------------------------------------------------
    # obs 采集
    # ------------------------------------------------------------------

    def get_camera_frames(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cameras.get_frames()

    def get_observation(self, prompt: str) -> dict:
        img1, img2 = self.get_camera_frames()

        img1_resized = self._resize_observation_image(img1)
        img2_resized = self._resize_observation_image(img2)

        if self._fa is not None:
            pos, rot, joints, half = self._get_robot_state_for_observation()
            state = self._build_observation_state(pos, rot, half)
        else:
            state = np.zeros(8, dtype=np.float64)
            joints = np.zeros(7, dtype=np.float64)

        return {
            "observation/image": img1_resized,
            "observation/wrist_image": img2_resized,
            "observation/state": state.astype(np.float64),
            "observation/joints": joints.astype(np.float64),
            "prompt": prompt,
        }, img1, img2  # 同时返回原始帧供前端推流和录屏

    # ------------------------------------------------------------------
    # action 执行
    # ------------------------------------------------------------------

    def start_skill_thread(self):
        """启动 dynamic skill 执行线程（如已在运行则忽略）。"""
        with self._lock:
            if self._skill_thread is not None and self._skill_thread.is_alive():
                return
            self._skill_stop.clear()
            # 重置历史旋转矢量，避免从旧的缓存开始导致奇异性处理错误
            self._prev_rotvec = None
            self._prev_commanded_rotvec = None
            self._skill_thread = threading.Thread(
                target=self._skill_loop, daemon=True
            )
            self._skill_thread.start()

    def _stop_skill_thread(self):
        """停止 dynamic skill 执行线程，等待其退出。"""
        self._skill_stop.set()
        # 放一个哨兵让线程从阻塞的 get 中醒来
        self._action_queue.put(None)
        thread = self._skill_thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("dynamic skill 线程在 3.0s 内未退出，保留线程引用并阻止重复启动")
            else:
                with self._lock:
                    if self._skill_thread is thread:
                        self._skill_thread = None

    def _skill_loop(self):
        """
        完全仿照 run_dynamic_pose.py 的 for 循环模式：
        1. goto_pose 启动 dynamic skill
        2. 循环从队列取 action，计算 target pose，publish sensor message
        3. 收到停止信号后发送 ShouldTerminate
        """
        if self._fa is None:
            return

        fa = self._fa
        thread = threading.current_thread()
        init_time: float | None = None

        try:
            commanded_pose, init_time = self._start_dynamic_skill(fa)
            msg_id = 0
            last_gripper: float | None = None

            while not self._skill_stop.is_set():
                self._refresh_robot_cache()
                state, joints = self._snapshot_cached_robot_state()

                action = self._next_action()
                if action is None and self._skill_stop.is_set():
                    break

                start_pose, goal_pose, transformed_action, raw_action = self._resolve_goal_pose(
                    commanded_pose,
                    state[:3],
                    state[3:6],
                    action,
                )
                commanded_pose = goal_pose
                self._set_commanded_pose_array(commanded_pose)

                if transformed_action is not None:
                    self._set_executed_control_snapshot(raw_action)
                    last_gripper = self._maybe_update_gripper(fa, transformed_action, last_gripper)
                    trace_action = raw_action
                else:
                    trace_action = np.zeros(7, dtype=np.float64)

                with self._fa_api_lock:
                    trace_timestamp = fa.get_time() - init_time
                self._record_control_trace(
                    timestamp=trace_timestamp,
                    state=state,
                    joint_state=joints,
                    action=trace_action,
                    commanded_pose=self._commanded_pose_array,
                )

                msg_id = self._publish_interp_pose(
                    fa,
                    start_pose,
                    goal_pose,
                    init_time,
                    msg_id,
                )
        except Exception:
            logger.exception("dynamic skill 线程异常退出")
            raise
        finally:
            if init_time is not None:
                try:
                    self._publish_termination(fa, init_time)
                except Exception:
                    logger.exception("发送 dynamic skill 终止消息失败")
            with self._lock:
                if self._skill_thread is thread:
                    self._skill_thread = None

    def enqueue_action(
        self,
        action: np.ndarray,
        *,
        transform: bool = True,
        latest_only: bool = False,
    ):
        """外部调用：将一个 action 放入队列，由 _skill_loop 消费。

        latest_only=True 时会丢弃队列中尚未执行的旧动作，仅保留最新动作，
        适用于键盘/手柄遥操作，避免控制延迟因 FIFO 积压而固定放大。
        """
        action_arr = np.asarray(action, dtype=np.float64)
        if self._fa is None:
            transformed = self.transform_action(action_arr, scale_motion=transform)
            with self._lock:
                self._commanded_pose_array = self._apply_delta_to_pose_array(
                    self._commanded_pose_array,
                    transformed[:6],
                )
                self._executed_action = action_arr.copy()
                self._executed_commanded_pose = self._commanded_pose_array.copy()
                next_timestamp = self._control_trace_sample.timestamp + CONTROL_DT
            half_width = transformed[6] / 2.0
            state = np.concatenate([
                self._commanded_pose_array.copy(),
                [half_width],
                [-half_width],
            ]).astype(np.float64)
            self._record_control_trace(
                timestamp=next_timestamp,
                state=state,
                joint_state=np.zeros(7, dtype=np.float64),
                action=action_arr,
                commanded_pose=self._commanded_pose_array,
            )
            return
        if latest_only:
            while True:
                try:
                    self._action_queue.get_nowait()
                except queue.Empty:
                    break
        self._action_queue.put((action_arr, transform))

    def get_pending_action_count(self) -> int:
        return self._action_queue.qsize()

    def hold_pose(self):
        """外部调用：放一个 None 进队列，让 _skill_loop 保持当前位姿一拍。"""
        # 队列非空时不重复放，避免积压
        if self._action_queue.empty():
            self._action_queue.put(None)

    @property
    def commanded_pose_array(self) -> np.ndarray:
        return self._commanded_pose_array

    def get_latest_control_trace(self) -> dict[str, np.ndarray | int | float]:
        """返回最近一拍已对齐的控制 trace。"""
        with self._lock:
            sample = self._control_trace_sample
            return {
                "seq_id": int(sample.seq_id),
                "timestamp": float(sample.timestamp),
                "state": sample.state.copy(),
                "joint_state": sample.joint_state.copy(),
                "action": sample.action.copy(),
                "commanded_pose": sample.commanded_pose.copy(),
            }

    def get_control_trace_since(self, last_seq_id: int) -> list[dict[str, np.ndarray | int | float]]:
        """返回所有 seq_id > last_seq_id 的控制 trace。"""
        with self._lock:
            samples = [
                sample for sample in self._control_trace_history
                if sample.seq_id > last_seq_id
            ]
        return [
            {
                "seq_id": int(sample.seq_id),
                "timestamp": float(sample.timestamp),
                "state": sample.state.copy(),
                "joint_state": sample.joint_state.copy(),
                "action": sample.action.copy(),
                "commanded_pose": sample.commanded_pose.copy(),
            }
            for sample in samples
        ]

    def get_executed_control_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """返回最近一次被执行线程真正采用的 (action, commanded_pose)。"""
        trace = self.get_latest_control_trace()
        return trace["action"], trace["commanded_pose"]

    @property
    def ee_force_torque(self) -> np.ndarray:
        """获取当前末端力矩信息 [force_x, force_y, force_z, torque_x, torque_y, torque_z]。"""
        with self._lock:
            if self._cached_ee_force_torque is None:
                return np.zeros(6)
            return self._cached_ee_force_torque.copy()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _compute_target_pose(self, current: "RigidTransform", delta: np.ndarray) -> "RigidTransform":
        """将 delta [dx,dy,dz,dax,day,daz] 叠加到当前位姿。"""
        delta_translation = delta[:3].copy()
        delta_axisangle = delta[3:6].copy()

        target = current.copy()
        target.translation = current.translation + delta_translation

        angle = np.linalg.norm(delta_axisangle)
        if angle > 1e-6:
            delta_rot = Rotation.from_rotvec(delta_axisangle).as_matrix()
            target.rotation = delta_rot @ current.rotation  # delta 在基座坐标系表达，先应用 delta

        return target

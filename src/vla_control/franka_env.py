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

POS_CLIP = 1.0          # 位置 action 输入截断范围 [-POS_CLIP, POS_CLIP]
ROT_CLIP = 0.1          # 旋转 action 输入截断范围 [-ROT_CLIP, ROT_CLIP]
POS_MAX_VEL = 0.1 / 3      # 末端位置最大速度 (m/s)
ROT_MAX_VEL = (math.pi / 4) / 3   # 末端旋转最大速度 (rad/s)
GRIPPER_SPEED = 0.08    # 夹爪运动速度 (m/s)，1s 走完全程 0.08m
POS_SCALE = POS_MAX_VEL * CONTROL_DT / POS_CLIP
ROT_SCALE = ROT_MAX_VEL * CONTROL_DT / ROT_CLIP

# 阻抗控制刚度参数，与 key_control 保持一致
TRANSLATIONAL_STIFFNESS = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES if HAS_FRANKA else [600.0, 600.0, 600.0]
ROTATIONAL_STIFFNESS = FC.DEFAULT_ROTATIONAL_STIFFNESSES if HAS_FRANKA else [50.0, 50.0, 50.0]


def transform_action(action: np.ndarray) -> np.ndarray:
    """将原始 action [7] 裁剪并缩放为机器人可执行的指令。"""
    ta = action.copy()
    ta[:3] = np.clip(ta[:3], -POS_CLIP, POS_CLIP) * POS_SCALE
    ta[3:6] = np.clip(ta[3:6], -ROT_CLIP, ROT_CLIP) * ROT_SCALE
    ta[6] = 0.0 if ta[6] >= 0 else 0.08
    return ta


@dataclass
class ActionTransformConfig:
    pos_clip: float = POS_CLIP
    rot_clip: float = ROT_CLIP
    pos_max_vel: float = POS_MAX_VEL
    rot_max_vel: float = ROT_MAX_VEL


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
    ta[6] = 0.0 if ta[6] >= 0 else 0.08
    return ta


class D435Camera:
    """单路 D435，pyrealsense2 直连 USB。serial_number=None 时自动选第一个设备。"""

    def __init__(self, serial_number: str | None = None, width=640, height=480, fps=30):
        self._serial = serial_number
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None

    def start(self):
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 未安装，无法启动相机")
        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        self._pipeline = rs.pipeline()
        self._pipeline.start(config)

    def get_frame(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data())

    def stop(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class DualD435:
    """双路 D435 封装，某路 serial 为 None 或启动失败时返回全黑帧。"""

    def __init__(self, cam1_serial: str | None = None, cam2_serial: str | None = None):
        self._cam1 = D435Camera(cam1_serial)
        self._cam2 = D435Camera(cam2_serial)
        self._black = np.zeros((480, 640, 3), dtype=np.uint8)

    def start(self):
        for cam in (self._cam1, self._cam2):
            try:
                cam.start()
            except Exception as e:
                logger.warning(f"相机启动失败，将使用全黑帧：{e}")

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
    """规范化旋转矢量到 [-π, π] 范围，并处理奇异性保持连续性。
    
    旋转矢量表示的奇异性问题：
    - 当旋转角度接近 ±π 时，旋转方向可能发生突变
    - 当旋转角度为 0 或 2π 时，表示不唯一
    
    处理方法：
    1. 将角度规范化到 [-π, π] 范围
    2. 检测并修复角度跳变（unwrap），保持旋转连续性
    """
    rotvec = np.asarray(rotvec, dtype=np.float64).copy()
    
    # 计算旋转角度（标量）
    angle = np.linalg.norm(rotvec)
    
    if angle < 1e-6:
        # 角度接近0，返回零向量
        return rotvec
    
    # 轴向
    axis = rotvec / angle
    
    # 规范化角度到 [-π, π]
    angle = np.angle(np.exp(1j * angle))  # 等价于 angle % (2*π) 然后移到 [-π, π]
    
    # 检查是否需要 unwrap：若与上一次角度差异超过 π，则调整
    # 这里假设单帧内不会有超过 π 的旋转，因此若出现大跳变则取相反方向
    if angle > np.pi - 0.1:  # 接近 +π 时，检测是否应取 -π 方向
        # 检查是否从负值跳变过来，若是从正方向接近 π，可能取反更连续
        pass  # 具体情况由调用者传入历史状态判断
    
    return axis * angle


def normalize_rotvec_with_history(rotvec: np.ndarray, prev_rotvec: np.ndarray | None) -> np.ndarray:
    """带历史信息的旋转矢量规范化，处理奇异性保持连续性。
    
    Args:
        rotvec: 当前旋转矢量
        prev_rotvec: 上一帧的旋转矢量，若为 None 则使用单帧规范化
    """
    rotvec = np.asarray(rotvec, dtype=np.float64).copy()
    
    if prev_rotvec is None:
        return normalize_rotvec(rotvec)
    
    # 计算当前角度和轴向
    current_angle = np.linalg.norm(rotvec)
    if current_angle < 1e-6:
        return rotvec
    
    current_axis = rotvec / current_angle
    current_angle = np.angle(np.exp(1j * current_angle))
    
    # 计算上一帧角度
    prev_angle = np.linalg.norm(prev_rotvec)
    if prev_angle < 1e-6:
        prev_angle = 0.0
        prev_axis = np.array([0., 0., 1.])
    else:
        prev_axis = prev_rotvec / prev_angle
        prev_angle = np.angle(np.exp(1j * prev_angle))
    
    # 检测轴向是否反转（检测 θ -> -θ + 2π 的跳变）
    angle_diff = current_angle - prev_angle
    
    # 若角度差超过 π，说明发生了奇异性跳变
    if abs(angle_diff) > np.pi:
        # 调整当前角度以保持连续
        if angle_diff > 0:
            current_angle -= 2 * np.pi
        else:
            current_angle += 2 * np.pi
    
    # 检测轴向是否反转（当角度很小时可能需要）
    if abs(current_angle) < 0.1:  # 角度很小时，检查是否应取相反轴
        axis_dot = np.dot(current_axis, prev_axis)
        if axis_dot < -0.9:  # 轴向几乎相反
            # 保持上一帧的轴向反转
            return -prev_rotvec
    
    return current_axis * current_angle


class FrankaEnv:
    """封装机械臂状态读取与 dynamic pose 控制。"""

    def __init__(
        self,
        cam1_serial: str | None = None,
        cam2_serial: str | None = None,
        dynamic_duration: float = 60.0,
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
        self._commanded_pose_array = np.zeros(6)
        
        # 实际位姿缓存：_skill_loop 定期更新，get_observation 读取，避免重复调用 ROS API
        self._cached_pose: np.ndarray | None = None  # [pos(3), rot_axisangle(3)]
        self._cached_joints: np.ndarray | None = None  # (7,)
        self._cached_gripper_width: float | None = None  # 单指开口(m)
        
        # 末端力矩缓存：[force(3), torque(3)]，单位 [N, Nm]
        self._cached_ee_force_torque: np.ndarray | None = None  # (6,)
        
        # 旋转矢量历史缓存（用于处理奇异性）
        self._prev_rotvec: np.ndarray | None = None  # 上一帧的轴角

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self):
        self._cameras.start()

    def stop(self):
        self.stop_control()

    def start_control(self):
        self.start()
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
        self._fa.reset_joints()

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

    def transform_action(self, action: np.ndarray, *, scale_motion: bool = True) -> np.ndarray:
        return _apply_action_transform(action, self._action_transform, scale_motion=scale_motion)

    def _resize_observation_image(self, image: np.ndarray) -> np.ndarray:
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, RESIZE, RESIZE))

    def _pose_to_rotvec(self, pose: "RigidTransform") -> np.ndarray:
        return Rotation.from_matrix(pose.rotation).as_rotvec()

    def _set_commanded_pose_array(self, pose: "RigidTransform"):
        rot_vec = normalize_rotvec(self._pose_to_rotvec(pose))
        self._commanded_pose_array = np.concatenate([pose.translation, rot_vec])

    def _read_robot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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

    def _refresh_robot_cache(self):
        with self._lock:
            actual_pose = self._fa.get_pose()
            actual_joints = self._fa.get_joints()
            actual_gripper_half = self._fa.get_gripper_width() / 2.0
            ee_force_torque = self._fa.get_ee_force_torque()
            raw_rot_vec = self._pose_to_rotvec(actual_pose)
            rot_vec = normalize_rotvec_with_history(raw_rot_vec, self._prev_rotvec)

            self._cached_pose = np.concatenate([actual_pose.translation, rot_vec])
            self._cached_joints = actual_joints
            self._cached_gripper_width = actual_gripper_half
            self._cached_ee_force_torque = ee_force_torque
            self._prev_rotvec = raw_rot_vec.copy()

    def _start_dynamic_skill(self, fa: "FrankaArm") -> tuple["RigidTransform", float]:
        logger.info("等待机械臂状态同步...")
        time.sleep(1.0)
        current_pose = fa.get_pose()

        logger.info("启动 dynamic skill...")
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
        logger.info("dynamic skill 就绪，开始接收 action")
        return current_pose.copy(), init_time

    def _next_action(self):
        try:
            return self._action_queue.get(timeout=CONTROL_DT * 2)
        except queue.Empty:
            return None

    def _resolve_goal_pose(self, commanded_pose: "RigidTransform", action):
        if action is None:
            return commanded_pose, commanded_pose, None

        should_transform = True
        if isinstance(action, tuple):
            action, should_transform = action

        transformed_action = self.transform_action(action, scale_motion=should_transform)
        goal_pose = self._compute_target_pose(commanded_pose, transformed_action[:6])
        return commanded_pose, goal_pose, transformed_action

    def _maybe_update_gripper(
        self,
        fa: "FrankaArm",
        action: np.ndarray,
        last_gripper: float | None,
    ) -> float:
        gripper_target = action[6]
        if last_gripper is None or gripper_target != last_gripper:
            fa.goto_gripper(gripper_target, block=False, speed=GRIPPER_SPEED)
            return gripper_target
        return last_gripper

    def _publish_interp_pose(
        self,
        fa: "FrankaArm",
        start_pose: "RigidTransform",
        goal_pose: "RigidTransform",
        init_time: float,
        msg_id: int,
    ) -> int:
        ts = np.arange(INTERP_DT, CONTROL_DT + INTERP_DT * 0.5, INTERP_DT)
        for t in ts:
            if self._skill_stop.is_set():
                break
            w = min_jerk_weight(t, CONTROL_DT)
            interp_pose = goal_pose.interpolate_with(start_pose, 1.0 - w)
            msg_id += 1
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
            fa.publish_sensor_data(ros_msg)
            time.sleep(INTERP_DT)
        return msg_id

    def _publish_termination(self, fa: "FrankaArm", init_time: float):
        timestamp = fa.get_time() - init_time
        term_msg = ShouldTerminateSensorMessage(
            timestamp=timestamp, should_terminate=True
        )
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_msg, SensorDataMessageType.SHOULD_TERMINATE
            )
        )
        fa.publish_sensor_data(ros_msg)
        logger.info("dynamic skill 已终止")

    # ------------------------------------------------------------------
    # obs 采集
    # ------------------------------------------------------------------

    def get_observation(self, prompt: str) -> dict:
        img1, img2 = self._cameras.get_frames()

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
            self._skill_thread = threading.Thread(
                target=self._skill_loop, daemon=True
            )
            self._skill_thread.start()

    def _stop_skill_thread(self):
        """停止 dynamic skill 执行线程，等待其退出。"""
        self._skill_stop.set()
        # 放一个哨兵让线程从阻塞的 get 中醒来
        self._action_queue.put(None)
        if self._skill_thread is not None:
            self._skill_thread.join(timeout=3.0)
            if self._skill_thread.is_alive():
                logger.warning("dynamic skill 线程在 3.0s 内未退出，保留线程引用并阻止重复启动")
            else:
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

                action = self._next_action()
                if action is None and self._skill_stop.is_set():
                    break

                start_pose, goal_pose, transformed_action = self._resolve_goal_pose(commanded_pose, action)
                commanded_pose = goal_pose
                self._set_commanded_pose_array(commanded_pose)

                if transformed_action is not None:
                    last_gripper = self._maybe_update_gripper(fa, transformed_action, last_gripper)

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

    def enqueue_action(self, action: np.ndarray, *, transform: bool = True):
        """外部调用：将一个 action 放入队列，由 _skill_loop 消费。"""
        self._action_queue.put((np.asarray(action, dtype=np.float64), transform))

    def hold_pose(self):
        """外部调用：放一个 None 进队列，让 _skill_loop 保持当前位姿一拍。"""
        # 队列非空时不重复放，避免积压
        if self._action_queue.empty():
            self._action_queue.put(None)

    @property
    def commanded_pose_array(self) -> np.ndarray:
        return self._commanded_pose_array

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

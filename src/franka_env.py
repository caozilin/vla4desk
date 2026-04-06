"""FrankaEnvironment：obs 采集 + action 执行，基于 libero 的字段格式。

obs 格式（发给 openpi 服务端）：
    observation/image       : (224, 224, 3) uint8  外部相机
    observation/wrist_image : (224, 224, 3) uint8  腕部相机
    observation/state       : (8,) float64  [eef_pos(3), eef_axisangle(3), finger1_qpos(1), finger2_qpos(1)]
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

POS_CLIP = 0.05
ROT_CLIP = 0.5
POS_SCALE = 1.0
ROT_SCALE = 1.0


def transform_action(action: np.ndarray) -> np.ndarray:
    """将原始 action [7] 裁剪并缩放为机器人可执行的指令。"""
    ta = action.copy()
    ta[:3] = np.clip(ta[:3], -POS_CLIP, POS_CLIP) * POS_SCALE
    ta[3:6] = np.clip(ta[3:6], -ROT_CLIP, ROT_CLIP) * ROT_SCALE
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


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """quaternion [x,y,z,w] → axis-angle (3,)，与 libero 保持一致。"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


CONTROL_DT = 0.1   # 每个 action 的执行周期（s），决定控制频率
INTERP_DT = 0.02   # 50Hz：每次 sensor message 间隔（与示例一致）
INTERP_STEPS = int(CONTROL_DT / INTERP_DT)  # = 5

# Action 缩放超参数
POS_CLIP = 1.0          # 位置 action 输入截断范围 [-POS_CLIP, POS_CLIP]
ROT_CLIP = 0.1          # 旋转 action 输入截断范围 [-ROT_CLIP, ROT_CLIP]
POS_MAX_VEL = 0.1       # 末端位置最大速度 (m/s)
ROT_MAX_VEL = math.pi   # 末端旋转最大速度 (rad/s)
GRIPPER_SPEED = 0.08    # 夹爪运动速度 (m/s)，1s 走完全程 0.08m
# 由超参数推导的每步缩放系数（当 action=clip 上限时，delta = MAX_VEL × CONTROL_DT）
POS_SCALE = POS_MAX_VEL * CONTROL_DT / POS_CLIP
ROT_SCALE = ROT_MAX_VEL * CONTROL_DT / ROT_CLIP


class FrankaEnv:
    """封装机械臂状态读取与 dynamic pose 控制。"""

    def __init__(
        self,
        cam1_serial: str | None = None,
        cam2_serial: str | None = None,
        dynamic_duration: float = 60.0,
        no_robot: bool = False,
    ):
        if no_robot or not HAS_FRANKA:
            if not no_robot:
                logger.warning("frankapy 未安装")
            self._fa = None
        else:
            self._fa = FrankaArm()

        self._cameras = DualD435(cam1_serial, cam2_serial)
        self._dynamic_duration = dynamic_duration

        # 动作队列：coordinator 往里放 action，_skill_thread 消费
        self._action_queue: queue.Queue = queue.Queue()
        self._skill_thread: threading.Thread | None = None
        self._skill_stop = threading.Event()
        self._lock = threading.Lock()
        self._commanded_pose_array = np.zeros(6)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self):
        self._cameras.start()

    def stop(self):
        self._stop_skill_thread()
        self._cameras.stop()

    def reset_to_home(self):
        """结束 dynamic skill 线程，回到 home 位姿。"""
        if self._fa is None:
            logger.warning("机械臂不可用，跳过 reset_to_home")
            return
        self._stop_skill_thread()
        self._fa.reset_joints()

    # ------------------------------------------------------------------
    # obs 采集
    # ------------------------------------------------------------------

    def get_observation(self, prompt: str) -> dict:
        img1, img2 = self._cameras.get_frames()

        img1_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img1, RESIZE, RESIZE)
        )
        img2_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img2, RESIZE, RESIZE)
        )

        if self._fa is not None:
            pose = self._fa.get_pose()  # RigidTransform
            joints = self._fa.get_joints()  # (7,) 关节角度
            half = self._fa.get_gripper_width() / 2.0  # 单指开口(m)
            state = np.concatenate([
                pose.translation,                    # (3,) eef pos
                quat2axisangle(pose.quaternion),     # (3,) eef rot
                [+half],                             # (1,) finger1 qpos，对应 libero robot0_gripper_qpos[0]
                [-half],                             # (1,) finger2 qpos，对应 libero robot0_gripper_qpos[1]
            ])
        else:
            state = np.zeros(8, dtype=np.float64)

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
        current_pose = fa.get_pose()

        logger.info("启动 dynamic skill...")
        fa.goto_pose(
            current_pose,
            duration=self._dynamic_duration,
            dynamic=True,
            buffer_time=10,
        )
        init_time = fa.get_time()
        msg_id = 0
        # 维护指令位姿（commanded pose），在其基础上叠加 delta，而非实际位姿
        commanded_pose = current_pose.copy()
        logger.info("dynamic skill 就绪，开始接收 action")

        last_gripper: float | None = None

        while not self._skill_stop.is_set():
            try:
                action = self._action_queue.get(timeout=CONTROL_DT * 2)
            except queue.Empty:
                action = None

            if action is None:
                if self._skill_stop.is_set():
                    break
                # 超时/保持：以 min_jerk 插值维持在当前指令位姿
                start_pose = commanded_pose
                goal_pose = commanded_pose
            else:
                # 使用统一的变换逻辑
                action = transform_action(action)

                start_pose = commanded_pose
                commanded_pose = self._compute_target_pose(commanded_pose, action[:6])
                goal_pose = commanded_pose

                # 更新指令位姿缓存供外部（Coordinator）读取用于前端显示
                rot_vec = Rotation.from_matrix(commanded_pose.rotation).as_rotvec()
                self._commanded_pose_array = np.concatenate([commanded_pose.translation, rot_vec])

                # 夹爪二值化逻辑已在 transform_action 中处理
                gripper_target = action[6]
                if last_gripper is None or gripper_target != last_gripper:
                    fa.goto_gripper(gripper_target, block=False, speed=GRIPPER_SPEED)
                    last_gripper = gripper_target

            # 在 CONTROL_DT 内以 INTERP_DT 为间隔用 min_jerk 插值发送
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
                    translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES,
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

        # 发送终止消息
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

    def enqueue_action(self, action: np.ndarray):
        """外部调用：将一个 action 放入队列，由 _skill_loop 消费。"""
        self._action_queue.put(action)

    def hold_pose(self):
        """外部调用：放一个 None 进队列，让 _skill_loop 保持当前位姿一拍。"""
        # 队列非空时不重复放，避免积压
        if self._action_queue.empty():
            self._action_queue.put(None)

    @property
    def commanded_pose_array(self) -> np.ndarray:
        return self._commanded_pose_array

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
            target.rotation = delta_rot @ current.rotation

        return target

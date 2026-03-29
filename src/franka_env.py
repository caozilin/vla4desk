"""FrankaEnvironment：obs 采集 + action 执行，基于 libero 的字段格式。

obs 格式（发给 openpi 服务端）：
    observation/image       : (224, 224, 3) uint8  外部相机
    observation/wrist_image : (224, 224, 3) uint8  腕部相机
    observation/state       : (8,) float64  [eef_pos(3), eef_axisangle(3), gripper(1), 0]
    prompt                  : str

action 格式（从 openpi 服务端接收）：
    actions : (5, 7) float64  [delta_pos(3), delta_axisangle(3), gripper(1)]
"""
import math
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "client"))

import image_tools
import numpy as np

try:
    from autolab_core import RigidTransform
    from frankapy import FrankaArm, SensorDataMessageType
    from frankapy import FrankaConstants as FC
    from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
    from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
    HAS_FRANKA = True
except ImportError:
    HAS_FRANKA = False

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


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
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if self._serial:
            cfg.enable_device(self._serial)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        self._pipeline.start(cfg)
        for _ in range(5):  # 预热
            self._pipeline.wait_for_frames()

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

    def get_frame(self) -> np.ndarray:
        """返回最新一帧 RGB 图像 (H, W, 3) uint8。"""
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("D435 未返回彩色帧")
        return np.asanyarray(color.get_data())

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


class FrankaEnv:
    """封装机械臂状态读取与 dynamic pose 控制。"""

    def __init__(
        self,
        cam1_serial: str | None = None,
        cam2_serial: str | None = None,
        dynamic_duration: float = 60.0,  # dynamic skill 的持续时长（秒），可按需调大
    ):
        if not HAS_FRANKA:
            raise RuntimeError("frankapy 未安装")

        self._fa = FrankaArm()
        self._cameras = DualD435(cam1_serial, cam2_serial)
        self._dynamic_duration = dynamic_duration

        self._in_dynamic = False
        self._init_time: float = 0.0
        self._msg_id: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self):
        self._cameras.start()

    def stop(self):
        self._end_dynamic_skill()
        self._cameras.stop()

    def reset_to_home(self):
        """结束 dynamic skill，回到 home 位姿。"""
        self._end_dynamic_skill()
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

        pose = self._fa.get_pose()  # RigidTransform
        gripper = self._fa.get_gripper_width()  # float (m)

        state = np.concatenate([
            pose.translation,                    # (3,) eef pos
            quat2axisangle(pose.quaternion),     # (3,) eef rotation
            [gripper],                           # (1,) gripper width
            [0.0],                               # (1,) 占位，与 libero state (8,) 对齐
        ])

        return {
            "observation/image": img1_resized,
            "observation/wrist_image": img2_resized,
            "observation/state": state.astype(np.float64),
            "prompt": prompt,
        }, img1, img2  # 同时返回原始帧供前端推流和录屏

    # ------------------------------------------------------------------
    # action 执行
    # ------------------------------------------------------------------

    def start_dynamic_skill(self):
        """启动 dynamic pose skill（非阻塞），在控制循环开始前调用一次。"""
        if self._in_dynamic:
            return
        current_pose = self._fa.get_pose()
        self._fa.goto_pose(
            current_pose,
            duration=self._dynamic_duration,
            dynamic=True,
            buffer_time=10,
        )
        self._init_time = self._fa.get_time()
        self._msg_id = 0
        self._in_dynamic = True

    def apply_action(self, action: np.ndarray):
        """执行单步 action (7,): [delta_pos(3), delta_axisangle(3), gripper(1)]。"""
        if not self._in_dynamic:
            self.start_dynamic_skill()

        current_pose = self._fa.get_pose()
        target_pose = self._compute_target_pose(current_pose, action[:6])

        self._msg_id += 1
        timestamp = self._fa.get_time() - self._init_time

        proto_msg = PosePositionSensorMessage(
            id=self._msg_id,
            timestamp=timestamp,
            position=target_pose.translation,
            quaternion=target_pose.quaternion,
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                proto_msg, SensorDataMessageType.POSE_POSITION
            )
        )
        self._fa.publish_sensor_data(ros_msg)

        # 夹爪控制（非阻塞）
        target_gripper = float(np.clip(action[6], 0.0, 0.08))
        self._fa.goto_gripper(target_gripper, block=False)

    def hold_current_pose(self):
        """发送当前位姿作为目标，实现静止保持。"""
        if not self._in_dynamic:
            self.start_dynamic_skill()
            return

        current_pose = self._fa.get_pose()
        self._msg_id += 1
        timestamp = self._fa.get_time() - self._init_time

        proto_msg = PosePositionSensorMessage(
            id=self._msg_id,
            timestamp=timestamp,
            position=current_pose.translation,
            quaternion=current_pose.quaternion,
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                proto_msg, SensorDataMessageType.POSE_POSITION
            )
        )
        self._fa.publish_sensor_data(ros_msg)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _compute_target_pose(self, current: "RigidTransform", delta: np.ndarray) -> "RigidTransform":
        """将 delta [dx,dy,dz, dax,day,daz] 叠加到当前位姿。"""
        delta_translation = delta[:3]
        delta_axisangle = delta[3:6]
        angle = np.linalg.norm(delta_axisangle)

        target = current.copy()
        target.translation = current.translation + delta_translation

        if angle > 1e-6:
            axis = delta_axisangle / angle
            delta_rot = RigidTransform.rotation_from_axes(axis, angle)
            target.rotation = delta_rot @ current.rotation

        return target

    def _end_dynamic_skill(self):
        if not self._in_dynamic:
            return
        timestamp = self._fa.get_time() - self._init_time
        term_msg = ShouldTerminateSensorMessage(
            timestamp=timestamp, should_terminate=True
        )
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_msg, SensorDataMessageType.SHOULD_TERMINATE
            )
        )
        self._fa.publish_sensor_data(ros_msg)
        self._in_dynamic = False

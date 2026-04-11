#!/usr/bin/env python3
"""
轨迹复现脚本
=============
按照 data_collection 收集的某一段轨迹来一比一复现。

使用方式：
    python trajectory_replay.py --episode collected/simple_pick_place/epo_1
    python trajectory_replay.py --task simple_pick_place --epo 1

功能：
    - 读取 data.csv 中的 action 序列
    - 使用 franka_env.py 的 FrankaEnv 和 _skill_loop 执行轨迹
    - 按照采集频率（默认 10Hz）逐步复现
    - 支持变速、暂停、中断
"""

import argparse
import csv
import json
import logging
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from franka_env import FrankaEnv, transform_action

logger = logging.getLogger(__name__)


def load_episode_data(episode_path: pathlib.Path) -> tuple[list[dict], dict]:
    """加载一个 episode 的 data.csv 文件。
    
    Returns:
        (actions, metadata) 
        actions: list of dict, 每个 dict 包含 state 和 action
        metadata: dict, 采集超参数
    """
    csv_path = episode_path / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    actions = []
    metadata = {}
    
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        
        # 第1行: metadata JSON
        meta_row = next(reader)
        metadata = json.loads(meta_row[0])
        logger.info(f"加载元数据: {metadata}")
        
        # 第2行: 列名
        header = next(reader)
        
        # 第3行起: 数据
        for row in reader:
            values = [float(v) for v in row]
            # state: 前8列, action: 后7列
            state = values[:8]
            action = values[8:15]
            actions.append({
                "state": state,
                "action": action
            })
    
    logger.info(f"加载 {len(actions)} 帧动作数据")
    return actions, metadata


def find_episode(task_name: str | None, epo_num: int, base_dir: pathlib.Path) -> pathlib.Path:
    """根据任务名和episode编号找到数据目录。"""
    if task_name:
        episode_path = base_dir / task_name / f"epo_{epo_num}"
    else:
        # 尝试从路径推断
        episode_path = base_dir
    
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode 目录不存在: {episode_path}")
    
    return episode_path


class TrajectoryReplayer:
    """轨迹复现器。
    
    Args:
        episode_path: Episode 数据目录的绝对路径
        cam1_serial: 外部相机序列号
        cam2_serial: 腕部相机序列号
        speed_factor: 速度倍率（1.0=原速，2.0=2倍速）
        no_robot: 是否不连接机械臂（测试模式）
    """
    
    def __init__(
        self,
        episode_path: pathlib.Path,
        cam1_serial: str | None = "346222072769",
        cam2_serial: str | None = "938422075745",
        speed_factor: float = 1.0,
        no_robot: bool = False,
        save_csv: bool = False,
        output_dir: str | None = None,
    ):
        self.episode_path = episode_path
        self.speed_factor = speed_factor
        self.save_csv = save_csv
        self.output_dir = pathlib.Path(output_dir) if output_dir else None
        
        # 加载数据
        self.actions, self.metadata = load_episode_data(episode_path)
        self.collect_hz = self.metadata.get("collect_hz", 20.0)
        
        # 强制使用 10Hz 控制频率（与 vla_control 一致）
        self.control_hz = 10.0
        self.control_dt = (1.0 / self.control_hz) / speed_factor
        
        # 初始化 FrankaEnv
        # 注意：需要传递 dynamic_duration 足够长以覆盖整个轨迹
        trajectory_duration = len(self.actions) * self.control_dt + 10.0  # 额外10秒缓冲
        self.env = FrankaEnv(
            cam1_serial=cam1_serial,
            cam2_serial=cam2_serial,
            dynamic_duration=trajectory_duration,
            no_robot=no_robot,
        )
        
        self.running = False
        self.paused = False
    
    def start(self):
        """启动复现。"""
        print("=" * 60, flush=True)
        print("轨迹复现器初始化", flush=True)
        print(f"Episode: {self.episode_path}", flush=True)
        print(f"总帧数: {len(self.actions)}", flush=True)
        print(f"采集频率: {self.collect_hz}Hz (原始)", flush=True)
        print(f"复现频率: {self.control_hz}Hz (强制10Hz)", flush=True)
        print(f"速度倍率: {self.speed_factor}x", flush=True)
        print(f"实际执行频率: {self.control_hz / self.speed_factor:.1f}Hz", flush=True)
        if self.save_csv:
            print(f"CSV 保存: 开启", flush=True)
            if self.output_dir:
                print(f"输出目录: {self.output_dir}", flush=True)
        print("=" * 60, flush=True)
        
        self.env.start()
        
        # 等待机械臂就绪
        if self.env._fa is not None:
            time.sleep(1.0)
            print("机械臂就绪", flush=True)
            
            # 先复位到 home 位姿
            print("开始复位到 home 位姿...", flush=True)
            self.env._fa.reset_joints()
            print("复位完成", flush=True)
            
            # 等待复位后稳定
            time.sleep(1.0)
        
        self.running = True
        self._replay_loop()
    
    def _replay_loop(self):
        """主复现循环 - 直接控制机械臂。"""
        print("开始复现轨迹...", flush=True)
        print("按 Ctrl+C 中断", flush=True)
        
        fa = self.env._fa
        
        if fa is not None:
            # 使用阻抗控制模式
            from frankapy import SensorDataMessageType
            from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
            from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
            from frankapy.utils import min_jerk_weight
            from scipy.spatial.transform import Rotation
            from autolab_core import RigidTransform
            
            # 获取初始位姿
            current_pose = fa.get_pose()
            
            # 启动 dynamic skill
            trajectory_duration = len(self.actions) * self.control_dt + 10.0
            print(f"启动 dynamic skill (duration={trajectory_duration:.1f}s)", flush=True)
            fa.goto_pose(
                current_pose,
                duration=trajectory_duration,
                dynamic=True,
                buffer_time=15.0,
                cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0],
                block=False
            )
            init_time = fa.get_time()
            
            # 维护 commanded_pose
            commanded_pose = current_pose.copy()
            msg_id = 0
            
            last_gripper_width = fa.get_gripper_width()
        
        # 调试信息：打印前10个 action 的详细信息
        print("=" * 80, flush=True)
        print("Action 序列调试信息（前10帧）：", flush=True)
        print(f"{'帧':>4} | {'dt(ms)':>6} | {'dx(mm)':>7} {'dy(mm)':>7} {'dz(mm)':>7} | {'dax(deg)':>8} {'day(deg)':>8} {'daz(deg)':>8} | {'gripper':>7}", flush=True)
        print("-" * 80, flush=True)
        
        try:
            for i, step_data in enumerate(self.actions):
                if not self.running:
                    break
                
                # 暂停检查
                while self.paused:
                    time.sleep(0.1)
                
                action = np.array(step_data["action"], dtype=np.float64)
                state = np.array(step_data["state"], dtype=np.float64)
                
                # 执行 action
                if fa is not None:
                    # 处理 action
                    # 位置和旋转：直接使用原始值
                    # 夹爪：需要二值化（参考 franka_env.py 的 transform_action）
                    raw_action = action.copy()
                    
                    # 夹爪二值化处理（与 transform_action 一致）
                    # action[6] >= 0 → 0.0 (闭合)
                    # action[6] < 0  → 0.08 (打开)
                    gripper_target = 0.0 if raw_action[6] >= 0 else 0.08
                    
                    # 2. 计算目标位姿
                    start_pose = commanded_pose.copy()
                    
                    # 位置增量
                    commanded_pose.translation += raw_action[:3]
                    
                    # 旋转增量（基座坐标系，左乘）
                    delta_rotvec = raw_action[3:6]
                    angle = np.linalg.norm(delta_rotvec)
                    if angle > 1e-6:
                        delta_rot = Rotation.from_rotvec(delta_rotvec).as_matrix()
                        commanded_pose.rotation = delta_rot @ commanded_pose.rotation
                    
                    goal_pose = commanded_pose
                    
                    # 3. 夹爪控制（二值化后的值）
                    if abs(gripper_target - last_gripper_width) > 1e-3:
                        fa.goto_gripper(gripper_target, block=False, speed=0.08)
                        last_gripper_width = gripper_target
                    
                    # 4. 调试输出：每帧都输出
                    elapsed_time = i * self.control_dt
                    
                    # 每秒输出一次详细信息（10Hz = 每10帧，20Hz = 每20帧）
                    frames_per_second = int(1.0 / self.control_dt)
                    if i % frames_per_second == 0 or i < 5:  # 前5帧也输出
                        # 目标位姿 6 维
                        target_pos = commanded_pose.translation
                        target_rotvec = Rotation.from_matrix(commanded_pose.rotation).as_rotvec()
                        target_rot_deg = np.rad2deg(target_rotvec)
                        
                        # 原始 action 信息
                        pos_increment_mm = raw_action[:3] * 1000
                        rot_increment_deg = np.rad2deg(raw_action[3:6])
                        
                        # 实际位姿
                        actual_pose = fa.get_pose()
                        actual_pos = actual_pose.translation
                        actual_rotvec = Rotation.from_matrix(actual_pose.rotation).as_rotvec()
                        
                        # 误差计算
                        pos_error = np.linalg.norm(target_pos - actual_pos)
                        
                        # 正确的旋转误差计算：通过相对旋转矩阵
                        rot_diff_matrix = commanded_pose.rotation @ actual_pose.rotation.T
                        rot_error_angle = np.arccos(np.clip((np.trace(rot_diff_matrix) - 1) / 2, -1.0, 1.0))
                        rot_error_deg = np.rad2deg(rot_error_angle)
                        
                        # 检测 axis-angle 跳变（用于调试）
                        rotvec_diff_simple = target_rotvec - actual_rotvec
                        rotvec_error_simple = np.linalg.norm(rotvec_diff_simple)
                        
                        print(f"\n[帧 {i+1}/{len(self.actions)}] t={elapsed_time:.2f}s", flush=True)
                        print(f"  目标位姿: pos=({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})m", flush=True)
                        print(f"             rot=({target_rot_deg[0]:.2f}, {target_rot_deg[1]:.2f}, {target_rot_deg[2]:.2f})deg", flush=True)
                        print(f"  实际位姿: pos=({actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f})m", flush=True)
                        print(f"             rot=({np.rad2deg(actual_rotvec[0]):.2f}, {np.rad2deg(actual_rotvec[1]):.2f}, {np.rad2deg(actual_rotvec[2]):.2f})deg", flush=True)
                        print(f"  误差:     pos_error={pos_error*1000:.2f}mm, rot_error={rot_error_deg:.2f}deg (正确)", flush=True)
                        print(f"            [DEBUG] axis-angle直接差={np.rad2deg(rotvec_error_simple):.2f}deg (错误，有跳变)", flush=True)
                        print(f"  Action增量: dpos=({pos_increment_mm[0]:.3f}, {pos_increment_mm[1]:.3f}, {pos_increment_mm[2]:.3f})mm", flush=True)
                        print(f"              drot=({rot_increment_deg[0]:.3f}, {rot_increment_deg[1]:.3f}, {rot_increment_deg[2]:.3f})deg", flush=True)
                        print(f"  夹爪: raw={raw_action[6]:.3f} -> target={gripper_target*1000:.1f}mm, current={last_gripper_width*1000:.1f}mm", flush=True)
                        
                        # 警告：如果跟踪误差过大
                        if pos_error > 0.02:  # 20mm
                            print(f"  ⚠️  警告: 位置跟踪误差过大 ({pos_error*1000:.1f}mm)", flush=True)
                        if rot_error_deg > 5.0:  # 5度
                            print(f"  ⚠️  警告: 旋转跟踪误差过大 ({rot_error_deg:.1f}deg)", flush=True)
                    
                    # 4. Min-jerk 插值发送
                    # 采集时是 20Hz 控制，但没有插值（直接发送 commanded_pose）
                    # 复现时为了平滑，使用 50Hz 插值
                    interp_dt = 0.02  # 50Hz 插值
                    interp_steps = max(1, int(self.control_dt / interp_dt))
                    
                    for step in range(1, interp_steps + 1):
                        if not self.running:
                            break
                        
                        t = step * interp_dt
                        w = min_jerk_weight(t, self.control_dt)
                        interp_pose = goal_pose.interpolate_with(start_pose, 1.0 - w)
                        
                        msg_id += 1
                        timestamp = fa.get_time() - init_time
                        
                        msg_pose = PosePositionSensorMessage(
                            id=msg_id,
                            timestamp=timestamp,
                            position=interp_pose.translation,
                            quaternion=interp_pose.quaternion
                        )
                        msg_imp = CartesianImpedanceSensorMessage(
                            id=msg_id,
                            timestamp=timestamp,
                            translational_stiffnesses=[600.0, 600.0, 600.0],
                            rotational_stiffnesses=[50.0, 50.0, 50.0]
                        )
                        ros_msg = make_sensor_group_msg(
                            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                                msg_pose, SensorDataMessageType.POSE_POSITION
                            ),
                            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                                msg_imp, SensorDataMessageType.CARTESIAN_IMPEDANCE
                            )
                        )
                        fa.publish_sensor_data(ros_msg)
                        time.sleep(interp_dt)
                else:
                    # 无机械臂模式，打印 action 信息
                    if i < 10:
                        pos_mm = action[:3] * 1000
                        rot_deg = np.rad2deg(action[3:6])
                        gripper = action[6]
                        dt_ms = self.control_dt * 1000
                        
                        logger.info(
                            f"{i+1:4d} | {dt_ms:6.1f} | "
                            f"{pos_mm[0]:7.3f} {pos_mm[1]:7.3f} {pos_mm[2]:7.3f} | "
                            f"{rot_deg[0]:8.3f} {rot_deg[1]:8.3f} {rot_deg[2]:8.3f} | "
                            f"{gripper:7.3f}"
                        )
                    
                    # 按照采集频率等待
                    time.sleep(self.control_dt)
                
                # 定期打印进度（每10帧）
                if i % 10 == 0:
                    progress = (i + 1) / len(self.actions) * 100
                    pos = state[:3]
                    gripper = state[6] - state[7]
                    print(
                        f"[进度] {i+1}/{len(self.actions)} ({progress:.1f}%) "
                        f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
                        f"gripper={gripper:.3f}m",
                        flush=True
                    )
            
            print("=" * 60, flush=True)
            print("轨迹复现完成！", flush=True)
            print("=" * 60, flush=True)
            
        except KeyboardInterrupt:
            print("\n用户中断复现", flush=True)
        except Exception as e:
            print(f"\n复现出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # 无论正常完成还是中断，都保存 CSV
            if self.save_csv:
                try:
                    self._save_csv()
                except Exception as e:
                    print(f"\n[CSV 保存] 失败: {e}", flush=True)
            
            self.running = False
            
            # 停止 dynamic skill
            if fa is not None:
                from frankapy.proto import ShouldTerminateSensorMessage
                from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
                
                timestamp = fa.get_time() - init_time
                term_msg = ShouldTerminateSensorMessage(
                    timestamp=timestamp, should_terminate=True
                )
                term_ros_msg = make_sensor_group_msg(
                    termination_handler_sensor_msg=sensor_proto2ros_msg(
                        term_msg, SensorDataMessageType.SHOULD_TERMINATE
                    )
                )
                fa.publish_sensor_data(term_ros_msg)
                time.sleep(0.1)
                print("Dynamic skill 已停止", flush=True)
    
    def stop(self):
        """停止复现。"""
        self.running = False
        print("停止复现...", flush=True)
        self.env.stop()
    
    def _save_csv(self):
        """保存 CSV 文件（格式与 data_collection 一致）"""
        print(f"\n[CSV 保存] 开始保存...", flush=True)
        print(f"  save_csv={self.save_csv}", flush=True)
        print(f"  actions 数量={len(self.actions)}", flush=True)
        
        if len(self.actions) == 0:
            print("\n[CSV 保存] 跳过：没有数据", flush=True)
            return
        
        # 确定输出目录
        if self.output_dir is None:
            # 默认：在 episode 同目录创建 replay_ 前缀的文件夹
            self.output_dir = self.episode_path.parent / f"replay_{self.episode_path.name}"
        
        print(f"  输出目录: {self.output_dir}", flush=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据
        meta = dict(
            task_name=f"replay_{self.episode_path.parent.name}_{self.episode_path.name}",
            collect_hz=self.control_hz,
            max_frames=len(self.actions),
            num_frames=len(self.actions),
            source_episode=str(self.episode_path),
            speed_factor=self.speed_factor,
        )
        
        # 保存 CSV
        path_csv = self.output_dir / "data.csv"
        print(f"  CSV 路径: {path_csv}", flush=True)
        
        with open(path_csv, "w", newline="") as f:
            w = csv.writer(f)
            # 第1行：元数据（JSON）
            w.writerow([json.dumps(meta, ensure_ascii=False)])
            # 第2行：列名
            w.writerow([f"state_{i}" for i in range(8)] + [f"action_{i}" for i in range(7)])
            # 数据行：直接使用加载的 actions
            for step_data in self.actions:
                state = step_data["state"]
                action = step_data["action"]
                w.writerow(state + action)
        
        print(f"[CSV 保存] ✓ 完成", flush=True)
        print(f"  路径: {path_csv}", flush=True)
        print(f"  帧数: {len(self.actions)}", flush=True)
        print(f"  元数据: {json.dumps(meta, indent=2, ensure_ascii=False)}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="轨迹复现脚本")
    
    # 指定 episode 的两种方式
    parser.add_argument(
        "--episode", 
        type=str,
        help="Episode 数据目录路径，如 collected/simple_pick_place/epo_1"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="任务名称，如 simple_pick_place"
    )
    parser.add_argument(
        "--epo",
        type=int,
        help="Episode 编号，如 1"
    )
    
    # 控制参数
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="速度倍率（默认 1.0，2.0 表示2倍速）"
    )
    parser.add_argument(
        "--cam1_serial",
        type=str,
        default="346222072769",
        help="外部相机序列号"
    )
    parser.add_argument(
        "--cam2_serial",
        type=str,
        default="938422075745",
        help="腕部相机序列号"
    )
    parser.add_argument(
        "--no_robot",
        action="store_true",
        help="不连接机械臂（测试模式）"
    )
    
    # CSV 保存
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="保存复现的 state 和 action 到 CSV（格式与 data_collection 一致）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="CSV 输出目录（默认: episode 同目录/replay_epo_X）"
    )
    
    # 基目录
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="数据基目录（默认: 项目根目录/collected）"
    )
    
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True  # 强制覆盖其他日志配置
    )
    
    # 确保日志输出
    print("=" * 60, flush=True)
    print("轨迹复现脚本启动", flush=True)
    print("=" * 60, flush=True)
    
    args = parse_args()
    
    # 确定 episode 路径
    if args.episode:
        episode_path = pathlib.Path(args.episode)
        if not episode_path.is_absolute():
            # 相对路径：优先使用项目根目录（向上3级：vla_control -> src -> vla4desk）
            project_root = pathlib.Path(__file__).parent.parent.parent
            episode_path = project_root / episode_path
    elif args.task and args.epo:
        if args.base_dir:
            base_dir = pathlib.Path(args.base_dir)
        else:
            # 默认 collected 目录在项目根目录
            # 容器内: /root/Documents/my_code/vla4desk/collected
            # 本地:   ~/franka_my_code/vla4desk/collected
            project_root = pathlib.Path(__file__).parent.parent.parent
            base_dir = project_root / "collected"
        episode_path = base_dir / args.task / f"epo_{args.epo}"
    else:
        logger.error("请指定 --episode 或同时指定 --task 和 --epo")
        sys.exit(1)
    
    if not episode_path.exists():
        logger.error(f"Episode 目录不存在: {episode_path}")
        logger.info(f"提示: 可使用 --base_dir 指定数据目录的绝对路径")
        sys.exit(1)
    
    # 创建复现器
    replayer = TrajectoryReplayer(
        episode_path=episode_path,
        cam1_serial=args.cam1_serial,
        cam2_serial=args.cam2_serial,
        speed_factor=args.speed,
        no_robot=args.no_robot,
        save_csv=args.save_csv,
        output_dir=args.output_dir,
    )
    
    try:
        replayer.start()
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    finally:
        replayer.stop()
        logger.info("程序退出")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
按已采集的 JSON 轨迹回放机械臂，并将 replay 时的 state/joint_state/action 记录到新 JSON。
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "vla_control"))

from franka_env import CONTROL_DT, FrankaEnv  # noqa: E402


COLLECTED_DIR = REPO_ROOT / "collected"
DEFAULT_CAM1_SERIAL = "346222072769"
DEFAULT_CAM2_SERIAL = "938422075745"


def _load_episode_json(json_path: pathlib.Path) -> tuple[dict, list[dict[str, list[float]]]]:
    with open(json_path, "r") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {json_path}")

    samples = payload.get("frames")
    if not isinstance(samples, list):
        raise ValueError(f"JSON 缺少 frames 数组: {json_path}")
    return payload, samples


def _decode_stored_action(action: list[float], action_scale: float) -> np.ndarray:
    """从 JSON 读取 action：仅还原前 6 维，夹爪维度保持原值。"""
    decoded = np.asarray(action, dtype=np.float64).copy()
    decoded[:6] /= action_scale
    return decoded


def _encode_action_for_storage(action: np.ndarray, action_scale: float) -> list[float]:
    """写回 JSON 时仅缩放前 6 维，夹爪维度保持 -1/1 原值。"""
    encoded = np.asarray(action, dtype=np.float64).copy()
    encoded[:6] *= action_scale
    return encoded.tolist()


def _resolve_episode_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.episode:
        episode_dir = pathlib.Path(args.episode).expanduser().resolve()
    else:
        if args.task is None or args.epo is None:
            raise ValueError("必须提供 --episode，或同时提供 --task 与 --epo")
        episode_dir = (COLLECTED_DIR / args.task / f"epo_{args.epo}").resolve()

    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode 目录不存在: {episode_dir}")
    if not (episode_dir / "data.json").is_file():
        raise FileNotFoundError(f"未找到 data.json: {episode_dir / 'data.json'}")
    return episode_dir


def _default_output_dir(episode_dir: pathlib.Path) -> pathlib.Path:
    name = episode_dir.name
    if name.startswith("epo_"):
        base_name = f"replay_{name}"
    else:
        base_name = f"replay_{name}"

    candidate = episode_dir.parent / base_name
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        fallback = episode_dir.parent / f"{base_name}_{index}"
        if not fallback.exists():
            return fallback
        index += 1


def _save_json(
    output_dir: pathlib.Path,
    payload: dict,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return json_path


class TrajectoryReplayRecorder:
    def __init__(
        self,
        episode_dir: pathlib.Path,
        output_dir: pathlib.Path,
        replay_hz: float | None,
        speed: float,
        no_robot: bool,
        cam1_serial: str | None,
        cam2_serial: str | None,
        prompt: str,
    ):
        self.episode_dir = episode_dir
        self.output_dir = output_dir
        self.source_json = episode_dir / "data.json"
        self.source_meta, self.samples = _load_episode_json(self.source_json)
        self.action_scale = float(self.source_meta.get("action_scale", 100.0))

        source_hz = float(self.source_meta.get("collect_hz", 10.0))
        if replay_hz is not None:
            self.replay_hz = replay_hz
        else:
            self.replay_hz = source_hz * speed

        if self.replay_hz <= 0:
            raise ValueError("回放频率必须大于 0")

        self.dt = 1.0 / self.replay_hz
        self.speed = speed
        self.prompt = prompt
        self.logged_rows: list[dict[str, list[float]]] = []
        self.no_robot = no_robot
        self.env = FrankaEnv(
            cam1_serial=cam1_serial,
            cam2_serial=cam2_serial,
            no_robot=no_robot,
        )

    def _prepare_env(self):
        self.env.start()
        if not self.no_robot:
            logging.info("回放前复位到 home...")
            self.env.reset_to_home()
        self.env.start_skill_thread()
        time.sleep(1.0)

    def _record_sample(self, sample: dict[str, list[float]]):
        obs, _, _ = self.env.get_observation(self.prompt)
        state = obs["observation/state"].astype(np.float64).tolist()
        joint_state = obs["observation/joints"].astype(np.float64).tolist()
        action = _decode_stored_action(sample["action"], self.action_scale)
        self.logged_rows.append(
            {
                "state": state,
                "joint_state": joint_state,
                "action": _encode_action_for_storage(action, self.action_scale),
            }
        )
        self.env.enqueue_action(action, transform=False)

    def _sleep_until(self, next_tick: float):
        remaining = next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def _build_output_payload(self) -> dict:
        return {
            "task_name": self.output_dir.name,
            "collect_hz": self.replay_hz,
            "max_frames": len(self.logged_rows),
            "num_frames": len(self.logged_rows),
            "action_scale": self.action_scale,
            "prompt": self.prompt,
            "source_episode": str(self.episode_dir),
            "source_json": str(self.source_json),
            "speed_factor": self.speed,
            "frames": self.logged_rows,
        }

    def run(self):
        logging.info("加载轨迹: %s", self.source_json)
        logging.info("源轨迹帧数: %d", len(self.samples))
        logging.info("源频率: %.3f Hz, 回放频率: %.3f Hz", self.source_meta.get("collect_hz", 10.0), self.replay_hz)
        if abs(self.dt - CONTROL_DT) > 1e-6:
            logging.warning(
                "当前回放频率 %.3f Hz 与 FrankaEnv 控制频率 %.3f Hz 不一致，实际执行会受 skill loop 影响",
                self.replay_hz,
                1.0 / CONTROL_DT,
            )

        self._prepare_env()

        try:
            next_tick = time.monotonic()
            for index, sample in enumerate(self.samples, start=1):
                self._record_sample(sample)
                next_tick += self.dt
                self._sleep_until(next_tick)

                if index % max(1, int(round(self.replay_hz))) == 0 or index == len(self.samples):
                    logging.info("回放进度: %d / %d", index, len(self.samples))
        finally:
            self.env.stop()

        payload = self._build_output_payload()
        json_path = _save_json(self.output_dir, payload)
        logging.info("replay 记录已保存: %s", json_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 JSON 回放轨迹并记录 replay state/joint_state/action")
    parser.add_argument("--episode", default=None, help="源 episode 目录，例如 collected/simple_pick_place/epo_21")
    parser.add_argument("--task", default=None, help="任务名，与 --epo 配合使用")
    parser.add_argument("--epo", type=int, default=None, help="episode 编号，与 --task 配合使用")
    parser.add_argument("--output_dir", default=None, help="输出目录，默认自动生成 replay_epo_x")
    parser.add_argument("--replay_hz", type=float, default=None, help="回放与记录频率，默认使用源 JSON 的 collect_hz * speed")
    parser.add_argument("--speed", type=float, default=1.0, help="速度倍率，仅在未显式设置 --replay_hz 时生效")
    parser.add_argument("--prompt", default="", help="传给 FrankaEnv.get_observation 的 prompt")
    parser.add_argument("--cam1_serial", default=DEFAULT_CAM1_SERIAL)
    parser.add_argument("--cam2_serial", default=DEFAULT_CAM2_SERIAL)
    parser.add_argument("--no_robot", action="store_true", help="无机械臂模式，仅用于联调 JSON 流程")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_argparser().parse_args()

    if args.speed <= 0:
        raise ValueError("--speed 必须大于 0")

    episode_dir = _resolve_episode_dir(args)
    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(episode_dir)
    )

    recorder = TrajectoryReplayRecorder(
        episode_dir=episode_dir,
        output_dir=output_dir,
        replay_hz=args.replay_hz,
        speed=args.speed,
        no_robot=args.no_robot,
        cam1_serial=args.cam1_serial,
        cam2_serial=args.cam2_serial,
        prompt=args.prompt,
    )
    recorder.run()


if __name__ == "__main__":
    main()

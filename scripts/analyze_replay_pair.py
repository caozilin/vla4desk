#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np


def load_frames(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    frames = payload["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} has no frames")
    return payload, frames


def to_array(frames: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([frame[key] for frame in frames], dtype=np.float64)


def summarize_norms(name: str, values: np.ndarray, unit_scale: float = 1.0, unit: str = "") -> list[str]:
    scaled = values * unit_scale
    return [
        f"{name} mean={scaled.mean():.3f}{unit} p95={np.percentile(scaled, 95):.3f}{unit} max={scaled.max():.3f}{unit}",
    ]


def angle_between_rotvecs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return np.linalg.norm(diff, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a source trajectory JSON against a replay JSON")
    parser.add_argument("source", nargs="?", default="collected/default/epo_15/data.json")
    parser.add_argument("replay", nargs="?", default="collected/default/replay_epo_15_3/data.json")
    args = parser.parse_args()

    source_path = pathlib.Path(args.source)
    replay_path = pathlib.Path(args.replay)

    src_meta, src_frames = load_frames(source_path)
    rep_meta, rep_frames = load_frames(replay_path)

    n = min(len(src_frames), len(rep_frames))
    src_frames = src_frames[:n]
    rep_frames = rep_frames[:n]

    src_state = to_array(src_frames, "state")
    rep_state = to_array(rep_frames, "state")
    src_action = to_array(src_frames, "action")
    rep_action = to_array(rep_frames, "action")
    src_cmd = to_array(src_frames, "commanded_pose")
    rep_cmd = to_array(rep_frames, "commanded_pose")

    action_diff = np.abs(src_action - rep_action)
    state_pos_err = np.linalg.norm(src_state[:, :3] - rep_state[:, :3], axis=1)
    state_rot_err = angle_between_rotvecs(src_state[:, 3:6], rep_state[:, 3:6])
    state_grip_err = np.linalg.norm(src_state[:, 6:8] - rep_state[:, 6:8], axis=1)
    cmd_pos_err = np.linalg.norm(src_cmd[:, :3] - rep_cmd[:, :3], axis=1)
    cmd_rot_err = angle_between_rotvecs(src_cmd[:, 3:6], rep_cmd[:, 3:6])

    worst_cmd_idx = int(np.argmax(cmd_pos_err))
    worst_state_idx = int(np.argmax(state_pos_err))

    print(f"source={source_path}")
    print(f"replay={replay_path}")
    print(f"frames_compared={n}")
    print(f"source_collect_hz={src_meta.get('collect_hz')} replay_collect_hz={rep_meta.get('collect_hz')}")
    print()
    print("Action consistency")
    print(f"action abs diff max={action_diff.max():.6f}")
    print(f"action exact match={bool(np.allclose(src_action, rep_action))}")
    print()
    print("State error")
    for line in summarize_norms("state pos err", state_pos_err, unit_scale=1000.0, unit="mm"):
        print(line)
    for line in summarize_norms("state rot err", state_rot_err, unit_scale=180.0 / math.pi, unit="deg"):
        print(line)
    for line in summarize_norms("state gripper err", state_grip_err, unit_scale=1000.0, unit="mm"):
        print(line)
    print(f"worst state pos frame={worst_state_idx} src={src_state[worst_state_idx,:3].tolist()} replay={rep_state[worst_state_idx,:3].tolist()}")
    print()
    print("Commanded pose error")
    for line in summarize_norms("commanded pos err", cmd_pos_err, unit_scale=1000.0, unit="mm"):
        print(line)
    for line in summarize_norms("commanded rot err", cmd_rot_err, unit_scale=180.0 / math.pi, unit="deg"):
        print(line)
    print(f"worst commanded pos frame={worst_cmd_idx} src={src_cmd[worst_cmd_idx,:3].tolist()} replay={rep_cmd[worst_cmd_idx,:3].tolist()}")
    print(f"worst commanded full src={src_cmd[worst_cmd_idx].tolist()}")
    print(f"worst commanded full replay={rep_cmd[worst_cmd_idx].tolist()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

@dataclass
class ValidationIssue:
    suite: str
    episode: str
    issue_type: str
    severity: str
    message: str

@dataclass
class SuiteMetrics:
    suite_name: str
    total_episodes: int = 0
    valid_episodes: int = 0
    missing_cam1: int = 0
    missing_cam2: int = 0
    missing_json: int = 0
    invalid_json: int = 0
    invalid_mp4_cam1: int = 0
    invalid_mp4_cam2: int = 0
    invalid_state_shape: int = 0
    invalid_action_shape: int = 0
    invalid_state_dtype: int = 0
    invalid_action_dtype: int = 0
    empty_prompt: int = 0
    invalid_frame_count: int = 0
    cam1_zero_frames: int = 0
    cam2_zero_frames: int = 0
    cam_mismatch_frames: int = 0
    cam1_black_frames: int = 0
    cam2_black_frames: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.valid_episodes / self.total_episodes * 100) if self.total_episodes > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_episodes": self.total_episodes,
            "valid_episodes": self.valid_episodes,
            "pass_rate": f"{self.pass_rate:.2f}%",
            "issues_count": len(self.issues),
            "missing_cam1": self.missing_cam1,
            "missing_cam2": self.missing_cam2,
            "missing_json": self.missing_json,
            "invalid_json": self.invalid_json,
            "invalid_mp4_cam1": self.invalid_mp4_cam1,
            "invalid_mp4_cam2": self.invalid_mp4_cam2,
            "invalid_state_shape": self.invalid_state_shape,
            "invalid_action_shape": self.invalid_action_shape,
            "invalid_state_dtype": self.invalid_state_dtype,
            "invalid_action_dtype": self.invalid_action_dtype,
            "empty_prompt": self.empty_prompt,
            "invalid_frame_count": self.invalid_frame_count,
            "cam1_zero_frames": self.cam1_zero_frames,
            "cam2_zero_frames": self.cam2_zero_frames,
            "cam_mismatch_frames": self.cam_mismatch_frames,
            "cam1_black_frames": self.cam1_black_frames,
            "cam2_black_frames": self.cam2_black_frames,
        }


class CollectedDataValidator:
    EXPECTED_STATE_SHAPE = (8,)
    EXPECTED_ACTION_SHAPE = (7,)
    REQUIRED_JSON_KEYS = ["task_name", "collect_hz", "max_frames", "num_frames",
                          "action_scale", "prompt", "frames"]
    REQUIRED_FRAME_KEYS = ["id", "timestamp", "state", "joint_state", "action", "commanded_pose"]

    def __init__(self, collected_dir: str):
        self.collected_dir = Path(collected_dir)
        self.suites: Dict[str, SuiteMetrics] = {}
        self.all_issues: List[ValidationIssue] = []

    def validate_all(self) -> Dict[str, SuiteMetrics]:
        if not self.collected_dir.exists():
            print(f"Error: Directory {self.collected_dir} does not exist")
            return {}

        suite_dirs = sorted([d for d in self.collected_dir.iterdir() if d.is_dir()])
        print(f"Found {len(suite_dirs)} task suites in {self.collected_dir}")

        for suite_dir in suite_dirs:
            suite_name = suite_dir.name
            self.suites[suite_name] = SuiteMetrics(suite_name=suite_name)
            self._validate_suite(suite_dir, self.suites[suite_name])

        return self.suites

    def _validate_suite(self, suite_dir: Path, metrics: SuiteMetrics):
        episode_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir() and d.name.startswith("epo_")],
                             key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0)
        metrics.total_episodes = len(episode_dirs)

        for ep_dir in episode_dirs:
            ep_name = ep_dir.name
            self._validate_episode(ep_dir, ep_name, metrics)

    def _validate_episode(self, ep_dir: Path, ep_name: str, metrics: SuiteMetrics):
        cam1_path = ep_dir / "cam1.mp4"
        cam2_path = ep_dir / "cam2.mp4"
        json_path = ep_dir / "data.json"

        has_cam1 = cam1_path.exists()
        has_cam2 = cam2_path.exists()
        has_json = json_path.exists()

        if not has_cam1:
            metrics.missing_cam1 += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="missing_file", severity="error",
                message="cam1.mp4 is missing"
            ))

        if not has_cam2:
            metrics.missing_cam2 += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="missing_file", severity="error",
                message="cam2.mp4 is missing"
            ))

        if not has_json:
            metrics.missing_json += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="missing_file", severity="error",
                message="data.json is missing"
            ))
            return

        if not self._validate_mp4(cam1_path, "cam1", ep_name, metrics) or \
           not self._validate_mp4(cam2_path, "cam2", ep_name, metrics):
            return

        if not self._validate_json(json_path, ep_name, metrics):
            return

        metrics.valid_episodes += 1

    def _validate_mp4(self, path: Path, cam_name: str, ep_name: str, metrics: SuiteMetrics) -> bool:
        if not path.exists():
            return False

        try:
            if OPENCV_AVAILABLE:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    if cam_name == "cam1":
                        metrics.invalid_mp4_cam1 += 1
                    else:
                        metrics.invalid_mp4_cam2 += 1
                    metrics.issues.append(ValidationIssue(
                        suite=metrics.suite_name, episode=ep_name,
                        issue_type="invalid_mp4", severity="error",
                        message=f"{cam_name}.mp4 cannot be opened"
                    ))
                    return False

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if frame_count <= 0:
                    if cam_name == "cam1":
                        metrics.cam1_zero_frames += 1
                    else:
                        metrics.cam2_zero_frames += 1
                    metrics.issues.append(ValidationIssue(
                        suite=metrics.suite_name, episode=ep_name,
                        issue_type="zero_frames", severity="warning",
                        message=f"{cam_name}.mp4 has zero frames"
                    ))
                    cap.release()
                    return True

                black_frame_count = 0
                sample_indices = np.linspace(0, frame_count - 1, min(30, frame_count), dtype=int)
                
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        mean_pixel = np.mean(frame)
                        if mean_pixel < 5.0:
                            black_frame_count += 1
                
                cap.release()
                
                if black_frame_count > 0:
                    if cam_name == "cam1":
                        metrics.cam1_black_frames += black_frame_count
                    else:
                        metrics.cam2_black_frames += black_frame_count
                    metrics.issues.append(ValidationIssue(
                        suite=metrics.suite_name, episode=ep_name,
                        issue_type="black_frames", severity="warning",
                        message=f"{cam_name}.mp4 has {black_frame_count}/{len(sample_indices)} sampled frames that are nearly black (mean pixel < 5)"
                    ))
            return True
        except Exception as e:
            if cam_name == "cam1":
                metrics.invalid_mp4_cam1 += 1
            else:
                metrics.invalid_mp4_cam2 += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="invalid_mp4", severity="error",
                message=f"{cam_name}.mp4 error: {str(e)}"
            ))
            return False

    def _validate_json(self, json_path: Path, ep_name: str, metrics: SuiteMetrics) -> bool:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            metrics.invalid_json += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="invalid_json", severity="error",
                message=f"data.json is not valid JSON: {str(e)}"
            ))
            return False
        except Exception as e:
            metrics.invalid_json += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="invalid_json", severity="error",
                message=f"data.json read error: {str(e)}"
            ))
            return False

        for key in self.REQUIRED_JSON_KEYS:
            if key not in data:
                metrics.invalid_json += 1
                metrics.issues.append(ValidationIssue(
                    suite=metrics.suite_name, episode=ep_name,
                    issue_type="missing_key", severity="error",
                    message=f"data.json missing required key: {key}"
                ))
                return False

        if not isinstance(data.get("prompt"), str):
            metrics.invalid_json += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="invalid_prompt_type", severity="error",
                message="prompt is not a string"
            ))
            return False

        if data["prompt"].strip() == "":
            metrics.empty_prompt += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="empty_prompt", severity="warning",
                message="prompt is empty string"
            ))

        frames = data.get("frames", [])
        if not isinstance(frames, list):
            metrics.invalid_frame_count += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="invalid_frames_type", severity="error",
                message="frames is not a list"
            ))
            return False

        num_frames = data.get("num_frames", 0)
        if num_frames != len(frames):
            metrics.invalid_frame_count += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type="frame_count_mismatch", severity="warning",
                message=f"num_frames={num_frames} but actual frames={len(frames)}"
            ))

        cam1_path = json_path.parent / "cam1.mp4"
        cam2_path = json_path.parent / "cam2.mp4"

        if OPENCV_AVAILABLE and cam1_path.exists() and cam2_path.exists():
            cap1 = cv2.VideoCapture(str(cam1_path))
            cap2 = cv2.VideoCapture(str(cam2_path))
            cam1_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            cam2_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            cap1.release()
            cap2.release()

            if abs(cam1_frames - len(frames)) > 1:
                metrics.cam_mismatch_frames += 1
                metrics.issues.append(ValidationIssue(
                    suite=metrics.suite_name, episode=ep_name,
                    issue_type="cam1_frame_mismatch", severity="warning",
                    message=f"cam1.mp4 has {cam1_frames} frames but json has {len(frames)} frames"
                ))
            if abs(cam2_frames - len(frames)) > 1:
                metrics.cam_mismatch_frames += 1
                metrics.issues.append(ValidationIssue(
                    suite=metrics.suite_name, episode=ep_name,
                    issue_type="cam2_frame_mismatch", severity="warning",
                    message=f"cam2.mp4 has {cam2_frames} frames but json has {len(frames)} frames"
                ))

        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                metrics.issues.append(ValidationIssue(
                    suite=metrics.suite_name, episode=ep_name,
                    issue_type="invalid_frame", severity="error",
                    message=f"Frame {i} is not a dictionary"
                ))
                continue

            for key in self.REQUIRED_FRAME_KEYS:
                if key not in frame:
                    metrics.issues.append(ValidationIssue(
                        suite=metrics.suite_name, episode=ep_name,
                        issue_type="missing_frame_key", severity="error",
                        message=f"Frame {i} missing key: {key}"
                    ))

            state = frame.get("state")
            action = frame.get("action")

            if state is not None:
                if not self._validate_array_shape(state, self.EXPECTED_STATE_SHAPE, "state", i, ep_name, metrics):
                    pass
                elif not self._validate_float64(state, "state", i, ep_name, metrics):
                    pass

            if action is not None:
                if not self._validate_array_shape(action, self.EXPECTED_ACTION_SHAPE, "action", i, ep_name, metrics):
                    pass
                elif not self._validate_float64(action, "action", i, ep_name, metrics):
                    pass

        return True

    def _validate_array_shape(self, arr: Any, expected_shape: Tuple[int, ...],
                              arr_name: str, frame_idx: int, ep_name: str,
                              metrics: SuiteMetrics) -> bool:
        if not isinstance(arr, (list, np.ndarray)):
            return False

        arr_np = np.array(arr)
        if arr_np.shape != expected_shape:
            if arr_name == "state":
                metrics.invalid_state_shape += 1
            else:
                metrics.invalid_action_shape += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type=f"invalid_{arr_name}_shape", severity="error",
                message=f"Frame {frame_idx} {arr_name} shape is {arr_np.shape}, expected {expected_shape}"
            ))
            return False
        return True

    def _validate_float64(self, arr: Any, arr_name: str, frame_idx: int,
                          ep_name: str, metrics: SuiteMetrics) -> bool:
        arr_np = np.array(arr)
        if not np.issubdtype(arr_np.dtype, np.floating) and not np.issubdtype(arr_np.dtype, np.integer):
            if arr_name == "state":
                metrics.invalid_state_dtype += 1
            else:
                metrics.invalid_action_dtype += 1
            metrics.issues.append(ValidationIssue(
                suite=metrics.suite_name, episode=ep_name,
                issue_type=f"invalid_{arr_name}_dtype", severity="error",
                message=f"Frame {frame_idx} {arr_name} dtype is {arr_np.dtype}, expected numeric"
            ))
            return False
        return True

    def print_report(self):
        print("\n" + "="*80)
        print("COLLECTED DATA VALIDATION REPORT")
        print("="*80)

        total_suites = len(self.suites)
        total_episodes = sum(m.total_episodes for m in self.suites.values())
        total_valid = sum(m.valid_episodes for m in self.suites.values())
        total_issues = sum(len(m.issues) for m in self.suites.values())

        print(f"\n{'Summary':^80}")
        print("-"*80)
        print(f"Total task suites: {total_suites}")
        print(f"Total episodes: {total_episodes}")
        print(f"Total valid episodes: {total_valid}")
        print(f"Total issues found: {total_issues}")
        print(f"Overall pass rate: {(total_valid/total_episodes*100):.2f}%" if total_episodes > 0 else "N/A")

        for suite_name, metrics in sorted(self.suites.items()):
            print(f"\n{'='*80}")
            print(f"Suite: {suite_name}")
            print("-"*80)

            d = metrics.to_dict()
            for key, value in d.items():
                if key != "suite_name":
                    print(f"  {key}: {value}")

            if metrics.issues:
                print(f"\n  {'Issue Details':^80}")
                print(f"  {'-'*80}")
                error_issues = [i for i in metrics.issues if i.severity == "error"]
                warning_issues = [i for i in metrics.issues if i.severity == "warning"]

                if error_issues:
                    print(f"  Errors ({len(error_issues)}):")
                    for issue in error_issues[:10]:
                        print(f"    - [{issue.episode}] {issue.message}")
                    if len(error_issues) > 10:
                        print(f"    ... and {len(error_issues) - 10} more errors")

                if warning_issues:
                    print(f"  Warnings ({len(warning_issues)}):")
                    for issue in warning_issues[:10]:
                        print(f"    - [{issue.episode}] {issue.message}")
                    if len(warning_issues) > 10:
                        print(f"    ... and {len(warning_issues) - 10} more warnings")

        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)

    def save_report(self, output_path: str):
        report = {
            "summary": {
                "total_suites": len(self.suites),
                "total_episodes": sum(m.total_episodes for m in self.suites.values()),
                "total_valid": sum(m.valid_episodes for m in self.suites.values()),
                "total_issues": sum(len(m.issues) for m in self.suites.values()),
            },
            "suites": {name: m.to_dict() for name, m in self.suites.items()},
            "all_issues": [
                {
                    "suite": i.suite,
                    "episode": i.episode,
                    "type": i.issue_type,
                    "severity": i.severity,
                    "message": i.message
                }
                for i in self.all_issues
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate collected data in collected/ directory")
    parser.add_argument("--collected-dir", type=str,
                       default="/home/k324/franka_my_code/vla4desk/collected",
                       help="Path to collected data directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON report path (optional)")
    args = parser.parse_args()

    validator = CollectedDataValidator(args.collected_dir)
    validator.validate_all()
    validator.print_report()

    if args.output:
        validator.save_report(args.output)

    all_metrics = list(validator.suites.values())
    if all_metrics:
        total_valid = sum(m.valid_episodes for m in all_metrics)
        total_episodes = sum(m.total_episodes for m in all_metrics)
        if total_valid < total_episodes:
            exit(1)

if __name__ == "__main__":
    main()

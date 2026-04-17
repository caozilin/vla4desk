#!/usr/bin/env python3
"""Batch write prompts into collected episode JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取任务目录内的 prompts.txt，把 prompt 写入对应任务目录下各 episode 的 data.json。"
    )
    parser.add_argument(
        "task_dir",
        nargs="?",
        type=Path,
        help="目标任务目录，例如 collected/simple_pick_place",
    )
    parser.add_argument(
        "--fill-empty-under-collected",
        type=Path,
        help="遍历 collected 下各任务目录，只给 prompt 为空的 data.json 补写 prompt",
    )
    args = parser.parse_args()
    if (args.task_dir is None) == (args.fill_empty_under_collected is None):
        parser.error("必须且只能提供 task_dir 或 --fill-empty-under-collected 其中之一")
    return args


def _load_prompts(prompts_file: Path) -> list[str]:
    if not prompts_file.is_file():
        raise FileNotFoundError(f"未找到 prompts 文件: {prompts_file}")

    prompts = [
        line.strip()
        for line in prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"prompts 文件为空: {prompts_file}")
    return prompts


def _find_json_files(task_dir: Path) -> list[Path]:
    if not task_dir.is_dir():
        raise NotADirectoryError(f"未找到任务目录: {task_dir}")

    json_files = sorted(path for path in task_dir.rglob("data.json") if path.is_file())
    if not json_files:
        raise FileNotFoundError(
            f"任务目录下未找到 data.json: {task_dir}"
        )
    return json_files


def _find_empty_prompt_json_files(task_dir: Path) -> list[Path]:
    json_files = _find_json_files(task_dir)
    empty_prompt_files: list[Path] = []

    for json_path in json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON 顶层必须是对象: {json_path}")
        if payload.get("prompt", "") == "":
            empty_prompt_files.append(json_path)

    return empty_prompt_files


def _write_prompt(json_path: Path, prompt: str) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {json_path}")

    payload["prompt"] = prompt
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_prompts_for_task_dir(task_dir: Path, only_empty: bool) -> tuple[int, int]:
    prompts_file = task_dir / "prompts.txt"
    prompts = _load_prompts(prompts_file)
    json_files = _find_empty_prompt_json_files(task_dir) if only_empty else _find_json_files(task_dir)

    for index, json_path in enumerate(json_files):
        prompt = prompts[index % len(prompts)]
        _write_prompt(json_path, prompt)
        print(f"[{index + 1}/{len(json_files)}] {json_path} <- {prompt}")

    return len(json_files), len(prompts)


def _find_task_dirs(collected_dir: Path) -> list[Path]:
    if not collected_dir.is_dir():
        raise NotADirectoryError(f"未找到 collected 目录: {collected_dir}")

    task_dirs = sorted(
        path for path in collected_dir.iterdir() if path.is_dir() and (path / "prompts.txt").is_file()
    )
    if not task_dirs:
        raise FileNotFoundError(f"collected 下未找到带 prompts.txt 的任务目录: {collected_dir}")
    return task_dirs


def main() -> None:
    args = _parse_args()
    if args.task_dir is not None:
        _write_prompts_for_task_dir(args.task_dir, only_empty=False)
        return

    total_written = 0
    for task_dir in _find_task_dirs(args.fill_empty_under_collected):
        written_count, _ = _write_prompts_for_task_dir(task_dir, only_empty=True)
        print(f"[task] {task_dir} wrote {written_count} file(s)")
        total_written += written_count

    print(f"[done] wrote {total_written} file(s) under {args.fill_empty_under_collected}")


if __name__ == "__main__":
    main()

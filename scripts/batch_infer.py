#!/usr/bin/env python3

"""Batch runner for ViPE inference across many videos."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fnmatch import fnmatch
from pathlib import Path
from queue import SimpleQueue
from threading import Lock, Thread
from typing import List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ViPE inference utility")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input mp4 files or directories to search (recursive)",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU indices to use (default: 0)",
    )
    parser.add_argument(
        "--pipeline",
        default="default",
        help="Pipeline configuration to use for vipe infer (default: default)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose pose npz already exists",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Pass --visualize to vipe infer",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional arguments to append to vipe infer (repeatable)",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subdirectory name inside each video folder to hold outputs (default: direct parent)",
    )
    parser.add_argument(
        "--name-pattern",
        default="*.mp4",
        help="Only queue videos whose filename matches this pattern (default: *.mp4)",
    )
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=4,
        help="Number of concurrent jobs to run per GPU (default: 4)",
    )
    return parser.parse_args()


def parse_gpu_list(gpu_str: str) -> List[str]:
    gpus = [gpu.strip() for gpu in gpu_str.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("No GPUs provided")
    return gpus


def natural_sort_key(path: Path) -> List[object]:
    tokens = re.split(r"(\d+)", path.as_posix())
    key: List[object] = []
    for token in tokens:
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token)
    return key


def collect_videos(inputs: Sequence[str], pattern: str) -> List[Path]:
    videos: List[Path] = []
    for item in inputs:
        path = Path(item)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping", file=sys.stderr)
            continue
        if path.is_file():
            if fnmatch(path.name, pattern):
                videos.append(path.resolve())
            else:
                print(f"Warning: {path} does not match pattern '{pattern}', skipping", file=sys.stderr)
        elif path.is_dir():
            videos.extend(sorted(p for p in path.resolve().rglob("*.mp4") if fnmatch(p.name, pattern)))
    unique_videos = sorted(set(videos), key=natural_sort_key)
    return unique_videos


def target_pose_path(video: Path, output_subdir: str | None) -> Path:
    base_dir = video.parent if output_subdir is None else video.parent / output_subdir
    return base_dir / "pose.npz"


def create_pipeline_output_dir(video: Path, output_subdir: str | None) -> Path:
    base_dir = video.parent if output_subdir is None else video.parent / output_subdir
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f".vipe_tmp_{video.stem}_", dir=str(base_dir))
    return Path(tmp_dir)


def run_vipe(video: Path, gpu: str, args: argparse.Namespace) -> int:
    final_pose_path = target_pose_path(video, args.output_subdir)
    final_pose_path.parent.mkdir(parents=True, exist_ok=True)
    output_base = create_pipeline_output_dir(video, args.output_subdir)
    cmd: List[str] = [
        "vipe",
        "infer",
        str(video),
        "-o",
        str(output_base),
        "-p",
        args.pipeline,
    ]
    if args.visualize:
        cmd.append("--visualize")
    cmd.extend(args.extra_arg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[GPU {gpu}] Failed: {video} (exit code {result.returncode})", file=sys.stderr)
        shutil.rmtree(output_base, ignore_errors=True)
        return result.returncode

    try:
        pose_dir = output_base / "pose"
        expected_pose = pose_dir / f"{video.stem}.npz"
        if expected_pose.exists():
            source_pose = expected_pose
        else:
            npz_files = list(pose_dir.glob("*.npz")) if pose_dir.exists() else []
            if len(npz_files) == 1:
                source_pose = npz_files[0]
            else:
                raise FileNotFoundError(
                    f"Pose npz not found in {pose_dir} (expected {expected_pose.name})"
                )

        if final_pose_path.exists():
            final_pose_path.unlink()
        shutil.move(str(source_pose), str(final_pose_path))
    except Exception as exc:
        print(f"[GPU {gpu}] Error finalizing pose for {video}: {exc}", file=sys.stderr)
        shutil.rmtree(output_base, ignore_errors=True)
        return 1

    shutil.rmtree(output_base, ignore_errors=True)
    return 0


def worker(
    gpu: str,
    args: argparse.Namespace,
    queue: SimpleQueue[Path | None],
    progress: "ProgressTracker",
) -> None:
    while True:
        video = queue.get()
        if video is None:
            queue.put(None)
            break
        retcode = run_vipe(video, gpu, args)
        success = retcode == 0
        progress.record(video, success)
        if not success:
            continue


class ProgressTracker:
    def __init__(self, total: int) -> None:
        self.total = max(total, 1)
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.lock = Lock()

    def record(self, video: Path, success: bool) -> None:
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            processed = self.completed + self.failed
            fraction = processed / self.total
            elapsed = time.time() - self.start_time
            eta_sec = compute_eta(elapsed, processed, self.total)
            print(
                f"[progress] {processed}/{self.total} ({fraction * 100:.1f}%) | "
                f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta_sec)} | "
                f"done {self.completed} | failed {self.failed}",
                flush=True,
            )


def compute_eta(elapsed: float, processed: int, total: int) -> float:
    if processed <= 0 or elapsed <= 0.0:
        return float("inf")
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    if rate <= 0.0:
        return float("inf")
    return remaining / rate


def format_duration(seconds: float) -> str:
    if not (seconds >= 0.0) or seconds == float("inf"):
        return "--"
    seconds_int = int(seconds)
    hrs, rem = divmod(seconds_int, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs:d}:{mins:02d}:{secs:02d}"
    return f"{mins:d}:{secs:02d}"


def main() -> None:
    args = parse_args()
    videos = collect_videos(args.inputs, args.name_pattern)
    if not videos:
        print("No mp4 videos found.", file=sys.stderr)
        sys.exit(1)

    if args.skip_existing:
        videos = [v for v in videos if not target_pose_path(v, args.output_subdir).exists()]
        if not videos:
            print("All videos already processed. Nothing to do.")
            return

    gpu_list = parse_gpu_list(args.gpus)
    if args.processes_per_gpu <= 0:
        print("processes-per-gpu must be >= 1", file=sys.stderr)
        sys.exit(1)
    progress = ProgressTracker(len(videos))
    queue: SimpleQueue[Path | None] = SimpleQueue()
    for video in videos:
        queue.put(video)
    queue.put(None)

    threads: List[Thread] = []
    for gpu in gpu_list:
        for _ in range(args.processes_per_gpu):
            thread = Thread(target=worker, args=(gpu, args, queue, progress), daemon=True)
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    if progress.failed:
        print(f"Completed with {progress.failed} failures (see logs above).", file=sys.stderr)
    else:
        print("All jobs finished successfully.")


if __name__ == "__main__":
    main()



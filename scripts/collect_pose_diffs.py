#!/usr/bin/env python3

"""Aggregate ViPE pose estimates with rendered camera poses for analysis."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pose comparisons for Walk_East clips")
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing clip subfolders (e.g. /data2/.../Walk_East_clips)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "walk_east_pose_comparison.json",
        help="Output JSON file path (default: ./walk_east_pose_comparison.json)",
    )
    parser.add_argument(
        "--pose-relpath",
        default="vipe_results/pose.npz",
        help="Relative path from each clip directory to the ViPE pose file (default: vipe_results/pose.npz)",
    )
    parser.add_argument(
        "--camera-json",
        default="camera_poses.json",
        help="Filename of the rendered camera pose JSON inside each clip directory",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> List[object]:
    import re

    tokens = re.split(r"(\d+)", path.as_posix())
    key: List[object] = []
    for token in tokens:
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token)
    return key


def iter_clip_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root directory {root} does not exist")

    clip_dirs = [p for p in root.iterdir() if p.is_dir()]
    clip_dirs.sort(key=natural_sort_key)
    return clip_dirs


def load_pose_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "data" not in data or "inds" not in data:
        raise KeyError(f"pose npz at {path} must contain 'data' and 'inds'")
    matrices = np.asarray(data["data"], dtype=np.float64)
    indices = np.asarray(data["inds"], dtype=np.int32)
    return matrices, indices


def load_camera_json(path: Path) -> Dict[int, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    mapping: Dict[int, np.ndarray] = {}
    for frame in frames:
        idx = int(frame.get("frame_index"))
        c2w = np.asarray(frame.get("camera_to_world"), dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera_to_world for frame {idx} in {path} must be 4x4 matrix")
        mapping[idx] = c2w
    return mapping


def rotation_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    r_pred = pred[:3, :3]
    r_gt = gt[:3, :3]
    r_err = r_pred @ r_gt.T
    trace = np.clip((np.trace(r_err) - 1) / 2.0, -1.0, 1.0)
    angle = math.acos(trace)
    return math.degrees(angle)


def translation_error(pred: np.ndarray, gt: np.ndarray) -> float:
    t_pred = pred[:3, 3]
    t_gt = gt[:3, 3]
    return float(np.linalg.norm(t_pred - t_gt))


@dataclass
class FrameComparison:
    frame_index: int
    rotation_error_deg: float
    translation_error: float
    vipe_pose: List[float]
    render_pose: List[float]


def compare_clip(
    clip_dir: Path,
    pose_path: Path,
    camera_path: Path,
) -> Dict[str, object]:
    vipe_mats, vipe_inds = load_pose_npz(pose_path)
    render_map = load_camera_json(camera_path)

    comparisons: List[FrameComparison] = []
    missing_from_camera = []
    for mat, idx in zip(vipe_mats, vipe_inds):
        idx_int = int(idx)
        render_mat = render_map.get(idx_int)
        if render_mat is None:
            missing_from_camera.append(idx_int)
            continue
        rot_err = rotation_error_deg(mat, render_mat)
        trans_err = translation_error(mat, render_mat)
        comparisons.append(
            FrameComparison(
                frame_index=idx_int,
                rotation_error_deg=rot_err,
                translation_error=trans_err,
                vipe_pose=mat.astype(float).reshape(-1).tolist(),
                render_pose=render_mat.astype(float).reshape(-1).tolist(),
            )
        )

    if not comparisons:
        raise ValueError(f"No overlapping frames between {pose_path} and {camera_path}")

    rot_errors = np.array([c.rotation_error_deg for c in comparisons], dtype=np.float64)
    trans_errors = np.array([c.translation_error for c in comparisons], dtype=np.float64)

    result = {
        "clip_name": clip_dir.name,
        "clip_path": str(clip_dir),
        "pose_path": str(pose_path),
        "camera_path": str(camera_path),
        "num_frames_pose": int(len(vipe_inds)),
        "num_frames_camera": int(len(render_map)),
        "num_frames_compared": int(len(comparisons)),
        "missing_camera_frames": sorted(missing_from_camera),
        "frame_comparisons": [
            {
                "frame_index": comp.frame_index,
                "rotation_error_deg": comp.rotation_error_deg,
                "translation_error": comp.translation_error,
                "vipe_pose_flat": comp.vipe_pose,
                "render_pose_flat": comp.render_pose,
            }
            for comp in sorted(comparisons, key=lambda c: c.frame_index)
        ],
        "stats": {
            "rotation_error_deg": summary_stats(rot_errors),
            "translation_error": summary_stats(trans_errors),
        },
    }
    return result


def summary_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()

    comparisons: List[Dict[str, object]] = []
    skipped: List[str] = []
    for clip_dir in iter_clip_dirs(root):
        pose_path = clip_dir / args.pose_relpath
        camera_path = clip_dir / args.camera_json
        if not pose_path.exists():
            skipped.append(f"{clip_dir.name}: missing pose {pose_path}")
            continue
        if not camera_path.exists():
            skipped.append(f"{clip_dir.name}: missing camera {camera_path}")
            continue
        try:
            comparison = compare_clip(clip_dir, pose_path, camera_path)
        except Exception as exc:  # noqa: BLE001 - want to capture for report
            skipped.append(f"{clip_dir.name}: error {exc}")
            continue
        comparisons.append(comparison)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "total_clips": len(comparisons),
        "skipped": skipped,
        "clips": comparisons,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote pose comparison data for {len(comparisons)} clips to {output}")
    if skipped:
        print("Skipped entries:")
        for msg in skipped:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()



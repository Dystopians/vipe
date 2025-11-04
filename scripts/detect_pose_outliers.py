#!/usr/bin/env python3

"""Detect anomalous frames by comparing ViPE and rendered camera poses."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class PoseData:
    frame_index: int
    T_gt: np.ndarray  # 4x4 world->cam from ViPE
    T_rend: np.ndarray  # 4x4 world->cam from renderer


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform:
    """Compute similarity transform that maps src -> dst.

    Both arrays are shape (N, 3). Returns scale, rotation, translation
    such that dst ≈ scale * R @ src + t.
    """

    if src.shape != dst.shape:
        raise ValueError("Source and destination must have the same shape")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for Umeyama alignment")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = dst_centered.T @ src_centered / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1

    R_align = U @ D @ Vt
    var_src = np.sum(src_centered ** 2) / src.shape[0]
    if var_src <= 1e-12:
        scale = 1.0
    else:
        scale = np.sum(S * np.diag(D)) / var_src

    t_align = mu_dst - scale * R_align @ mu_src

    return SimilarityTransform(scale=float(scale), rotation=R_align, translation=t_align)


def camera_center_from_extrinsic(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return -R.T @ t


def aligned_render_extrinsic(
    T_rend: np.ndarray,
    center_aligned: np.ndarray,
    R_align: np.ndarray,
) -> np.ndarray:
    """Convert renderer world->cam to aligned world->cam after similarity transform."""

    R_rend = T_rend[:3, :3]
    R_aligned = R_rend @ R_align.T
    t_aligned = -R_aligned @ center_aligned
    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = R_aligned
    T_out[:3, 3] = t_aligned
    return T_out


def relative_transform(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_a) @ T_b


def rotation_angle_deg(R_err: np.ndarray) -> float:
    angle = Rotation.from_matrix(R_err).magnitude()
    return math.degrees(angle)


def direction_error_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def robust_threshold(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("inf")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med + 3.0 * mad + 1e-6


def smooth_track(points: np.ndarray, window: int = 5) -> np.ndarray:
    if len(points) < window:
        return points.copy()
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(points)
    for dim in range(points.shape[1]):
        padded = np.pad(points[:, dim], (window // 2, window - 1 - window // 2), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def detect_self_anomalies(track: np.ndarray) -> Dict[str, np.ndarray]:
    centers = track
    smooth = smooth_track(centers, window=7)

    diffs = np.vstack([[0, 0, 0], centers[1:] - centers[:-1]])
    speeds = np.linalg.norm(diffs, axis=1)

    accel = np.vstack([[0, 0, 0], diffs[1:] - diffs[:-1]])
    accel_norm = np.linalg.norm(accel, axis=1)

    jitter = np.linalg.norm(centers - smooth, axis=1)

    thr_speed = robust_threshold(speeds)
    thr_accel = robust_threshold(accel_norm)
    thr_jitter = robust_threshold(jitter)

    bad_speed = speeds > thr_speed
    bad_accel = accel_norm > thr_accel
    bad_jitter = jitter > thr_jitter

    bad_indices = set(np.where(bad_speed | bad_accel | bad_jitter)[0])

    return {
        "speeds": speeds,
        "accel": accel_norm,
        "jitter": jitter,
        "bad_indices": sorted(bad_indices),
        "thresholds": {
            "speed": thr_speed,
            "accel": thr_accel,
            "jitter": thr_jitter,
        },
    }


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------


def parse_clip(clip: Dict[str, object]) -> List[PoseData]:
    data = []
    for frame in clip.get("frame_comparisons", []):
        fi = int(frame["frame_index"])
        T_gt = np.asarray(frame["vipe_pose_flat"], dtype=np.float64).reshape(4, 4)
        T_r = np.asarray(frame["render_pose_flat"], dtype=np.float64).reshape(4, 4)
        data.append(PoseData(frame_index=fi, T_gt=T_gt, T_rend=T_r))
    data.sort(key=lambda x: x.frame_index)
    return data


def analyze_clip(clip: Dict[str, object]) -> Dict[str, object]:
    poses = parse_clip(clip)
    if len(poses) < 2:
        return {
            "clip_name": clip.get("clip_name", "unknown"),
            "bad_frames": [],
            "self_filter_frames": [],
            "status": "too_short",
        }

    centers_gt = np.array([camera_center_from_extrinsic(p.T_gt) for p in poses])
    centers_r = np.array([camera_center_from_extrinsic(p.T_rend) for p in poses])

    self_check = detect_self_anomalies(centers_r)
    self_bad_set = set(self_check["bad_indices"])

    try:
        sim = umeyama_alignment(centers_r, centers_gt)
    except ValueError:
        return {
            "clip_name": clip.get("clip_name", "unknown"),
            "bad_frames": sorted(p.frame_index for p in poses),
            "self_filter_frames": sorted(poses[i].frame_index for i in self_bad_set),
            "status": "alignment_failed",
        }

    centers_r_aligned = sim.scale * (centers_r @ sim.rotation.T) + sim.translation

    pos_err = np.linalg.norm(centers_r_aligned - centers_gt, axis=1)

    T_r_aligned = [
        aligned_render_extrinsic(p.T_rend, c_aligned, sim.rotation)
        for p, c_aligned in zip(poses, centers_r_aligned)
    ]

    rot_err_list: List[float] = []
    dir_err_list: List[float] = []

    for idx in range(len(poses) - 1):
        T_gt_rel = relative_transform(poses[idx].T_gt, poses[idx + 1].T_gt)
        T_r_rel = relative_transform(T_r_aligned[idx], T_r_aligned[idx + 1])

        R_err = T_gt_rel[:3, :3] @ T_r_rel[:3, :3].T
        rot_err_list.append(rotation_angle_deg(R_err))

        t_gt_rel = T_gt_rel[:3, 3]
        t_r_rel = T_r_rel[:3, 3]
        if np.linalg.norm(t_gt_rel) < 5e-2 or np.linalg.norm(t_r_rel) < 5e-2:
            dir_err_list.append(0.0)
        else:
            dir_err_list.append(direction_error_deg(t_gt_rel, t_r_rel))

    thr_pos = min(1.5, max(0.3, robust_threshold(pos_err)))
    thr_rot = min(10.0, max(5.0, robust_threshold(rot_err_list)))
    thr_dir = min(60.0, max(15.0, robust_threshold(dir_err_list)))

    bad = []
    for i, pose in enumerate(poses):
        is_bad = pos_err[i] > thr_pos
        if i > 0 and i - 1 < len(rot_err_list):
            is_bad = is_bad or rot_err_list[i - 1] > thr_rot or dir_err_list[i - 1] > thr_dir
        if i < len(rot_err_list):
            is_bad = is_bad or rot_err_list[i] > thr_rot or dir_err_list[i] > thr_dir
        if i in self_bad_set:
            is_bad = True
        if is_bad:
            bad.append(pose.frame_index)

    return {
        "clip_name": clip.get("clip_name", "unknown"),
        "bad_frames": bad,
        "self_filter_frames": sorted(poses[i].frame_index for i in self_bad_set),
        "status": "ok" if not bad else "has_outliers",
        "num_frames": len(poses),
        "stats": {
            "thr_pos": thr_pos,
            "thr_rot": thr_rot,
            "thr_dir": thr_dir,
            "max_pos": float(np.max(pos_err)) if len(pos_err) else 0.0,
            "max_rot": float(np.max(rot_err_list)) if len(rot_err_list) else 0.0,
            "max_dir": float(np.max(dir_err_list)) if len(dir_err_list) else 0.0,
        },
    }


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def categorize_clip(entry: Dict[str, object]) -> str:
    status = entry.get("status", "")
    num_frames = int(entry.get("num_frames", 0))
    bad_frames: List[int] = entry.get("bad_frames", [])  # type: ignore[assignment]
    self_frames: List[int] = entry.get("self_filter_frames", [])  # type: ignore[assignment]
    stats: Dict[str, float] = entry.get("stats", {})  # type: ignore[assignment]

    num_bad = len(bad_frames)
    num_self = len(self_frames)
    ratio_bad = num_bad / num_frames if num_frames else 0.0
    ratio_self = num_self / num_frames if num_frames else 0.0

    max_pos = float(stats.get("max_pos", 0.0)) if stats else 0.0
    max_rot = float(stats.get("max_rot", 0.0)) if stats else 0.0
    max_dir = float(stats.get("max_dir", 0.0)) if stats else 0.0

    if status == "alignment_failed":
        return "poor"

    if num_bad == 0 and num_self == 0:
        if max_pos < 0.3 and max_rot < 3.0 and max_dir < 15.0:
            return "excellent"
        return "good"

    if num_bad == 0 and num_self > 0:
        return "self_filtered"

    if ratio_bad <= 0.05 and max_pos <= 1.2 and max_rot <= 8.0 and max_dir <= 70.0:
        return "good"

    severe = (
        ratio_bad >= 0.6
        or max_pos > 3.0
        or max_rot > 15.0
        or max_dir > 120.0
    )

    if severe:
        return "poor"

    moderate = (
        ratio_bad >= 0.2
        or max_pos > 2.0
        or max_rot > 10.0
        or max_dir > 90.0
    )

    if moderate:
        return "warning"

    return "good"


def classify_clips(results: List[Dict[str, object]]) -> Dict[str, List[str]]:
    summary = {
        "excellent_clips": [],
        "good_clips": [],
        "warning_clips": [],
        "poor_clips": [],
        "self_filtered_clips": [],
    }

    for entry in results:
        name = entry.get("clip_name", "unknown")
        category = categorize_clip(entry)
        entry["category"] = category

        if entry.get("self_filter_frames"):
            summary["self_filtered_clips"].append(name)

        key_map = {
            "excellent": "excellent_clips",
            "good": "good_clips",
            "warning": "warning_clips",
            "poor": "poor_clips",
            "self_filtered": "self_filtered_clips",
        }
        summary[key_map.get(category, "warning_clips")].append(name)

    # Deduplicate lists while preserving order
    for key, names in summary.items():
        seen = set()
        dedup = []
        for n in names:
            if n not in seen:
                dedup.append(n)
                seen.add(n)
        summary[key] = dedup

    return summary


def run_detection(input_path: Path) -> Dict[str, object]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips")
    if not isinstance(clips, list):
        raise ValueError("Input JSON must contain a 'clips' list")

    clip_results = [analyze_clip(clip) for clip in clips]
    summary = classify_clips(clip_results)
    return {"clips": clip_results, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect anomalous frames from pose comparison JSON")
    parser.add_argument("input", type=Path, help="Input pose comparison JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file to write anomalies (default: input stem + '_outliers.json')",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_name(f"{args.input.stem}_outliers.json")

    report = run_detection(args.input.resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    clips = report["clips"]
    total_bad = sum(len(entry["bad_frames"]) for entry in clips)
    print(f"Processed {len(clips)} clips; flagged {total_bad} frames")
    print("Summary:")
    for key, names in report["summary"].items():
        print(f"  {key}: {len(names)} clips")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()



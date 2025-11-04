#!/usr/bin/env python3

"""Filter metadata to only include clips that pass pose-quality checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter metadata based on pose outlier report")
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to original metadata.json",
    )
    parser.add_argument(
        "pose_outliers",
        type=Path,
        help="Path to walk_east_pose_outliers.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output metadata file (default: metadata_with_pose.json next to input)",
    )
    return parser.parse_args()


def load_allowed_clips(pose_outliers: Path) -> set[str]:
    with pose_outliers.open("r", encoding="utf-8") as f:
        report = json.load(f)

    summary: Dict[str, List[str]] = report.get("summary", {})  # type: ignore[assignment]
    allowed = set()
    for key in ("excellent_clips", "good_clips", "warning_clips"):
        allowed.update(summary.get(key, []))
    return allowed


def extract_clip_name(entry: dict) -> str:
    base_name = entry.get("base_name")
    if base_name is not None:
        return str(base_name)
    render_path = entry.get("outputs", {}).get("render")
    if isinstance(render_path, str):
        parts = Path(render_path).parts
        for part in reversed(parts):
            if part.isdigit():
                return part
    return ""


def filter_metadata(metadata_path: Path, allowed_clips: set[str]) -> dict:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    clip_entries = metadata.get("clips")
    if not isinstance(clip_entries, list):
        clip_entries = metadata.get("entries")
    if not isinstance(clip_entries, list):
        raise ValueError("metadata.json must contain 'clips' or 'entries' list")

    filtered = [
        clip for clip in clip_entries if extract_clip_name(clip) in allowed_clips
    ]

    new_meta = dict(metadata)
    if "clips" in new_meta:
        new_meta["clips"] = filtered
    else:
        new_meta["entries"] = filtered
    new_meta["pose_filter"] = {
        "source_outliers": str(metadata_path.resolve()),
        "allowed_clips_count": len(allowed_clips),
        "kept_clips_count": len(filtered),
        "dropped_clips_count": len(clip_entries) - len(filtered),
    }
    return new_meta


def main() -> None:
    args = parse_args()
    allowed_clips = load_allowed_clips(args.pose_outliers)

    filtered_metadata = filter_metadata(args.metadata, allowed_clips)

    if args.output is None:
        output_path = args.metadata.with_name("metadata_with_pose.json")
    else:
        output_path = args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_metadata, f, indent=2)

    print(
        "Saved filtered metadata with",
        filtered_metadata["pose_filter"]["kept_clips_count"],
        "clips to",
        output_path,
    )


if __name__ == "__main__":
    main()



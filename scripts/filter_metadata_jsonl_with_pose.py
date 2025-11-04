#!/usr/bin/env python3

"""Filter metadata JSONL entries using pose-quality summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter metadata.jsonl by pose-quality report")
    parser.add_argument("metadata", type=Path, help="Path to metadata.jsonl")
    parser.add_argument("pose_outliers", type=Path, help="Path to walk_east_pose_outliers.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Filtered output jsonl (default: metadata.filtered.jsonl alongside input)",
    )
    return parser.parse_args()


def load_allowed_clips(report_path: Path) -> Set[str]:
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    summary: Dict[str, List[str]] = report.get("summary", {})  # type: ignore[assignment]
    allowed: Set[str] = set()
    for key in ("excellent_clips", "good_clips", "warning_clips"):
        allowed.update(summary.get(key, []))
    return allowed


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clip_name_from_record(record: dict) -> str:
    if clip_id := record.get("clip_id"):
        parts = str(clip_id).split("__", 1)
        return parts[0].split("-")[-1] if parts else str(clip_id)
    video_path = record.get("video")
    if isinstance(video_path, str):
        parts = Path(video_path).parts
        for part in reversed(parts):
            if part.isdigit():
                return part
    return ""


def main() -> None:
    args = parse_args()
    output_path = args.output or args.metadata.with_name(args.metadata.stem + "_filtered.jsonl")

    allowed = load_allowed_clips(args.pose_outliers)

    kept = 0
    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(args.metadata):
            total += 1
            clip_name = clip_name_from_record(record)
            if clip_name and clip_name in allowed:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Filtered {kept}/{total} records -> {output_path}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3

"""Trim pose comparison JSON files to a subset of clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep only the first N clips in a comparison JSON")
    parser.add_argument("input", type=Path, help="Input JSON file (e.g. walk_east_pose_comparison.json)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (defaults to input stem + '_topN.json')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of clips to retain (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips")
    if not isinstance(clips, list):
        raise ValueError("Input JSON does not contain a top-level 'clips' list")

    trimmed = clips[: args.limit]
    output_data = {"clips": trimmed}

    if args.output is None:
        output_path = args.input.with_name(f"{args.input.stem}_top{args.limit}.json")
    else:
        output_path = args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Wrote {len(trimmed)} clips to {output_path}")


if __name__ == "__main__":
    main()



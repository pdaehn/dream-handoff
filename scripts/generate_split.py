#!/usr/bin/env python3
"""Regenerate or verify the final DreamHandoff episode split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dream_handoff.dataset import EpisodeSplit, create_final_episode_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="local LeRobot v3 dataset root")
    parser.add_argument("--output", type=Path, help="write the regenerated manifest here")
    parser.add_argument("--check", type=Path, help="compare regeneration with this manifest")
    parser.add_argument(
        "--print-lerobot-episodes",
        type=Path,
        metavar="MANIFEST",
        help="print train then validation episode indices as compact JSON",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.print_lerobot_episodes is not None:
        if args.dataset is not None or args.output is not None or args.check is not None:
            raise SystemExit(
                "--print-lerobot-episodes cannot be combined with regeneration options"
            )
        split = EpisodeSplit.read(args.print_lerobot_episodes)
        print(
            json.dumps(
                [*split.train_episode_indices, *split.validation_episode_indices],
                separators=(",", ":"),
            )
        )
        return
    if args.dataset is None:
        raise SystemExit("--dataset is required to regenerate the split")
    if args.output is None and args.check is None:
        raise SystemExit("select --output, --check, or both")

    split = create_final_episode_split(args.dataset)
    if args.check is not None:
        expected = EpisodeSplit.read(args.check)
        if split.to_dict() != expected.to_dict():
            raise SystemExit(f"regenerated split differs from {args.check}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        split.write(args.output)
    print(
        json.dumps(
            {
                "canonical_sha256": split.canonical_sha256,
                "episode_metadata_sha256": split.episode_metadata_sha256,
                "train_episodes": len(split.train_episode_indices),
                "validation_episodes": len(split.validation_episode_indices),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

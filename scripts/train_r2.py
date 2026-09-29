#!/usr/bin/env python3
"""Train the final DreamHandoff-compatible R2 world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dream_handoff.r2dreamer.config import R2DreamerTrainingConfig, architecture_signature
from dream_handoff.r2dreamer.training import (
    FINAL_LOSS_CONFIG,
    FINAL_PREPROCESSING_CONFIG,
    final_model_config,
    train_world_model,
)

DEFAULTS: dict[str, Any] = {
    "device": "cuda",
    "replay_storage_device": "cuda",
    "seed": 0,
    "steps": 25_000,
    "sequence_length": 64,
    "batch_size": 16,
    "num_workers": 4,
    "validation_interval": 500,
    "checkpoint_interval": 500,
    "validation_batches": 20,
    "learning_rate": 4e-5,
    "warmup_steps": 1_000,
    "cache_preprocessed_dataset": True,
}


def _json_value(value: Any) -> Any:
    """Normalize dataclass tuples to their JSON representation."""
    return json.loads(json.dumps(value))


def load_reproduction_config(path: Path) -> dict[str, Any]:
    """Validate and return settings from the public final-run config."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read R2 reproduction config {path}: {exc}") from exc
    if value.get("schema_version") != 1:
        raise ValueError("R2 reproduction config must use schema_version 1")
    model = final_model_config()
    expected_sections = {
        "model": _json_value(model.to_dict()),
        "preprocessing": _json_value(FINAL_PREPROCESSING_CONFIG.to_dict()),
        "loss": _json_value(FINAL_LOSS_CONFIG.to_dict()),
    }
    for name, expected in expected_sections.items():
        if value.get(name) != expected:
            raise ValueError(f"R2 reproduction config {name} does not match the retained trainer")
    if value.get("architecture_signature") != architecture_signature(model):
        raise ValueError("R2 reproduction config architecture signature is incompatible")

    training = value.get("training")
    if not isinstance(training, dict):
        raise ValueError("R2 reproduction config training must be an object")
    fixed_semantics = {
        "effective_num_workers": 0,
        "optimizer": "laprop",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-20,
        "weight_decay": 0.0,
        "agc_clip": 0.3,
        "agc_pmin": 0.001,
        "gradient_clip_norm": None,
        "amp": False,
        "deterministic_validation": True,
        "checkpoint_filename": "latest.pt",
    }
    for name, expected in fixed_semantics.items():
        if training.get(name) != expected:
            raise ValueError(f"R2 reproduction config {name} must be {expected!r}")
    settings = {name: training[name] for name in DEFAULTS}
    R2DreamerTrainingConfig(
        dataset_root=Path("dataset"),
        split_path=Path("split.json"),
        output_dir=Path("output"),
        **settings,
    )
    return settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--replay-storage-device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--no-cache", action="store_true", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = dict(DEFAULTS)
    if args.config is not None:
        settings.update(load_reproduction_config(args.config))
    for name in DEFAULTS:
        argument = getattr(args, name, None)
        if argument is not None:
            settings[name] = argument
    if args.no_cache:
        settings["cache_preprocessed_dataset"] = False
    result = train_world_model(
        R2DreamerTrainingConfig(
            dataset_root=args.dataset,
            split_path=args.split,
            output_dir=args.output,
            cache_dir=args.cache_dir,
            resume_path=args.resume,
            video_backend=args.video_backend,
            **settings,
        )
    )
    print(
        json.dumps(
            {
                "latest_checkpoint": str(result.latest_checkpoint),
                "metrics": str(result.metrics_path),
                "resolved_config": str(result.resolved_config_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

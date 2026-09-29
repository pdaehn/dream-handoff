#!/usr/bin/env python3
"""Reproduce all five final analyses and write curated publication results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dream_handoff.analysis.handoff_prediction import run as run_handoff_prediction
from dream_handoff.analysis.hysteresis import run as run_hysteresis
from dream_handoff.analysis.hysteresis_calibration import run_reconstruction
from dream_handoff.analysis.prefix_conditioning import run as run_prefix_conditioning
from dream_handoff.analysis.prefix_conditioning import (
    run_conditioned_real_policy_replay_gate,
    run_real_policy_replay_gate,
)
from dream_handoff.analysis.rollout_characterization import run as run_rollout

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_MANIFEST = ROOT / "configs/artifacts.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _difference_block(raw: dict[str, Any]) -> dict[str, Any]:
    def norms(name: str) -> dict[str, Any]:
        return raw[name]["vector_l2_norm"]

    return {
        "all_within_bank_l2": norms("all_within_bank"),
        "bank_boundary_l2": norms("bank_boundary"),
        "strict_bank_interior_l2": norms("strict_bank_interior"),
        "boundary_to_all_within_bank_p95_l2_ratio": raw["boundary_to_all_within_bank_p95_l2_ratio"],
        "boundary_to_strict_interior_p95_l2_ratio": raw["boundary_to_strict_interior_p95_l2_ratio"],
    }


def _physical_evidence(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": artifact["artifact_id"],
        "repo_id": artifact["hf_repo"],
        "release_revision": artifact["hf_revision"],
        "capture_file": artifact["diagnostic_name"],
        "capture_sha256": artifact["sha256"],
        "collection_provenance": artifact["collection_provenance"],
    }


def curate_rollout(raw: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    source = raw["input"]
    return {
        "schema_version": 4,
        "analysis": "Candidate-Bank Behavior and Chunk Handoffs",
        "evidence": {
            **_physical_evidence(artifact),
            "episodes": source["episodes"],
            "capture_episodes": source["capture_episodes"],
            "recorded_capture_episode_ids": source["recorded_capture_episode_ids"],
            "discarded_rerecord_capture_episode_ids": source["discarded_capture_episode_ids"],
            "control_steps": source["control_rows"],
            "candidate_banks": source["candidate_banks"],
            "requests": source["requests"],
            "handoffs": source["handoffs"],
        },
        "candidate_selection": raw["candidate_selection"],
        "candidate_matching_geometry": raw["candidate_matching_geometry"],
        "candidate_diversity": raw["candidate_diversity"],
        "action_space_differences": {
            "definition": raw["action_differences"]["definition"],
            "first_difference": _difference_block(raw["action_differences"]["first_difference"]),
            "second_difference": _difference_block(raw["action_differences"]["second_difference"]),
        },
    }


def curate_hysteresis(
    raw: dict[str, Any], artifact: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "switch_count",
        "switch_rate",
        "operational_switch_count",
        "operational_switch_rate",
        "rapid_successive_switch_within_3_rows_count",
        "reversal_aba_count",
        "mean_matching_cost",
        "top1_match_rate",
        "dwell_ticks",
    )

    def selector(name: str) -> dict[str, Any]:
        result = {field: raw[name][field] for field in fields}
        result["mean_matching_distance_penalty"] = raw[name]["mean_regret"]
        return result

    return {
        "schema_version": 3,
        "analysis": "Selector Stabilization with Hysteresis",
        "evidence": {**_physical_evidence(artifact), "rows": raw["rows"]},
        "rule": raw["rule"],
        "tau": raw["tau"],
        "metric_definitions": {
            "rapid_successive_switch_within_3_rows_count": (
                "switch events occurring within 3 control rows of the preceding switch"
            ),
            "reversal_aba_count": (
                "A-to-B-to-A candidate patterns formed by switches on consecutive control rows"
            ),
        },
        "memoryless_nearest": selector("memoryless_nearest"),
        "absolute_hysteresis": selector("absolute_hysteresis"),
        "factual_selection_match_rate": raw["factual_selection_match_rate"],
        "offline_calibration": {
            "result": "results/hysteresis_calibration.json",
            "publication_result_sha256": artifact["publication_calibration_result_sha256"],
            "phase0_rng": calibration["runtime_calibration"]["phase0_rng"],
            "selected_tau": calibration["runtime_calibration"]["selected_tau"],
            "calibration_memoryless": calibration["runtime_calibration"]["calibration"][
                "memoryless_nearest"
            ],
            "calibration_selected_point": calibration["runtime_calibration"]["calibration"][
                "selected_point"
            ],
            "pareto_curve": calibration["runtime_calibration"]["sweep"]["pareto_curve"],
            "moderate_rule": calibration["runtime_calibration"]["sweep"]["moderate_rule"],
            "held_out_evaluation": calibration["runtime_calibration"]["held_out_evaluation"],
            "metric_distinction": (
                "Offline calibration uses recorded-continuation regret; prospective physical "
                "replay uses matching-distance penalty."
            ),
        },
    }


def curate_handoff(raw: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    target_bootstrap = raw["episode_clustered_bootstrap"][
        "target_selected_best_of_n_static_minus_exact_history"
    ]
    path_bootstrap = raw["episode_clustered_bootstrap"][
        "action_path_nearest_static_minus_exact_history"
    ]
    for stratum in ("0", "1", ">=2", "all"):
        numerical = raw["numerical_summary"][stratum]
        strata[stratum] = {
            "events": numerical["events"],
            "factual_path_mismatch": numerical["path_error"],
            "exact_history_wm_error": numerical["exact_history_wm_error"],
            "target_selected_best_of_ten": {
                "static_wm_error": numerical["target_selected_best_of_n_static_wm_error"],
                "static_minus_exact_history_error": {
                    **numerical["target_selected_best_of_n_static_minus_exact_history"],
                    "mean_clustered_ci95_low": target_bootstrap[stratum]["mean_ci95_low"],
                    "mean_clustered_ci95_high": target_bootstrap[stratum]["mean_ci95_high"],
                },
            },
            "action_path_nearest": {
                "static_wm_error": numerical["action_path_nearest_static_wm_error"],
                "static_minus_exact_history_error": {
                    **numerical["action_path_nearest_static_minus_exact_history"],
                    "mean_clustered_ci95_low": path_bootstrap[stratum]["mean_ci95_low"],
                    "mean_clustered_ci95_high": path_bootstrap[stratum]["mean_ci95_high"],
                },
            },
        }
    return {
        "schema_version": 3,
        "analysis": "Activation-Time Grounding",
        "evidence": _physical_evidence(artifact),
        "event_inclusion": raw["event_inclusion"],
        "switch_strata": strata,
        "metric_definitions": {
            **{
                key: value
                for key, value in raw["metric_definitions"].items()
                if key != "target_selected_best_of_n"
            },
            "target_selected_best_of_ten": (
                "among ten static imagined endpoints, choose the one with minimum "
                "activation-posterior target error"
            ),
        },
        "bootstrap": raw["episode_clustered_bootstrap"]["settings"],
        "zero_switch_sanity": raw["zero_switch_sanity"],
        "interpretation": (
            "Factual path mismatch grows with switching. Exact factual action history does not "
            "eliminate the activation-posterior grounding gap, and whether its endpoint error "
            "is lower or higher than a static counterfactual depends on comparator selection."
        ),
    }


def curate_prefix(
    raw: dict[str, Any], artifact: dict[str, Any], generated_artifact: dict[str, Any]
) -> dict[str, Any]:
    source = raw["numerical_summary"]
    condition_fields = {
        "command_seam": "command_seam",
        "action_difference_seam": "velocity_seam",
        "reactive_reference_error": "reactive_best_control",
        "candidate_diversity": "diversity_control",
        "state_seam": "state_seam",
    }
    conditions: dict[str, Any] = {}
    for condition in ("plain", "static", "gt", "static_live", "reactive"):
        values = source["by_condition"][condition]
        conditions[condition] = {
            output: values[input_name]
            for output, input_name in condition_fields.items()
            if input_name in values
        }

    paired: dict[str, Any] = {}
    for first, second in (
        ("static", "plain"),
        ("gt", "plain"),
        ("gt", "static"),
        ("static_live", "plain"),
    ):
        for output_metric, input_metric in (
            ("command_seam", "command_seam"),
            ("action_difference_seam", "velocity_seam"),
        ):
            input_key = f"{first}_minus_{second}_{input_metric}"
            output_key = f"{first}_minus_{second}_{output_metric}"
            observed = source["paired_differences"][input_key]
            ci = source["episode_clustered_bootstrap"][input_key]
            paired[output_key] = {
                **observed,
                "mean_clustered_ci95_low": ci["mean_ci95_low"],
                "mean_clustered_ci95_high": ci["mean_ci95_high"],
            }

    return {
        "schema_version": 3,
        "analysis": "Prefix-Conditioned Asynchronous Handoffs",
        "evidence": {
            **_physical_evidence(artifact),
            "generated_banks": generated_artifact["filename"],
            "generated_banks_sha256": generated_artifact["sha256"],
        },
        "event_inclusion": raw["event_inclusion"],
        "condition_definitions": raw["conditions"],
        "display_order": ["plain", "static", "gt", "static_live"],
        "condition_means": conditions,
        "paired_effects": paired,
        "matched_delay_static_fraction_of_exact_prefix_benefit": {
            "command_seam": raw["matched_L_e_static_fraction_of_exact_prefix_benefit"][
                "command_seam"
            ],
            "action_difference_seam": raw["matched_L_e_static_fraction_of_exact_prefix_benefit"][
                "velocity_seam"
            ],
        },
        "plain_replay_gate": raw["gate_0"],
        "bootstrap": {
            "unit": raw["statistics"]["bootstrap_unit"],
            "episode_clusters": raw["statistics"]["episode_clusters"],
            "resamples": raw["statistics"]["bootstrap_resamples"],
            "seed": raw["statistics"]["bootstrap_seed"],
        },
        "interpretation": raw["interpretation"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--calibration-dataset", type=Path, required=True)
    parser.add_argument("--calibration-policy", type=Path, required=True)
    parser.add_argument("--r2-checkpoint", type=Path, required=True)
    parser.add_argument("--generated-banks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, help="also regenerate publication figures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--calibration-diagnostics",
        type=Path,
        help="optional run-specific calibration diagnostics outside the publication directory",
    )
    parser.add_argument(
        "--calibration-reference-matrix",
        type=Path,
        help="optional reference matrix used only in --calibration-diagnostics",
    )
    parser.add_argument("--policy-path", type=Path, help="run optional exact SmolVLA replay gates")
    return parser


def reproduce(args: argparse.Namespace) -> list[str]:
    """Run the public five-result reproduction path for parsed CLI arguments."""
    manifest = json.loads(ARTIFACT_MANIFEST.read_text())
    evidence = manifest["final_evidence"]
    _verify_file(args.diagnostic, evidence["sha256"], label="diagnostic artifact")
    _verify_file(args.r2_checkpoint, manifest["r2"]["checkpoint_sha256"], label="R2 checkpoint")
    _verify_file(
        args.generated_banks,
        manifest["prefix_conditioning_replay"]["sha256"],
        label="generated-bank artifact",
    )
    calibration_path = args.output_dir / "hysteresis_calibration.json"
    calibration = run_reconstruction(
        dataset_root=args.calibration_dataset,
        policy_path=args.calibration_policy,
        checkpoint_path=args.r2_checkpoint,
        split_path=ROOT / "configs/data/episode_split.json",
        output_path=calibration_path,
        device=args.device,
        reference_matrix=args.calibration_reference_matrix,
        diagnostics_output=args.calibration_diagnostics,
    )
    _verify_file(
        calibration_path,
        evidence["publication_calibration_result_sha256"],
        label="deterministic publication hysteresis calibration result",
    )

    rollout = run_rollout(args.diagnostic)
    hysteresis = run_hysteresis(args.diagnostic, threshold=evidence["hysteresis_tau"])
    handoff = run_handoff_prediction(args.diagnostic, args.r2_checkpoint, device=args.device)
    prefix = run_prefix_conditioning(args.diagnostic, args.generated_banks)

    outputs = {
        "rollout_characterization.json": curate_rollout(rollout, evidence),
        "hysteresis.json": curate_hysteresis(hysteresis, evidence, calibration),
        "handoff_prediction.json": curate_handoff(handoff, evidence),
        "prefix_conditioning.json": curate_prefix(
            prefix, evidence, manifest["prefix_conditioning_replay"]
        ),
        "hysteresis_calibration.json": calibration,
    }
    for filename, value in outputs.items():
        _write_json(args.output_dir / filename, value)

    if args.policy_path is not None:
        model = args.policy_path / manifest["smolvla"]["model_filename"]
        _verify_file(
            model,
            manifest["smolvla"]["model_safetensors_sha256"],
            label="SmolVLA model",
        )
        plain_gate = run_real_policy_replay_gate(
            args.diagnostic, args.policy_path, device=args.device
        )
        conditioned_gate = run_conditioned_real_policy_replay_gate(
            args.diagnostic,
            args.policy_path,
            args.generated_banks,
            device=args.device,
        )
        if not plain_gate["passed"] or not conditioned_gate["passed"]:
            raise RuntimeError("optional real-policy replay verification failed")

    if args.figures is not None:
        from plot_results import generate_figures

        generate_figures(args.output_dir, args.figures)

    return sorted(outputs)


def main() -> None:
    args = build_parser().parse_args()
    outputs = reproduce(args)
    print(json.dumps({"results": outputs, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

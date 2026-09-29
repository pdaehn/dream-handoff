"""Characterize candidate selection and action-space seams in a factual rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dream_handoff.capture import DiagnosticCapture, load_diagnostic_capture


def _control_rows(capture: DiagnosticCapture) -> np.ndarray:
    step = capture.array("control_step").astype(np.int64, copy=False)
    episode = capture.array("episode_id").astype(np.int64, copy=False)
    keep = np.ones(len(step), dtype=bool)
    if len(step) > 1:
        keep[:-1] = (episode[:-1] != episode[1:]) | (step[:-1] != step[1:])
    recorded = getattr(capture, "recorded_episode_ids", tuple(np.unique(episode)))
    keep &= np.isin(episode, recorded)
    return np.flatnonzero(keep)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _compact_distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
    }


def candidate_matching_geometry(
    candidate_distances: np.ndarray,
    phase: np.ndarray,
) -> dict[str, Any]:
    """Summarize nearest and runner-up distances on operational matching rows."""
    distances = np.asarray(candidate_distances, dtype=np.float64)
    phases = np.asarray(phase, dtype=np.int64)
    if distances.ndim != 2 or distances.shape[0] != len(phases):
        raise ValueError("candidate_distances and phase have incompatible shapes")
    if distances.shape[1] < 2:
        raise ValueError("candidate matching geometry requires at least two candidates")
    if not np.isfinite(distances).all():
        raise ValueError("candidate_distances contains NaN or Inf")

    operational = phases > 0
    ordered = np.sort(distances[operational], axis=1)
    d1 = ordered[:, 0]
    margin_12 = ordered[:, 1] - ordered[:, 0]
    return {
        "definition": {
            "ordering": (
                "candidate distances are sorted independently on each row: "
                + "d1 <= d2 <= ... <= dN"
            ),
            "d1": ("distance from the current/live posterior to the nearest dreamed candidate"),
            "margin_12": "d2 - d1, the runner-up distance minus the nearest distance",
            "representation": "learned R2 matching representation",
            "operational_rows": (
                "phase > 0; phase 0 is excluded because incumbent initialization has "
                "special semantics"
            ),
        },
        "operational_row_count": int(operational.sum()),
        "d1": _compact_distribution(d1),
        "margin_12": _compact_distribution(margin_12),
    }


def _difference_summary(values: np.ndarray, action_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "vector_l2_norm": _distribution(np.linalg.norm(values, axis=1)),
        "per_action_dimension_abs": {
            name: _distribution(np.abs(values[:, index])) for index, name in enumerate(action_names)
        },
    }


def action_difference_comparison(
    actions: np.ndarray,
    bank_origin: np.ndarray,
    phase: np.ndarray,
    episode: np.ndarray,
    *,
    action_names: tuple[str, ...],
    policy_chunk_size: int,
    interior_margin: int = 10,
) -> dict[str, Any]:
    """Compare first/second action differences at bank boundaries and interiors."""
    first = np.diff(actions, axis=0)
    valid_first = episode[:-1] == episode[1:]
    within_first = valid_first & (bank_origin[:-1] == bank_origin[1:])
    boundary_first = valid_first & ~within_first
    interior_upper = policy_chunk_size - interior_margin
    interior_first = within_first & (phase[:-1] >= interior_margin) & (phase[1:] < interior_upper)

    second = np.diff(actions, n=2, axis=0)
    valid_second = (episode[:-2] == episode[1:-1]) & (episode[1:-1] == episode[2:])
    within_second = (
        valid_second
        & (bank_origin[:-2] == bank_origin[1:-1])
        & (bank_origin[1:-1] == bank_origin[2:])
    )
    boundary_second = valid_second & ~within_second
    interior_second = (
        within_second
        & (phase[:-2] >= interior_margin)
        & (phase[1:-1] >= interior_margin)
        & (phase[2:] < interior_upper)
    )

    def block(values: np.ndarray, within: np.ndarray, boundary: np.ndarray, interior: np.ndarray):
        within_summary = _difference_summary(values[within], action_names)
        boundary_summary = _difference_summary(values[boundary], action_names)
        interior_summary = _difference_summary(values[interior], action_names)
        boundary_p95 = boundary_summary["vector_l2_norm"]["p95"]
        within_p95 = within_summary["vector_l2_norm"]["p95"]
        interior_p95 = interior_summary["vector_l2_norm"]["p95"]

        def ratio(denominator: float | None) -> float | None:
            if boundary_p95 is None or denominator in (None, 0):
                return None
            return boundary_p95 / denominator

        return {
            "all_within_bank": within_summary,
            "bank_boundary": boundary_summary,
            "strict_bank_interior": interior_summary,
            "boundary_to_all_within_bank_p95_l2_ratio": ratio(within_p95),
            "boundary_to_strict_interior_p95_l2_ratio": ratio(interior_p95),
            "legacy_frozen_parity": {
                "field_name": "boundary_to_within_p95_l2_ratio",
                "actual_denominator": "all_within_bank.vector_l2_norm.p95",
                "value": ratio(within_p95),
            },
        }

    return {
        "definition": {
            "space": "canonical action space; not physical velocity or acceleration",
            "episode_boundaries": "excluded",
            "bank_interior": {
                "margin": interior_margin,
                "start_inclusive": interior_margin,
                "stop_exclusive": interior_upper,
                "meaning": "strict subset of all within-bank differences",
            },
        },
        "first_difference": block(first, within_first, boundary_first, interior_first),
        "second_difference": block(second, within_second, boundary_second, interior_second),
    }


def _phase_diversity(values: np.ndarray) -> np.ndarray:
    """Frozen mean unordered-pair Euclidean distance at each phase."""
    values_64 = np.asarray(values, dtype=np.float64)
    squared_norm = np.sum(np.square(values_64), axis=-1)
    pairwise = (
        squared_norm[:, None, :]
        + squared_norm[None, :, :]
        - 2.0 * np.einsum("nhd,mhd->nmh", values_64, values_64, optimize=True)
    )
    np.maximum(pairwise, 0.0, out=pairwise)
    if len(values_64) < 2:
        return np.zeros(values_64.shape[1], dtype=np.float32)
    upper = np.triu_indices(len(values_64), k=1)
    return np.asarray(np.sqrt(pairwise)[upper[0], upper[1]].mean(axis=0), dtype=np.float32)


def _bank_diversity(capture: DiagnosticCapture, field: str) -> float | None:
    banks = capture.array(field)
    horizons = capture.array("candidate_bank_horizon").astype(np.int64, copy=False)
    keep = np.isin(capture.array("candidate_bank_episode_id"), capture.recorded_episode_ids)
    values = [
        _phase_diversity(bank[:, : int(horizon)])
        for bank, horizon in zip(banks[keep], horizons[keep], strict=True)
    ]
    finite = np.concatenate(values).astype(np.float64, copy=False)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def _same_phase_switch_deltas(
    capture: DiagnosticCapture,
    rows: np.ndarray,
    switched: np.ndarray,
) -> np.ndarray:
    bank_keys = {
        (int(episode), int(origin)): index
        for index, (episode, origin) in enumerate(
            zip(
                capture.array("candidate_bank_episode_id"),
                capture.array("candidate_bank_origin_action_count"),
                strict=True,
            )
        )
    }
    episode = capture.array("episode_id")[rows]
    origin = capture.array("bank_origin_action_count")[rows]
    phase = capture.array("phase")[rows]
    selected = capture.array("selected_candidate_index")[rows]
    banks = capture.array("candidate_bank_control_actions")
    result = []
    for target in np.flatnonzero(switched) + 1:
        bank = banks[bank_keys[(int(episode[target]), int(origin[target]))]]
        old = int(selected[target - 1])
        new = int(selected[target])
        result.append(
            float(np.linalg.norm(bank[new, int(phase[target])] - bank[old, int(phase[target])]))
        )
    return np.asarray(result, dtype=np.float64)


def analyze_capture(capture: DiagnosticCapture) -> dict[str, Any]:
    rows = _control_rows(capture)
    selected = capture.array("selected_candidate_index")[rows].astype(np.int64, copy=False)
    episode = capture.array("episode_id")[rows].astype(np.int64, copy=False)
    origin = capture.array("bank_origin_action_count")[rows].astype(np.int64, copy=False)
    phase = capture.array("phase")[rows].astype(np.int64, copy=False)
    same_bank = (episode[:-1] == episode[1:]) & (origin[:-1] == origin[1:])
    switched = same_bank & (selected[:-1] != selected[1:])
    structural = same_bank & (phase[:-1] == 0) & (phase[1:] == 1)
    operational = same_bank & ~structural

    bank_start = np.ones(len(rows), dtype=bool)
    bank_start[1:] = (episode[1:] != episode[:-1]) | (origin[1:] != origin[:-1])
    dwell: list[int] = []
    run = 0
    for index in range(len(rows)):
        if index == 0 or bank_start[index] or selected[index] != selected[index - 1]:
            if run:
                dwell.append(run)
            run = 1
        else:
            run += 1
    if run:
        dwell.append(run)
    dwell_array = np.asarray(dwell, dtype=np.float64)

    actions = capture.array("command_sent_to_robot")[rows]
    policy_chunk_size = int(capture.metadata["policy_chunk_size"])
    differences = action_difference_comparison(
        actions,
        origin,
        phase,
        episode,
        action_names=capture.action_keys,
        policy_chunk_size=policy_chunk_size,
    )
    matching_geometry = candidate_matching_geometry(
        capture.array("candidate_distances")[rows],
        phase,
    )
    switch_deltas = _same_phase_switch_deltas(capture, rows, switched)
    return {
        "analysis": "DREAM-Chunk rollout characterization",
        "input": {
            "format_version": capture.format_version,
            "episodes": capture.recorded_episode_count,
            "capture_episodes": capture.episode_count,
            "discarded_capture_episode_ids": list(capture.discarded_episode_ids),
            "recorded_capture_episode_ids": list(capture.recorded_episode_ids),
            "command_rows": int(len(rows)),
            "control_rows": int(len(rows)),
            "candidate_banks": int(
                np.isin(
                    capture.array("candidate_bank_episode_id"), capture.recorded_episode_ids
                ).sum()
            ),
            "requests": int(
                np.isin(
                    capture.array("async_request_episode_id"), capture.recorded_episode_ids
                ).sum()
            ),
            "handoffs": sum(
                handoff.request.episode_id in capture.recorded_episode_ids
                for handoff in capture.handoffs()
            ),
        },
        "candidate_selection": {
            "within_bank_transitions": int(same_bank.sum()),
            "switches": int(switched.sum()),
            "switch_rate": float(switched.sum() / same_bank.sum()),
            "operational_transitions": int(operational.sum()),
            "operational_switches": int((switched & operational).sum()),
            "structural_phase_0_to_1": {
                "transitions": int(structural.sum()),
                "switches": int((switched & structural).sum()),
            },
            "switch_rate_excluding_phase_0_to_1": float(
                (switched & operational).sum() / operational.sum()
            ),
            "dwell_ticks": {
                "runs": int(len(dwell_array)),
                "mean": float(np.mean(dwell_array)),
                "median": float(np.median(dwell_array)),
                "p90": float(np.percentile(dwell_array, 90)),
            },
            "same_phase_switch_action_delta": _distribution(switch_deltas),
        },
        "candidate_diversity": {
            "definition": (
                "mean unordered-pair Euclidean distance, then mean over valid bank phases"
            ),
            "action_policy_space": _bank_diversity(capture, "candidate_bank_policy_actions"),
            "dreamed_matching_feature_space": _bank_diversity(
                capture, "candidate_bank_dreamed_matching_features"
            ),
        },
        "candidate_matching_geometry": matching_geometry,
        "action_differences": differences,
    }


def run(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    with load_diagnostic_capture(input_path) as capture:
        result = analyze_capture(capture)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.capture, args.output)


if __name__ == "__main__":
    main()

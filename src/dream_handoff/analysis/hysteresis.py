"""Compare memoryless selection with an absolute-hysteresis factual rollout."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dream_handoff.capture import DiagnosticCapture, load_diagnostic_capture

V1_ABSOLUTE_TAU = 0.027628261595964432


@dataclass(frozen=True)
class SelectionStatistics:
    switched: np.ndarray
    reversal_aba: np.ndarray
    rapid_successive_switch_within_3_rows: np.ndarray
    dwell_lengths: np.ndarray
    switch_count: int
    transitions: int


def _control_rows(capture: DiagnosticCapture) -> np.ndarray:
    step = capture.array("control_step").astype(np.int64, copy=False)
    episode = capture.array("episode_id").astype(np.int64, copy=False)
    keep = np.ones(len(step), dtype=bool)
    if len(step) > 1:
        keep[:-1] = (episode[:-1] != episode[1:]) | (step[:-1] != step[1:])
    recorded = getattr(capture, "recorded_episode_ids", tuple(np.unique(episode)))
    keep &= np.isin(episode, recorded)
    return np.flatnonzero(keep)


def replay_absolute_hysteresis(
    distances: np.ndarray,
    bank_id: np.ndarray,
    *,
    threshold: float = V1_ABSOLUTE_TAU,
    bank_first_index: np.ndarray,
) -> np.ndarray:
    """Replay ``d(incumbent)-d(nearest) > tau`` with bank-local state."""
    distances = np.asarray(distances, dtype=np.float64)
    bank_id = np.asarray(bank_id)
    first = np.asarray(bank_first_index, dtype=np.int64)
    if distances.ndim != 2 or bank_id.shape != (len(distances),) or first.shape != bank_id.shape:
        raise ValueError("distances, bank_id, and bank_first_index have incompatible shapes")
    selected = np.empty(len(distances), dtype=np.int64)
    previous_bank: Any = None
    incumbent = -1
    for index, current_bank in enumerate(bank_id):
        nearest = int(np.argmin(distances[index]))
        if current_bank != previous_bank:
            incumbent = int(first[index])
            if not 0 <= incumbent < distances.shape[1]:
                raise ValueError("every bank-first row needs a valid recorded phase-0 index")
            previous_bank = current_bank
        elif float(distances[index, incumbent] - distances[index, nearest]) > threshold:
            incumbent = nearest
        selected[index] = incumbent
    return selected


def selection_statistics(selected: np.ndarray, bank_id: np.ndarray) -> SelectionStatistics:
    selected = np.asarray(selected, dtype=np.int64)
    bank_id = np.asarray(bank_id)
    switched = np.zeros(len(selected), dtype=bool)
    reversal = np.zeros(len(selected), dtype=bool)
    rapid = np.zeros(len(selected), dtype=bool)
    dwell_lengths: list[int] = []
    dwell = 0
    last_switch: int | None = None
    previous_bank: Any = None
    for index, current_bank in enumerate(bank_id):
        if current_bank != previous_bank:
            if dwell:
                dwell_lengths.append(dwell)
            dwell = 1
            last_switch = None
            previous_bank = current_bank
            continue
        did_switch = selected[index] != selected[index - 1]
        switched[index] = did_switch
        if did_switch:
            dwell_lengths.append(dwell)
            dwell = 1
            if (
                index >= 2
                and bank_id[index - 2] == current_bank
                and switched[index - 1]
                and selected[index] == selected[index - 2]
            ):
                reversal[index] = True
            if last_switch is not None and index - last_switch <= 3:
                rapid[index] = True
            last_switch = index
        else:
            dwell += 1
    if dwell:
        dwell_lengths.append(dwell)
    transitions = int(np.count_nonzero(bank_id[1:] == bank_id[:-1]))
    return SelectionStatistics(
        switched=switched,
        reversal_aba=reversal,
        rapid_successive_switch_within_3_rows=rapid,
        dwell_lengths=np.asarray(dwell_lengths, dtype=np.int64),
        switch_count=int(switched.sum()),
        transitions=transitions,
    )


def _summary(
    selected: np.ndarray,
    distances: np.ndarray,
    bank_id: np.ndarray,
    phase: np.ndarray,
) -> dict[str, Any]:
    stats = selection_statistics(selected, bank_id)
    nearest = np.min(distances, axis=1)
    selected_distance = distances[np.arange(len(selected)), selected]
    regret = selected_distance - nearest
    operational = phase > 0
    operational_stats = selection_statistics(selected[operational], bank_id[operational])
    dwell = stats.dwell_lengths.astype(np.float64)
    return {
        "switch_count": stats.switch_count,
        "switch_rate": stats.switch_count / stats.transitions,
        "operational_switch_count": operational_stats.switch_count,
        "operational_switch_rate": (operational_stats.switch_count / operational_stats.transitions),
        "reversal_aba_count": int(stats.reversal_aba.sum()),
        "rapid_successive_switch_within_3_rows_count": int(
            stats.rapid_successive_switch_within_3_rows.sum()
        ),
        "mean_matching_cost": float(np.mean(selected_distance[operational])),
        "mean_regret": float(np.mean(regret[operational])),
        "top1_match_rate": float(np.mean(regret[operational] == 0.0)),
        "dwell_ticks": {
            "runs": int(len(dwell)),
            "mean": float(np.mean(dwell)),
            "median": float(np.median(dwell)),
            "p90": float(np.percentile(dwell, 90)),
        },
    }


def analyze_capture(
    capture: DiagnosticCapture,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    if threshold is None:
        threshold = float(capture.metadata["controller"]["hysteresis_tau"])
    rows = _control_rows(capture)
    episode = capture.array("episode_id")[rows].astype(np.int64, copy=False)
    origin = capture.array("bank_origin_action_count")[rows].astype(np.int64, copy=False)
    phase = capture.array("phase")[rows].astype(np.int64, copy=False)
    distances = capture.array("candidate_distances")[rows].astype(np.float64, copy=False)
    factual = capture.array("selected_candidate_index")[rows].astype(np.int64, copy=False)
    identities = list(zip(episode.tolist(), origin.tolist(), strict=True))
    ids: dict[tuple[int, int], int] = {}
    bank_id = np.asarray([ids.setdefault(identity, len(ids)) for identity in identities])
    bank_first = np.full(len(rows), -1, dtype=np.int64)
    starts = np.r_[True, bank_id[1:] != bank_id[:-1]]
    bank_first[starts] = factual[starts]

    baseline = np.argmin(distances, axis=1).astype(np.int64)
    baseline[starts] = factual[starts]
    hysteresis = replay_absolute_hysteresis(
        distances,
        bank_id,
        threshold=threshold,
        bank_first_index=bank_first,
    )
    agreement = float(np.mean(hysteresis == factual))
    if agreement != 1.0:
        raise ValueError(f"absolute-hysteresis replay disagrees with factual rows: {agreement:.6%}")
    return {
        "analysis": "selector stabilization with absolute hysteresis",
        "rule": "switch iff distance(incumbent) - distance(nearest) > tau",
        "tau": threshold,
        "rows": int(len(rows)),
        "banks": int(len(ids)),
        "phase_zero_seed": "recorded factual seeded-uniform selection",
        "memoryless_nearest": _summary(baseline, distances, bank_id, phase),
        "absolute_hysteresis": _summary(hysteresis, distances, bank_id, phase),
        "factual_selection_match_rate": agreement,
    }


def run(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    with load_diagnostic_capture(input_path) as capture:
        result = analyze_capture(capture, threshold=threshold)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tau",
        type=float,
        help="override the factual capture threshold (defaults to capture metadata)",
    )
    args = parser.parse_args()
    run(args.capture, args.output, threshold=args.tau)


if __name__ == "__main__":
    main()

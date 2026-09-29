"""Test mixed handoff paths and their effect on R2 activation prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dream_handoff.analysis.statistics import distribution_summary, episode_clustered_bootstrap
from dream_handoff.capture import DiagnosticCapture, load_diagnostic_capture
from dream_handoff.r2dreamer.model import R2State
from dream_handoff.r2dreamer.runtime import R2DreamerRuntime

STRATA = ("all", "0", "1", ">=2")


@dataclass(frozen=True)
class HandoffEvent:
    episode_id: int
    request_id: int
    generation: int
    request_action_count: int
    activation_action_count: int
    request_phase: int
    old_bank_index: int
    old_bank_horizon: int
    new_bank_index: int
    new_bank_horizon: int
    static_prefixes: np.ndarray
    static_policy_prefixes: np.ndarray
    gt_prefix: np.ndarray
    gt_policy_prefix: np.ndarray
    selected_candidate_sequence: np.ndarray
    n_switches: int
    request_stoch: np.ndarray
    request_deter: np.ndarray
    activation_stoch: np.ndarray
    activation_deter: np.ndarray
    activation_proprioception: np.ndarray
    activation_selected_index: int
    predicted_inference_delay: int
    request_index: int

    @property
    def latency_length(self) -> int:
        return self.activation_action_count - self.request_action_count


def switch_stratum(switches: int) -> str:
    if switches < 0:
        raise ValueError("switch count must be non-negative")
    if switches == 0:
        return "0"
    if switches == 1:
        return "1"
    return ">=2"


def _control_rows(capture: DiagnosticCapture) -> np.ndarray:
    step = capture.array("control_step").astype(np.int64, copy=False)
    episode = capture.array("episode_id").astype(np.int64, copy=False)
    keep = np.ones(len(step), dtype=bool)
    if len(step) > 1:
        keep[:-1] = (episode[:-1] != episode[1:]) | (step[:-1] != step[1:])
    recorded = getattr(capture, "recorded_episode_ids", tuple(np.unique(episode)))
    keep &= np.isin(episode, recorded)
    return np.flatnonzero(keep)


def reconstruct_events(
    capture: DiagnosticCapture,
) -> tuple[tuple[HandoffEvent, ...], dict[str, int]]:
    """Recover final activation-anchored events through the format-16 reader."""
    rows = _control_rows(capture)
    episode = capture.array("episode_id").astype(np.int64, copy=False)
    step = capture.array("control_step").astype(np.int64, copy=False)
    row_by_step = {(int(episode[row]), int(step[row])): int(row) for row in rows}
    bank_episode = capture.array("candidate_bank_episode_id").astype(np.int64, copy=False)
    bank_origin = capture.array("candidate_bank_origin_action_count").astype(np.int64, copy=False)
    bank_horizon = capture.array("candidate_bank_horizon").astype(np.int64, copy=False)
    banks = capture.array("candidate_bank_control_actions")
    policy_banks = capture.array("candidate_bank_policy_actions")
    bank_by_key = {
        (int(ep), int(origin)): index
        for index, (ep, origin) in enumerate(zip(bank_episode, bank_origin, strict=True))
    }
    row_origin = capture.array("bank_origin_action_count").astype(np.int64, copy=False)
    phase = capture.array("phase").astype(np.int64, copy=False)
    selected = capture.array("selected_candidate_index").astype(np.int64, copy=False)
    switched = capture.array("candidate_switched").astype(bool, copy=False)
    actions = capture.array("final_executed_action")
    policy_actions = capture.array("final_executed_policy_action")
    commands = capture.array("command_sent_to_robot")
    measured = capture.array("measured_joint_position")
    stoch = capture.array("live_world_model_stoch")
    deter = capture.array("live_world_model_deter")
    activation_phase = capture.array("async_request_activation_phase").astype(np.int64, copy=False)

    categories = (
        "stale_result",
        "initial_inline_build",
        "unactivated_request",
        "nonpositive_window",
        "activation_phase_field_mismatch",
        "incomplete_control_coverage",
        "activation_row_not_phase_zero",
        "old_bank_changed_in_window",
        "missing_old_candidate_bank",
        "missing_new_candidate_bank",
        "invalid_candidate_bank",
        "phase_horizon_overflow",
        "padded_or_nonfinite_bank_slice",
        "missing_request_origin_state",
        "missing_activation_state",
        "invalid_gt_prefix",
        "selector_provenance_mismatch",
        "command_provenance_mismatch",
        "selector_switch_flag_mismatch",
    )
    excluded = dict.fromkeys(categories, 0)
    events: list[HandoffEvent] = []
    all_requests = capture.requests()
    recorded_episode_ids = set(getattr(capture, "recorded_episode_ids", tuple(np.unique(episode))))
    requests = tuple(
        request for request in all_requests if request.episode_id in recorded_episode_ids
    )
    for request in requests:
        index = request.index
        origin = request.origin_action_count
        activation = request.activation_action_count
        length = activation - origin
        if request.stale:
            excluded["stale_result"] += 1
            continue
        if activation == origin and int(activation_phase[index]) == 0:
            excluded["initial_inline_build"] += 1
            continue
        if activation < 0:
            excluded["unactivated_request"] += 1
            continue
        if length <= 0:
            excluded["nonpositive_window"] += 1
            continue
        if int(activation_phase[index]) != length:
            excluded["activation_phase_field_mismatch"] += 1
            continue
        identities = [(request.episode_id, value) for value in range(origin, activation + 1)]
        if any(identity not in row_by_step for identity in identities):
            excluded["incomplete_control_coverage"] += 1
            continue
        event_rows = np.asarray([row_by_step[identity] for identity in identities])
        prefix_rows = event_rows[:-1]
        request_row = int(prefix_rows[0])
        activation_row = int(event_rows[-1])
        if int(row_origin[activation_row]) != activation or int(phase[activation_row]) != 0:
            excluded["activation_row_not_phase_zero"] += 1
            continue
        old_origin = int(row_origin[request_row])
        request_phase = int(phase[request_row])
        if (
            old_origin == activation
            or not np.all(row_origin[prefix_rows] == old_origin)
            or not np.array_equal(phase[prefix_rows], request_phase + np.arange(length))
        ):
            excluded["old_bank_changed_in_window"] += 1
            continue
        old_key = (request.episode_id, old_origin)
        new_key = (request.episode_id, activation)
        if old_key not in bank_by_key:
            excluded["missing_old_candidate_bank"] += 1
            continue
        if new_key not in bank_by_key:
            excluded["missing_new_candidate_bank"] += 1
            continue
        old_index = bank_by_key[old_key]
        if banks.ndim != 4 or banks.shape[1] == 0 or int(bank_horizon[old_index]) <= 0:
            excluded["invalid_candidate_bank"] += 1
            continue
        if request_phase < 0 or request_phase + length > int(bank_horizon[old_index]):
            excluded["phase_horizon_overflow"] += 1
            continue
        static = banks[old_index, :, request_phase : request_phase + length].copy()
        static_policy = policy_banks[old_index, :, request_phase : request_phase + length].copy()
        if not np.isfinite(static).all() or not np.isfinite(static_policy).all():
            excluded["padded_or_nonfinite_bank_slice"] += 1
            continue
        request_stoch, request_deter = stoch[request_row], deter[request_row]
        activation_stoch, activation_deter = stoch[activation_row], deter[activation_row]
        if not (
            request_stoch.size
            and request_deter.size
            and np.isfinite(request_stoch).all()
            and np.isfinite(request_deter).all()
        ):
            excluded["missing_request_origin_state"] += 1
            continue
        if not (
            activation_stoch.size
            and activation_deter.size
            and np.isfinite(activation_stoch).all()
            and np.isfinite(activation_deter).all()
        ):
            excluded["missing_activation_state"] += 1
            continue
        gt = actions[prefix_rows].copy()
        gt_policy = policy_actions[prefix_rows].copy()
        candidate_sequence = selected[prefix_rows]
        if (
            gt.shape != static.shape[1:]
            or gt_policy.shape != static_policy.shape[1:]
            or not np.isfinite(gt).all()
            or not np.isfinite(gt_policy).all()
        ):
            excluded["invalid_gt_prefix"] += 1
            continue
        if np.any(candidate_sequence < 0) or np.any(candidate_sequence >= static.shape[0]):
            excluded["invalid_gt_prefix"] += 1
            continue
        if not np.array_equal(
            static[candidate_sequence, np.arange(length)], gt
        ) or not np.array_equal(static_policy[candidate_sequence, np.arange(length)], gt_policy):
            excluded["selector_provenance_mismatch"] += 1
            continue
        if not np.array_equal(commands[prefix_rows], gt):
            excluded["command_provenance_mismatch"] += 1
            continue
        expected_switch = np.zeros(length, dtype=bool)
        for offset, row in enumerate(prefix_rows):
            previous = row_by_step.get((request.episode_id, int(step[row]) - 1))
            if previous is not None and int(row_origin[previous]) == old_origin:
                expected_switch[offset] = selected[previous] != candidate_sequence[offset]
        if not np.array_equal(switched[prefix_rows], expected_switch):
            excluded["selector_switch_flag_mismatch"] += 1
            continue
        events.append(
            HandoffEvent(
                episode_id=request.episode_id,
                request_id=request.request_id,
                generation=request.generation,
                request_action_count=origin,
                activation_action_count=activation,
                request_phase=request_phase,
                old_bank_index=old_index,
                old_bank_horizon=int(bank_horizon[old_index]),
                new_bank_index=bank_by_key[new_key],
                new_bank_horizon=int(bank_horizon[bank_by_key[new_key]]),
                static_prefixes=static,
                static_policy_prefixes=static_policy,
                gt_prefix=gt,
                gt_policy_prefix=gt_policy,
                selected_candidate_sequence=candidate_sequence.copy(),
                n_switches=int(expected_switch.sum()),
                request_stoch=request_stoch.copy(),
                request_deter=request_deter.copy(),
                activation_stoch=activation_stoch.copy(),
                activation_deter=activation_deter.copy(),
                activation_proprioception=measured[activation_row].copy(),
                activation_selected_index=int(selected[activation_row]),
                predicted_inference_delay=request.predicted_delay,
                request_index=request.index,
            )
        )
    counts = {
        "capture_requests_total": len(all_requests),
        "requests_total": len(requests),
        "excluded_discarded_rerecord_requests": len(all_requests) - len(requests),
        **{f"excluded_{key}": value for key, value in excluded.items()},
        "included_events": len(events),
    }
    counts["excluded_total"] = len(requests) - len(events)
    if counts["excluded_total"] != sum(excluded.values()):
        raise RuntimeError("exclusive request accounting does not sum")
    return tuple(events), counts


def action_path_metrics(static_prefixes: np.ndarray, gt_prefix: np.ndarray) -> dict[str, Any]:
    static = np.asarray(static_prefixes, dtype=np.float64)
    gt = np.asarray(gt_prefix, dtype=np.float64)
    if static.ndim != 3 or gt.ndim != 2 or static.shape[1:] != gt.shape:
        raise ValueError("expected compatible [N,L,A] and [L,A] prefixes")
    difference = static - gt[None]
    path_distances = np.sqrt(np.mean(np.sum(difference**2, axis=2), axis=1))
    path_best = int(np.argmin(path_distances))
    endpoint_distances = np.linalg.norm(static[:, -1] - gt[-1], axis=1)
    if len(gt) < 2:
        delta_error = None
    else:
        delta = np.diff(static, axis=1) - np.diff(gt, axis=0)[None]
        delta_error = float(np.min(np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=1))))
    matches = np.flatnonzero(np.max(np.abs(difference), axis=(1, 2)) <= 1e-6)
    return {
        "endpoint_error": float(np.min(endpoint_distances)),
        "path_error": float(path_distances[path_best]),
        "path_best_candidate": path_best,
        "delta_path_error": delta_error,
        "path_matching_candidates": matches.tolist(),
    }


def _state(event_stoch: np.ndarray, event_deter: np.ndarray, device: torch.device) -> R2State:
    return R2State(
        torch.as_tensor(event_stoch, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(event_deter, dtype=torch.float32, device=device).unsqueeze(0),
    )


def comparator_metrics(
    static_endpoint_errors: np.ndarray,
    exact_history_error: float,
    *,
    action_path_nearest_candidate: int,
) -> dict[str, float]:
    """Score target-selected and action-path-selected static comparators separately."""
    static = np.asarray(static_endpoint_errors, dtype=np.float64)
    if static.ndim != 1 or not static.size or not np.isfinite(static).all():
        raise ValueError("static endpoint errors must be a non-empty finite vector")
    if not np.isfinite(exact_history_error):
        raise ValueError("exact-history endpoint error must be finite")
    if not 0 <= action_path_nearest_candidate < len(static):
        raise ValueError("action-path-nearest candidate index is out of range")
    target_selected = float(np.min(static))
    action_path_nearest = float(static[action_path_nearest_candidate])
    return {
        "exact_history_wm_error": exact_history_error,
        "target_selected_best_of_n_static_wm_error": target_selected,
        "target_selected_best_of_n_static_minus_exact_history": (
            target_selected - exact_history_error
        ),
        "action_path_nearest_static_wm_error": action_path_nearest,
        "action_path_nearest_static_minus_exact_history": (
            action_path_nearest - exact_history_error
        ),
    }


def world_model_metrics(
    event: HandoffEvent,
    action: dict[str, Any],
    runtime: R2DreamerRuntime,
) -> dict[str, float]:
    device = runtime.device
    request_state = _state(event.request_stoch, event.request_deter, device)
    activation_state = _state(event.activation_stoch, event.activation_deter, device)
    static_actions = torch.as_tensor(event.static_prefixes, dtype=torch.float32, device=device)
    gt_actions = torch.as_tensor(event.gt_prefix[None], dtype=torch.float32, device=device)
    with torch.inference_mode():
        static_state = runtime.imagine_final_state(request_state, static_actions)
        gt_state = runtime.imagine_final_state(request_state, gt_actions)
        real_feature = runtime.matching_feature(activation_state)
        static_errors = torch.linalg.vector_norm(
            runtime.matching_feature(static_state) - real_feature, dim=-1
        )
        gt_error = torch.linalg.vector_norm(
            runtime.matching_feature(gt_state) - real_feature, dim=-1
        )
    static_np = static_errors.detach().float().cpu().numpy()
    gt_value = float(gt_error.item())
    return comparator_metrics(
        static_np,
        gt_value,
        action_path_nearest_candidate=int(action["path_best_candidate"]),
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "endpoint_error",
        "path_error",
        "delta_path_error",
        "exact_history_wm_error",
        "target_selected_best_of_n_static_wm_error",
        "target_selected_best_of_n_static_minus_exact_history",
        "action_path_nearest_static_wm_error",
        "action_path_nearest_static_minus_exact_history",
    )
    result: dict[str, Any] = {}
    for stratum in STRATA:
        selected = [row for row in rows if stratum == "all" or row["switch_stratum"] == stratum]
        result[stratum] = {"events": len(selected)}
        for metric in metrics:
            result[stratum][metric] = distribution_summary([row[metric] for row in selected])
    return result


def analyze_capture(
    capture: DiagnosticCapture,
    *,
    runtime: R2DreamerRuntime,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    events, inclusion = reconstruct_events(capture)
    rows: list[dict[str, Any]] = []
    for event in events:
        action = action_path_metrics(event.static_prefixes, event.gt_prefix)
        row = {
            "episode_id": event.episode_id,
            "request_id": event.request_id,
            "n_switches": event.n_switches,
            "switch_stratum": switch_stratum(event.n_switches),
            **action,
        }
        row.update(world_model_metrics(event, action, runtime))
        rows.append(row)
    if not all(row["endpoint_error"] <= 1e-6 for row in rows):
        raise ValueError("handoff endpoint alignment gate failed")
    numerical = _summary(rows)
    bootstrap = {
        "target_selected_best_of_n_static_minus_exact_history": {},
        "action_path_nearest_static_minus_exact_history": {},
    }
    for stratum in STRATA:
        selected = [row for row in rows if stratum == "all" or row["switch_stratum"] == stratum]
        for metric in bootstrap:
            bootstrap[metric][stratum] = episode_clustered_bootstrap(
                selected,
                value_key=metric,
                num_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
    return {
        "analysis": "handoff prediction and grounding error",
        "event_inclusion": inclusion,
        "switch_counts": {
            "0": sum(row["switch_stratum"] == "0" for row in rows),
            "1": sum(row["switch_stratum"] == "1" for row in rows),
            ">=2": sum(row["switch_stratum"] == ">=2" for row in rows),
        },
        "metric_definitions": {
            "interval": "[request_action_count, activation_action_count)",
            "path_error": "min_j sqrt(mean_time(sum_action_dim((static[j]-factual)^2)))",
            "target_selected_best_of_n": (
                "choose the static imagined endpoint with minimum activation-posterior target error"
            ),
            "action_path_nearest": (
                "choose argmin_j sqrt(mean_time(sum_action_dim((static[j]-factual)^2))) "
                "before comparing imagined endpoint errors"
            ),
            "exact_history_wm_error": (
                "exact executed-history imagined endpoint to activation posterior feature"
            ),
            "advantage_sign": (
                "selected_static_error - exact_history_error; positive means the exact-history "
                "endpoint has lower activation-target error"
            ),
        },
        "numerical_summary": numerical,
        "episode_clustered_bootstrap": {
            "settings": {
                "unit": "physical episode cluster",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
            },
            **bootstrap,
        },
        "zero_switch_sanity": (
            "For a zero-switch event, the factual exact-history path is one of the static "
            "candidate continuations. Target selection takes the minimum target error over "
            "all static candidates, so its error is structurally no greater than the factual "
            "candidate's error apart from numerical effects. A negative target-selected "
            "difference in this stratum is not evidence that exact history is harmful."
        ),
        "interpretation": (
            "Exact factual action history does not eliminate the activation-posterior grounding "
            "gap. Whether its endpoint error is lower or higher than a static counterfactual "
            "depends on how that static comparator is selected."
        ),
    }


def run(
    capture_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    *,
    device: str = "cuda",
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    runtime = R2DreamerRuntime.from_checkpoint(checkpoint_path, device=device)
    with load_diagnostic_capture(capture_path) as capture:
        result = analyze_capture(
            capture,
            runtime=runtime,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    run(
        args.capture,
        args.checkpoint,
        args.output,
        device=args.device,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()

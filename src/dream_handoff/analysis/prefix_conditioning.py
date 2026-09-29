"""Evaluate direct prefix-conditioned asynchronous next-bank generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dream_handoff.analysis.handoff_prediction import (
    HandoffEvent,
    reconstruct_events,
    switch_stratum,
)
from dream_handoff.analysis.statistics import distribution_summary, episode_clustered_bootstrap
from dream_handoff.capture import DiagnosticCapture, load_diagnostic_capture
from dream_handoff.inference.sampling import (
    SmolVLACandidateSampler,
    configure_rtc_prefix_guidance,
)

CONDITIONS = ("plain", "static", "static_live", "gt")
REFERENCE_CONDITION = "reactive"
PAIRED_COMPARISONS = (
    ("static", "plain"),
    ("gt", "plain"),
    ("gt", "static"),
    ("static_live", "plain"),
    ("gt", "static_live"),
    ("static_live", "static"),
)
PRIMARY_PAIR_METRICS = (
    "reactive_best_control",
    "reactive_median_candidate_control",
    "reactive_paired_same_noise_control",
    "command_seam",
    "state_seam",
    "velocity_seam",
)
CONDITION_METRICS = (
    "prefix_adherence_mean",
    "command_seam",
    "command_seam_factual_draw",
    "command_seam_min",
    "command_seam_max",
    "state_seam",
    "velocity_seam",
    "reactive_best_control",
    "reactive_median_candidate_control",
    "reactive_best_policy",
    "reactive_paired_same_noise_control",
    "reactive_paired_same_noise_policy",
    "diversity_control",
    "diversity_policy",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_schedule(
    condition: str,
    *,
    latency_length: int,
    guidance_horizon: int,
    predicted_delay: int,
) -> dict[str, int | None]:
    """Return the four final schedules; deliberately has no synthetic GT-live arm."""
    if condition in {"plain", "reactive"}:
        return {"prefix_length": None, "inference_delay": None, "execution_horizon": None}
    if condition in {"static", "gt"}:
        return {
            "prefix_length": latency_length,
            "inference_delay": latency_length,
            "execution_horizon": latency_length,
        }
    if condition == "static_live":
        return {
            "prefix_length": guidance_horizon,
            "inference_delay": predicted_delay,
            "execution_horizon": guidance_horizon,
        }
    raise ValueError(f"unknown final prefix condition {condition!r}")


def trajectory_distance_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        raise ValueError(f"trajectory banks do not align: {left.shape} and {right.shape}")
    return np.sqrt(np.mean((left[:, None] - right[None]) ** 2, axis=(2, 3)))


def prefix_adherence(generated: np.ndarray, prefixes: np.ndarray) -> np.ndarray:
    bank = np.asarray(generated, dtype=np.float64)
    target = np.asarray(prefixes, dtype=np.float64)
    if bank.ndim != 3 or target.ndim != 3 or bank.shape[0] != target.shape[0]:
        raise ValueError("generated bank and prefixes must align on candidate axis")
    if target.shape[1] <= 0 or target.shape[1] > bank.shape[1] or target.shape[2] != bank.shape[2]:
        raise ValueError("prefix shape is incompatible with generated bank")
    return np.sqrt(np.mean((bank[:, : target.shape[1]] - target) ** 2, axis=(1, 2)))


def bank_diversity(bank: np.ndarray) -> float:
    values = trajectory_distance_matrix(bank, bank)
    if len(values) < 2:
        return 0.0
    upper = np.triu_indices(len(values), k=1)
    return float(np.mean(values[upper]))


def array_replay_error(reference: np.ndarray, replayed: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(reference)
    actual = np.asarray(replayed)
    if expected.shape != actual.shape or expected.ndim != 3:
        raise ValueError(f"replay banks must share [N,H,A], got {expected.shape}/{actual.shape}")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "exact_match": bool(np.array_equal(expected, actual)),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def crop_like_engine(generated: np.ndarray, *, elapsed: int, horizon: int) -> np.ndarray:
    values = np.asarray(generated)
    if values.ndim != 3 or elapsed < 0 or horizon <= 0 or elapsed + horizon > values.shape[1]:
        raise ValueError("activation crop does not fit the generated bank")
    return values[:, elapsed : elapsed + horizon]


def _prefix_metrics(condition: str, generated: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    values = prefix_adherence(generated, targets)
    return {
        f"{condition}_prefix_adherence_mean": float(np.mean(values)),
        f"{condition}_prefix_adherence_median": float(np.median(values)),
        f"{condition}_prefix_adherence_max": float(np.max(values)),
        f"{condition}_prefix_adherence_per_candidate": values.tolist(),
    }


def _score_condition(
    condition: str,
    *,
    policy: np.ndarray,
    control: np.ndarray,
    event: HandoffEvent,
    reactive_policy_suffix: np.ndarray,
    reactive_control_suffix: np.ndarray,
) -> dict[str, Any]:
    length = event.latency_length
    if policy.shape != control.shape or policy.shape[1] <= length:
        raise ValueError(f"{condition} bank cannot be aligned at the factual activation")
    outgoing = event.gt_prefix[-1]
    incoming = np.asarray(control, dtype=np.float64)[:, length]
    command = np.linalg.norm(incoming - outgoing, axis=1)
    state = np.linalg.norm(incoming - event.activation_proprioception, axis=1)
    factual = event.activation_selected_index
    row: dict[str, Any] = {
        f"{condition}_command_seam": float(np.mean(command)),
        f"{condition}_command_seam_factual_draw": float(command[factual]),
        f"{condition}_command_seam_min": float(np.min(command)),
        f"{condition}_command_seam_max": float(np.max(command)),
        f"{condition}_state_seam": float(np.mean(state)),
    }
    if length < 2:
        row[f"{condition}_velocity_seam"] = None
    else:
        outgoing_velocity = event.gt_prefix[-1] - event.gt_prefix[-2]
        velocity = np.linalg.norm((incoming - outgoing) - outgoing_velocity, axis=1)
        row[f"{condition}_velocity_seam"] = float(np.mean(velocity))

    policy_suffix = policy[:, length:]
    control_suffix = control[:, length:]
    policy_distances = trajectory_distance_matrix(policy_suffix, reactive_policy_suffix)
    control_distances = trajectory_distance_matrix(control_suffix, reactive_control_suffix)
    candidate_error = np.min(control_distances, axis=1)
    row[f"{condition}_reactive_best_control"] = float(np.min(candidate_error))
    row[f"{condition}_reactive_median_candidate_control"] = float(np.median(candidate_error))
    row[f"{condition}_reactive_candidate_errors_control"] = candidate_error.tolist()
    row[f"{condition}_reactive_best_policy"] = float(np.min(policy_distances))
    row[f"{condition}_reactive_paired_same_noise_control"] = float(
        np.mean(np.diag(control_distances))
    )
    row[f"{condition}_reactive_paired_same_noise_policy"] = float(
        np.mean(np.diag(policy_distances))
    )
    row[f"{condition}_diversity_control"] = bank_diversity(control_suffix)
    row[f"{condition}_diversity_policy"] = bank_diversity(policy_suffix)
    return row


def _generated_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "episode_id",
            "request_id",
            "generation",
            *(
                f"{condition}_{space}_actions"
                for condition in (*CONDITIONS, "reactive")
                for space in ("policy", "control")
            ),
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"generated-bank artifact is missing {sorted(missing)}")
        return {name: archive[name] for name in archive.files}


def _paired(row: dict[str, Any]) -> None:
    for metric in PRIMARY_PAIR_METRICS:
        for first, second in PAIRED_COMPARISONS:
            left = row.get(f"{first}_{metric}")
            right = row.get(f"{second}_{metric}")
            row[f"{first}_minus_{second}_{metric}"] = (
                None if left is None or right is None else float(left - right)
            )


def _summarize(
    rows: list[dict[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in (*CONDITIONS, REFERENCE_CONDITION):
        block = {
            metric: distribution_summary(
                [row.get(f"{condition}_{metric}") for row in rows if f"{condition}_{metric}" in row]
            )
            for metric in CONDITION_METRICS
            if any(f"{condition}_{metric}" in row for row in rows)
        }
        prefix_values = [
            value
            for row in rows
            for value in row.get(f"{condition}_prefix_adherence_per_candidate", [])
        ]
        if prefix_values:
            block["prefix_adherence_all_candidates"] = distribution_summary(prefix_values)
        candidate_values = [
            value
            for row in rows
            for value in row.get(f"{condition}_reactive_candidate_errors_control", [])
        ]
        if candidate_values:
            block["reactive_candidate_errors_control"] = distribution_summary(candidate_values)
        by_condition[condition] = block
    paired: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for metric in PRIMARY_PAIR_METRICS:
        for first, second in PAIRED_COMPARISONS:
            key = f"{first}_minus_{second}_{metric}"
            paired[key] = distribution_summary([row.get(key) for row in rows])
            usable = [row for row in rows if row.get(key) is not None and np.isfinite(row[key])]
            bootstrap[key] = episode_clustered_bootstrap(
                usable,
                value_key=key,
                num_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
    return {
        "by_condition": by_condition,
        "paired_differences": paired,
        "episode_clustered_bootstrap": bootstrap,
    }


def analyze_generated_banks(
    capture: DiagnosticCapture,
    generated_path: str | Path,
    *,
    guidance_horizon: int = 15,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    events, inclusion = reconstruct_events(capture)
    generated = _generated_arrays(generated_path)
    identities = list(
        zip(
            generated["episode_id"].tolist(),
            generated["generation"].tolist(),
            generated["request_id"].tolist(),
            strict=True,
        )
    )
    event_by_identity = {
        (event.episode_id, event.generation, event.request_id): event for event in events
    }
    if set(identities) != set(event_by_identity):
        raise ValueError("generated banks and factual handoffs have different event identities")

    policy_banks = capture.array("candidate_bank_policy_actions")
    factual_control_banks = capture.array("candidate_bank_control_actions")
    factual_policy_banks = capture.array("candidate_bank_policy_actions")
    rows: list[dict[str, Any]] = []
    gate_policy: list[dict[str, Any]] = []
    gate_control: list[dict[str, Any]] = []
    for index, identity in enumerate(identities):
        event = event_by_identity[identity]
        length = event.latency_length
        if event.request_phase + guidance_horizon > event.old_bank_horizon:
            raise ValueError("static-live prefix reaches beyond the factual outgoing bank")
        schedules = {
            condition: condition_schedule(
                condition,
                latency_length=length,
                guidance_horizon=guidance_horizon,
                predicted_delay=event.predicted_inference_delay,
            )
            for condition in (*CONDITIONS, REFERENCE_CONDITION)
        }
        if "gt_live" in schedules:
            raise AssertionError("synthetic GT-live must never be constructed")

        condition_banks = {
            condition: (
                generated[f"{condition}_policy_actions"][index],
                generated[f"{condition}_control_actions"][index],
            )
            for condition in (*CONDITIONS, REFERENCE_CONDITION)
        }
        plain_policy, plain_control = condition_banks["plain"]
        replay_policy = crop_like_engine(
            plain_policy, elapsed=length, horizon=event.new_bank_horizon
        )
        replay_control = crop_like_engine(
            plain_control, elapsed=length, horizon=event.new_bank_horizon
        )
        gate_policy.append(
            array_replay_error(
                factual_policy_banks[event.new_bank_index, :, : event.new_bank_horizon],
                replay_policy,
            )
        )
        gate_control.append(
            array_replay_error(
                factual_control_banks[event.new_bank_index, :, : event.new_bank_horizon],
                replay_control,
            )
        )

        reactive_policy, reactive_control = condition_banks["reactive"]
        reactive_policy_suffix = reactive_policy[:, :-length]
        reactive_control_suffix = reactive_control[:, :-length]
        static_live_prefix = policy_banks[
            event.old_bank_index,
            :,
            event.request_phase : event.request_phase + guidance_horizon,
        ]
        gt_prefix = np.repeat(event.gt_policy_prefix[None], len(event.static_prefixes), axis=0)
        row: dict[str, Any] = {
            "episode_id": event.episode_id,
            "request_id": event.request_id,
            "generation": event.generation,
            "L_e": length,
            "guidance_horizon": guidance_horizon,
            "predicted_inference_delay": event.predicted_inference_delay,
        }
        row.update(
            _prefix_metrics("static", condition_banks["static"][0], event.static_policy_prefixes)
        )
        row.update(
            _prefix_metrics("static_live", condition_banks["static_live"][0], static_live_prefix)
        )
        row.update(_prefix_metrics("gt", condition_banks["gt"][0], gt_prefix))
        for condition in CONDITIONS:
            policy, control = condition_banks[condition]
            row.update(
                _score_condition(
                    condition,
                    policy=policy,
                    control=control,
                    event=event,
                    reactive_policy_suffix=reactive_policy_suffix,
                    reactive_control_suffix=reactive_control_suffix,
                )
            )
        row["reactive_diversity_control"] = bank_diversity(reactive_control_suffix)
        row["reactive_diversity_policy"] = bank_diversity(reactive_policy_suffix)
        _paired(row)
        rows.append(row)

    policy_max = max(item["max_abs"] for item in gate_policy)
    control_max = max(item["max_abs"] for item in gate_control)
    gate = {
        "events": len(rows),
        "passed": policy_max <= 1e-5 and control_max <= 1e-5,
        "max_abs_tolerance": 1e-5,
        "policy_exact_match_rate": float(np.mean([item["exact_match"] for item in gate_policy])),
        "control_exact_match_rate": float(np.mean([item["exact_match"] for item in gate_control])),
        "policy_max_abs": policy_max,
        "control_max_abs": control_max,
    }
    if not gate["passed"]:
        raise ValueError(f"plain paired-noise replay gate failed: {gate}")
    numerical = _summarize(
        rows, bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed
    )
    command_static = -numerical["paired_differences"]["static_minus_plain_command_seam"]["mean"]
    command_gt = -numerical["paired_differences"]["gt_minus_plain_command_seam"]["mean"]
    velocity_static = -numerical["paired_differences"]["static_minus_plain_velocity_seam"]["mean"]
    velocity_gt = -numerical["paired_differences"]["gt_minus_plain_velocity_seam"]["mean"]
    return {
        "analysis": "prefix-conditioned asynchronous handoffs",
        "event_inclusion": {
            **inclusion,
            "handoff_events": inclusion["included_events"],
            "excluded_causal_horizon_overflow": 0,
            "excluded_missing_sparse_observation": 0,
            "excluded_invalid_sparse_identity": 0,
        },
        "conditions": {
            "plain": "unguided request-time generation",
            "static": "outgoing-candidate prefix; L=inference_delay=execution_horizon=L_e",
            "gt": "exact factual actions; L=inference_delay=execution_horizon=L_e",
            "static_live": (
                "outgoing-candidate prefix length 15; recorded latency prediction; "
                "execution_horizon 15"
            ),
            "reactive": "activation-time SmolVLA reference; not an optimal-controller oracle",
        },
        "gate_0": gate,
        "numerical_summary": numerical,
        "matched_L_e_static_fraction_of_exact_prefix_benefit": {
            "command_seam": command_static / command_gt,
            "velocity_seam": velocity_static / velocity_gt,
        },
        "statistics": {
            "unit": "paired async event",
            "bootstrap_unit": "physical episode cluster",
            "episode_clusters": len({event.episode_id for event in events}),
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
        },
        "interpretation": {
            "controlled_prefix_information": "GT versus Static under matched L_e scheduling",
            "deployable": "Static-live versus Plain",
            "scheduling_caveat": (
                "Static-live uses different scheduling and taper semantics, so it is not a "
                "causal ordered comparison with GT. Its state-seam result does not support a "
                "generic state-continuity claim."
            ),
        },
    }


def run_real_policy_replay_gate(
    capture_path: str | Path,
    policy_path: str | Path,
    *,
    device: str = "cuda",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Regenerate every factual plain bank from sparse observations and stored noise."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy = SmolVLAPolicy.from_pretrained(str(policy_path)).to(target).eval()
    if int(getattr(policy.config, "n_obs_steps", 1)) != 1:
        raise ValueError("sparse independent replay requires SmolVLA n_obs_steps=1")
    configure_rtc_prefix_guidance(policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(target)}},
    )
    sampler = SmolVLACandidateSampler(
        policy,
        preprocessor,
        postprocessor,
        device=str(target),
    )
    with load_diagnostic_capture(capture_path) as capture:
        capture.require_exact_replay()
        events, _ = reconstruct_events(capture)
        generation_noise = capture.array("async_request_generation_noise")
        factual_policy = capture.array("candidate_bank_policy_actions")
        factual_control = capture.array("candidate_bank_control_actions")
        sparse_meta = capture.metadata["sparse_exact_observations"]
        policy_errors: list[dict[str, Any]] = []
        control_errors: list[dict[str, Any]] = []
        for event in events:
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            for processor in (preprocessor, postprocessor):
                reset_processor = getattr(processor, "reset", None)
                if callable(reset_processor):
                    reset_processor()
            raw = capture.sparse_observation(
                episode_id=event.episode_id,
                generation=event.generation,
                request_id=event.request_id,
                event_role="request_origin",
            )
            prepared = prepare_observation_for_inference(
                raw,
                target,
                str(sparse_meta.get("task", "")),
                str(sparse_meta.get("robot_type", "")),
            )
            noise = torch.as_tensor(
                generation_noise[event.request_index],
                dtype=torch.float32,
                device=target,
            )
            replay_policy, replay_control = sampler.sample_candidate_bank(
                prepared,
                int(noise.shape[0]),
                noise=noise,
            )
            replay_policy_np = crop_like_engine(
                replay_policy.detach().float().cpu().numpy(),
                elapsed=event.latency_length,
                horizon=event.new_bank_horizon,
            )
            replay_control_np = crop_like_engine(
                replay_control.detach().float().cpu().numpy(),
                elapsed=event.latency_length,
                horizon=event.new_bank_horizon,
            )
            policy_errors.append(
                array_replay_error(
                    factual_policy[event.new_bank_index, :, : event.new_bank_horizon],
                    replay_policy_np,
                )
            )
            control_errors.append(
                array_replay_error(
                    factual_control[event.new_bank_index, :, : event.new_bank_horizon],
                    replay_control_np,
                )
            )
    policy_max = max(error["max_abs"] for error in policy_errors)
    control_max = max(error["max_abs"] for error in control_errors)
    result = {
        "name": "exact sparse-observation and paired-noise SmolVLA replay",
        "events": len(policy_errors),
        "max_abs_tolerance": 1e-5,
        "policy_exact_match_rate": float(
            np.mean([error["exact_match"] for error in policy_errors])
        ),
        "control_exact_match_rate": float(
            np.mean([error["exact_match"] for error in control_errors])
        ),
        "policy_max_abs": policy_max,
        "control_max_abs": control_max,
        "passed": policy_max <= 1e-5 and control_max <= 1e-5,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise ValueError(f"real-policy paired-noise replay failed: {result}")
    return result


def generate_real_policy_banks(
    capture_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cuda",
    guidance_horizon: int = 15,
) -> dict[str, Any]:
    """Generate every final prefix condition with paired factual SmolVLA noise."""

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    if guidance_horizon != 15:
        raise ValueError("the final Static-live condition requires guidance_horizon=15")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy_root = Path(policy_path)
    policy = SmolVLAPolicy.from_pretrained(str(policy_root)).to(target).eval()
    if int(getattr(policy.config, "n_obs_steps", 1)) != 1:
        raise ValueError("sparse independent replay requires SmolVLA n_obs_steps=1")
    configure_rtc_prefix_guidance(policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_root),
        preprocessor_overrides={"device_processor": {"device": str(target)}},
    )
    sampler = SmolVLACandidateSampler(
        policy,
        preprocessor,
        postprocessor,
        device=str(target),
    )
    conditions = (*CONDITIONS, REFERENCE_CONDITION)
    generated: dict[str, list[np.ndarray]] = {
        f"{condition}_{space}_actions": []
        for condition in conditions
        for space in ("policy", "control")
    }
    identities: dict[str, list[int]] = {
        "episode_id": [],
        "request_id": [],
        "generation": [],
    }
    with load_diagnostic_capture(capture_path) as capture:
        capture.require_exact_replay()
        events, _ = reconstruct_events(capture)
        generation_noise = capture.array("async_request_generation_noise")
        policy_banks = capture.array("candidate_bank_policy_actions")
        sparse_meta = capture.metadata["sparse_exact_observations"]
        for event_number, event in enumerate(events, start=1):
            static_live_prefix = policy_banks[
                event.old_bank_index,
                :,
                event.request_phase : event.request_phase + guidance_horizon,
            ]
            if (
                static_live_prefix.shape[1] != guidance_horizon
                or not np.isfinite(static_live_prefix).all()
            ):
                raise ValueError("Static-live prefix is incomplete")
            gt_prefix = np.repeat(
                event.gt_policy_prefix[None], len(event.static_policy_prefixes), axis=0
            )
            schedules = {
                "plain": ("request_origin", None, None, None),
                "static": (
                    "request_origin",
                    event.static_policy_prefixes,
                    event.latency_length,
                    event.latency_length,
                ),
                "gt": (
                    "request_origin",
                    gt_prefix,
                    event.latency_length,
                    event.latency_length,
                ),
                "static_live": (
                    "request_origin",
                    static_live_prefix,
                    event.predicted_inference_delay,
                    guidance_horizon,
                ),
                "reactive": ("activation", None, None, None),
            }
            noise = torch.as_tensor(
                generation_noise[event.request_index], dtype=torch.float32, device=target
            )
            for condition, (role, prefix, inference_delay, execution_horizon) in schedules.items():
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
                for processor in (preprocessor, postprocessor):
                    reset_processor = getattr(processor, "reset", None)
                    if callable(reset_processor):
                        reset_processor()
                raw = capture.sparse_observation(
                    episode_id=event.episode_id,
                    generation=event.generation,
                    request_id=event.request_id,
                    event_role=role,
                )
                prepared = prepare_observation_for_inference(
                    raw,
                    target,
                    str(sparse_meta.get("task", "")),
                    str(sparse_meta.get("robot_type", "")),
                )
                prefix_tensor = (
                    None
                    if prefix is None
                    else torch.as_tensor(prefix, dtype=torch.float32, device=target)
                )
                policy_actions, control_actions = sampler.sample_candidate_bank(
                    prepared,
                    int(noise.shape[0]),
                    noise=noise,
                    prefix_actions=prefix_tensor,
                    inference_delay=inference_delay,
                    execution_horizon=execution_horizon,
                )
                generated[f"{condition}_policy_actions"].append(
                    policy_actions.detach().float().cpu().numpy()
                )
                generated[f"{condition}_control_actions"].append(
                    control_actions.detach().float().cpu().numpy()
                )
            identities["episode_id"].append(event.episode_id)
            identities["request_id"].append(event.request_id)
            identities["generation"].append(event.generation)
            if event_number % 10 == 0 or event_number == len(events):
                print(f"generated prefix conditions for {event_number}/{len(events)} events")

    model_path = policy_root / "model.safetensors"
    metadata = {
        "schema_version": 1,
        "capture_sha256": _sha256(capture_path),
        "policy_model_sha256": _sha256(model_path),
        "events": len(identities["episode_id"]),
        "conditions": list(conditions),
        "guidance_horizon": guidance_horizon,
        "paired_noise": "async_request_generation_noise",
    }
    arrays: dict[str, np.ndarray] = {
        name: np.asarray(values, dtype=np.int64) for name, values in identities.items()
    }
    arrays.update(
        {
            name: np.stack(values).astype(np.float32, copy=False)
            for name, values in generated.items()
        }
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {**metadata, "output": str(destination), "sha256": _sha256(destination)}


def _representative_conditioned_events(
    events: tuple[HandoffEvent, ...], identities: list[tuple[int, int, int]]
) -> list[tuple[int, HandoffEvent]]:
    """Select the first frozen event in every final switch stratum."""
    event_by_identity = {
        (event.episode_id, event.generation, event.request_id): event for event in events
    }
    ordered = [(index, event_by_identity[identity]) for index, identity in enumerate(identities)]
    selected: list[tuple[int, HandoffEvent]] = []
    for stratum in ("0", "1", ">=2"):
        match = next(
            (item for item in ordered if switch_stratum(item[1].n_switches) == stratum),
            None,
        )
        if match is None:
            raise ValueError(f"generated-bank oracle has no {stratum}-switch handoff")
        selected.append(match)
    selected_indices = {index for index, _ in selected}
    elapsed = {event.latency_length for _, event in selected}
    if len(elapsed) == 1:
        different = next(
            (
                item
                for item in ordered
                if item[0] not in selected_indices and item[1].latency_length not in elapsed
            ),
            None,
        )
        if different is None:
            raise ValueError("generated-bank oracle has only one elapsed handoff delay")
        selected.append(different)
    return sorted(selected)


def run_conditioned_real_policy_replay_gate(
    capture_path: str | Path,
    policy_path: str | Path,
    generated_banks_path: str | Path,
    *,
    device: str = "cuda",
    guidance_horizon: int = 15,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Exactly regenerate representative Static, GT, Static-live, and Reactive banks."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    if guidance_horizon != 15:
        raise ValueError("the frozen Static-live oracle used guidance_horizon=15")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy = SmolVLAPolicy.from_pretrained(str(policy_path)).to(target).eval()
    if int(getattr(policy.config, "n_obs_steps", 1)) != 1:
        raise ValueError("sparse independent replay requires SmolVLA n_obs_steps=1")
    configure_rtc_prefix_guidance(policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(target)}},
    )
    sampler = SmolVLACandidateSampler(
        policy,
        preprocessor,
        postprocessor,
        device=str(target),
    )
    oracle = _generated_arrays(generated_banks_path)
    identities = list(
        zip(
            oracle["episode_id"].tolist(),
            oracle["generation"].tolist(),
            oracle["request_id"].tolist(),
            strict=True,
        )
    )
    conditions = ("static", "gt", "static_live", "reactive")
    errors: dict[str, dict[str, list[dict[str, Any]]]] = {
        condition: {"policy": [], "control": []} for condition in conditions
    }
    selected_rows: list[dict[str, Any]] = []

    with load_diagnostic_capture(capture_path) as capture:
        capture.require_exact_replay()
        events, _ = reconstruct_events(capture)
        if set(identities) != {
            (event.episode_id, event.generation, event.request_id) for event in events
        }:
            raise ValueError("generated banks and factual handoffs have different event identities")
        selected = _representative_conditioned_events(events, identities)
        generation_noise = capture.array("async_request_generation_noise")
        policy_banks = capture.array("candidate_bank_policy_actions")
        sparse_meta = capture.metadata["sparse_exact_observations"]

        for oracle_index, event in selected:
            static_live_prefix = policy_banks[
                event.old_bank_index,
                :,
                event.request_phase : event.request_phase + guidance_horizon,
            ]
            if (
                static_live_prefix.shape[1] != guidance_horizon
                or not np.isfinite(static_live_prefix).all()
            ):
                raise ValueError("Static-live prefix is incomplete")
            gt_prefix = np.repeat(
                event.gt_policy_prefix[None], len(event.static_policy_prefixes), axis=0
            )
            schedules = {
                "static": (
                    "request_origin",
                    event.static_policy_prefixes,
                    event.latency_length,
                    event.latency_length,
                ),
                "gt": (
                    "request_origin",
                    gt_prefix,
                    event.latency_length,
                    event.latency_length,
                ),
                "static_live": (
                    "request_origin",
                    static_live_prefix,
                    event.predicted_inference_delay,
                    guidance_horizon,
                ),
                "reactive": ("activation", None, None, None),
            }
            noise = torch.as_tensor(
                generation_noise[event.request_index],
                dtype=torch.float32,
                device=target,
            )
            for condition, (role, prefix, inference_delay, execution_horizon) in schedules.items():
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
                for processor in (preprocessor, postprocessor):
                    reset_processor = getattr(processor, "reset", None)
                    if callable(reset_processor):
                        reset_processor()
                raw = capture.sparse_observation(
                    episode_id=event.episode_id,
                    generation=event.generation,
                    request_id=event.request_id,
                    event_role=role,
                )
                prepared = prepare_observation_for_inference(
                    raw,
                    target,
                    str(sparse_meta.get("task", "")),
                    str(sparse_meta.get("robot_type", "")),
                )
                prefix_tensor = (
                    None
                    if prefix is None
                    else torch.as_tensor(prefix, dtype=torch.float32, device=target)
                )
                replay_policy, replay_control = sampler.sample_candidate_bank(
                    prepared,
                    int(noise.shape[0]),
                    noise=noise,
                    prefix_actions=prefix_tensor,
                    inference_delay=inference_delay,
                    execution_horizon=execution_horizon,
                )
                errors[condition]["policy"].append(
                    array_replay_error(
                        oracle[f"{condition}_policy_actions"][oracle_index],
                        replay_policy.detach().float().cpu().numpy(),
                    )
                )
                errors[condition]["control"].append(
                    array_replay_error(
                        oracle[f"{condition}_control_actions"][oracle_index],
                        replay_control.detach().float().cpu().numpy(),
                    )
                )
            selected_rows.append(
                {
                    "oracle_index": oracle_index,
                    "episode_id": event.episode_id,
                    "generation": event.generation,
                    "request_id": event.request_id,
                    "selection_reason": (
                        f"first frozen generated-bank event in {switch_stratum(event.n_switches)} "
                        "switch stratum"
                    ),
                    "switch_stratum": switch_stratum(event.n_switches),
                    "switches": event.n_switches,
                    "actual_elapsed_delay": event.latency_length,
                    "static_gt_prefix_length": event.latency_length,
                    "static_live_prefix_length": guidance_horizon,
                    "static_live_predicted_inference_delay": event.predicted_inference_delay,
                }
            )

    by_condition: dict[str, Any] = {}
    for condition in conditions:
        policy_errors = errors[condition]["policy"]
        control_errors = errors[condition]["control"]
        by_condition[condition] = {
            "events": len(policy_errors),
            "policy_exact_match_rate": float(
                np.mean([error["exact_match"] for error in policy_errors])
            ),
            "control_exact_match_rate": float(
                np.mean([error["exact_match"] for error in control_errors])
            ),
            "policy_max_abs": max(error["max_abs"] for error in policy_errors),
            "control_max_abs": max(error["max_abs"] for error in control_errors),
        }
    passed = all(
        metrics["policy_exact_match_rate"] == 1.0
        and metrics["control_exact_match_rate"] == 1.0
        and metrics["policy_max_abs"] == 0.0
        and metrics["control_max_abs"] == 0.0
        for metrics in by_condition.values()
    )
    result = {
        "name": "representative exact prefix-conditioned SmolVLA replay",
        "selection_rule": (
            "first frozen generated-bank event in each of the 0, 1, and >=2 switch strata; "
            "add the first different-L_e event only if those three share one delay"
        ),
        "comparison": "exact bit equality; no numerical tolerance",
        "events": selected_rows,
        "conditions": by_condition,
        "passed": passed,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise ValueError(f"conditioned real-policy replay failed: {result}")
    return result


def run(
    capture_path: str | Path,
    generated_banks_path: str | Path,
    output_path: str | Path | None = None,
    *,
    guidance_horizon: int = 15,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    with load_diagnostic_capture(capture_path) as capture:
        result = analyze_generated_banks(
            capture,
            generated_banks_path,
            guidance_horizon=guidance_horizon,
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
    parser.add_argument("--generated-banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidance-horizon", type=int, default=15)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument(
        "--regenerate-generated-banks",
        action="store_true",
        help="generate --generated-banks from the real policy before analysis",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.regenerate_generated_banks:
        if args.policy_path is None:
            parser.error("--regenerate-generated-banks requires --policy-path")
        generate_real_policy_banks(
            args.capture,
            args.policy_path,
            args.generated_banks,
            device=args.device,
            guidance_horizon=args.guidance_horizon,
        )
    result = run(
        args.capture,
        args.generated_banks,
        args.output,
        guidance_horizon=args.guidance_horizon,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.policy_path is not None:
        result["real_policy_replay_gate"] = run_real_policy_replay_gate(
            args.capture,
            args.policy_path,
            device=args.device,
        )
        result["conditioned_real_policy_replay_gate"] = run_conditioned_real_policy_replay_gate(
            args.capture,
            args.policy_path,
            args.generated_banks,
            device=args.device,
            guidance_horizon=args.guidance_horizon,
        )
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

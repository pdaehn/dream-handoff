"""Reconstruct the final absolute-hysteresis calibration from recovered artifacts.

This is deliberately a single-system analysis.  It regenerates the final
candidate/matching/recorded-continuation matrix, calibrates absolute hysteresis
on the fixed calibration episode subset, freezes the resulting threshold, and scores
that threshold on the untouched evaluation episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from dream_handoff.dataset import FINAL_DATASET_REPO_ID, EpisodeSplit
from dream_handoff.inference import FINAL_HYSTERESIS_TAU, SmolVLACandidateSampler
from dream_handoff.r2dreamer import R2DreamerRuntime

LOGGER = logging.getLogger(__name__)

DATASET_REPO_ID = "pdaehn/dreamhandoff-so101-rectangle-on-peg"
DATASET_REVISION = "70d08de50664a8c973375484c56ca553815719e5"
POLICY_REPO_ID = "pdaehn/dreamhandoff-smolvla-rectangle-on-peg"
POLICY_REVISION = "7e3d6e4c8ec1d43673b0ca96035e8995ed179601"
POLICY_MODEL_SHA256 = "668cbfa273b30bd023ebcdc246e5fd0d533633bdf0d0d3c6ccd2891aef5dd64c"
R2_ARTIFACT_ID = "dreamhandoff-r2-rectangle-dynamics"
R2_REPO_ID = "pdaehn/dreamhandoff-r2-rectangle-dynamics"
R2_REVISION = "b3510497b0ff135220ea92468d2efbd5908e4d41"
R2_EXTERNAL_BUNDLE_PATH = "r2/dreamhandoff-r2-rectangle-dynamics/checkpoints/latest.pt"
R2_SHA256 = "5260b7c5df88929e21bd60c8686fe7b9dc43db7b2c6238cf49524b42a33ac9fc"
HISTORICAL_TAU = 0.027628261595964432

NUM_CANDIDATES = 10
LEGAL_BANK_HORIZON = 35
PRIMARY_LAG = 1
POLICY_SEED = 0
PHASE_ZERO_SEED = 0
SPLIT_SEED = 0
CALIBRATION_FRACTION = 1.0 / 3.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

PhaseZeroRng = Literal["historical_numpy", "production_torch"]


@dataclass(frozen=True)
class CandidateProxyMatrix:
    """Compact replay substrate with one row per bank phase and one column per candidate."""

    episode: np.ndarray
    bank_origin: np.ndarray
    phase: np.ndarray
    control_step: np.ndarray
    matching_distance: np.ndarray
    continuation_error: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.episode)
        for name in ("bank_origin", "phase", "control_step"):
            if np.asarray(getattr(self, name)).shape != (rows,):
                raise ValueError(f"{name} must have shape [{rows}]")
        expected = (rows, NUM_CANDIDATES)
        if np.asarray(self.matching_distance).shape != expected:
            raise ValueError(f"matching_distance must have shape {expected}")
        if np.asarray(self.continuation_error).shape != expected:
            raise ValueError(f"continuation_error must have shape {expected}")
        if not np.isfinite(self.matching_distance).all():
            raise ValueError("matching_distance contains NaN or Inf")
        if np.any(np.asarray(self.phase) < 0):
            raise ValueError("phase must be nonnegative")

    @property
    def rows(self) -> int:
        return int(len(self.episode))

    @property
    def bank_id(self) -> np.ndarray:
        identities = list(zip(self.episode.tolist(), self.bank_origin.tolist(), strict=True))
        ids: dict[tuple[int, int], int] = {}
        return np.asarray([ids.setdefault(pair, len(ids)) for pair in identities])


@dataclass(frozen=True)
class SwitchStatistics:
    switched: np.ndarray
    reversal_aba: np.ndarray
    rapid_switch_1: np.ndarray
    rapid_switch_2: np.ndarray
    rapid_switch_3: np.ndarray


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bank_origins_for_episode(length: int, horizon: int = LEGAL_BANK_HORIZON) -> list[int]:
    if length <= 0 or horizon <= 0:
        raise ValueError("episode length and horizon must be positive")
    return list(range(0, length, horizon))


def recorded_continuation_errors(
    control_actions: np.ndarray,
    recorded_state: np.ndarray,
    *,
    bank_origin: int,
    lag: int,
    phases: int | None = None,
) -> np.ndarray:
    """L2(candidate action[p], recorded observation.state[origin+p+lag]).

    Both operands are in the dataset/checkpoint's canonical six-dimensional
    control order.  Rows whose continuation target falls outside the episode
    are retained as NaN so selector state remains chronologically unbroken.
    """
    actions = np.asarray(control_actions, dtype=np.float32)
    states = np.asarray(recorded_state, dtype=np.float32)
    if actions.ndim != 3 or states.ndim != 2:
        raise ValueError("control_actions and recorded_state must be [N,H,A] and [T,A]")
    if actions.shape[0] != NUM_CANDIDATES:
        raise ValueError(f"expected {NUM_CANDIDATES} candidates")
    if actions.shape[2] != states.shape[1]:
        raise ValueError("candidate action and recorded state dimensions/order differ")
    count = min(actions.shape[1], phases if phases is not None else actions.shape[1])
    result = np.full((count, actions.shape[0]), np.nan, dtype=np.float64)
    targets = bank_origin + np.arange(count) + lag
    valid = (targets >= 0) & (targets < len(states))
    if valid.any():
        difference = actions[:, :count][:, valid] - states[targets[valid]][None, :, :]
        result[valid] = np.linalg.norm(difference, axis=-1).T
    return result


def _episode_frame_indices(dataset: Any) -> dict[int, np.ndarray]:
    episode_index = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    result: dict[int, np.ndarray] = {}
    for episode in sorted(int(value) for value in np.unique(episode_index)):
        rows = np.flatnonzero(episode_index == episode)
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError(f"episode {episode} rows are not contiguous")
        result[episode] = rows
    return result


def regenerate_candidate_proxy_matrix(
    *,
    dataset_root: Path,
    policy_path: Path,
    checkpoint_path: Path,
    validation_episodes: tuple[int, ...],
    device: str = "cuda",
) -> tuple[CandidateProxyMatrix, dict[str, Any]]:
    """Run final SmolVLA and R2 once to reconstruct the full final calibration matrix."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = LeRobotDataset(DATASET_REPO_ID, root=dataset_root, video_backend="pyav")
    action_names = tuple(dataset.meta.features["action"]["names"])
    runtime = R2DreamerRuntime.from_checkpoint(checkpoint_path, target)
    if runtime.action_names != action_names:
        raise ValueError(
            "R2 canonical action order does not match dataset observation/action order: "
            f"{runtime.action_names!r} != {action_names!r}"
        )

    policy = SmolVLAPolicy.from_pretrained(str(policy_path)).to(target).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(target)}},
    )
    sampler = SmolVLACandidateSampler(policy, preprocessor, postprocessor, device=str(target))
    generator = torch.Generator(device=target).manual_seed(POLICY_SEED)
    frame_indices = _episode_frame_indices(dataset)

    meta_episode: list[np.ndarray] = []
    meta_origin: list[np.ndarray] = []
    meta_phase: list[np.ndarray] = []
    meta_step: list[np.ndarray] = []
    matching_blocks: list[np.ndarray] = []
    proxy_blocks: list[np.ndarray] = []
    bank_count = 0

    for episode in sorted(validation_episodes):
        if episode not in frame_indices:
            raise ValueError(f"dataset does not contain validation episode {episode}")
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        preprocessor.reset()
        postprocessor.reset()
        runtime.reset()

        rows = frame_indices[episode]
        origins = set(bank_origins_for_episode(len(rows)))
        live_features: list[np.ndarray] = []
        recorded_states: list[np.ndarray] = []
        banks: list[tuple[int, np.ndarray, np.ndarray]] = []
        previous_action: torch.Tensor | None = None

        for local_step, global_row in enumerate(rows):
            frame = dataset[int(global_row)]
            recorded_states.append(
                frame["observation.state"].detach().cpu().numpy().astype(np.float32)
            )
            real_action = frame["action"].detach().cpu().numpy().astype(np.float32)
            with torch.inference_mode():
                live_state = runtime.observe(frame, previous_action=previous_action)
                live_features.append(
                    runtime.matching_feature(live_state)
                    .detach()
                    .cpu()
                    .numpy()[0]
                    .astype(np.float32)
                )
            previous_action = torch.as_tensor(real_action, device=target)

            if local_step in origins:
                with torch.inference_mode():
                    _, control = sampler.sample_candidate_bank(
                        frame, NUM_CANDIDATES, generator=generator
                    )
                    control = control[:, :LEGAL_BANK_HORIZON]
                    dreamed = runtime.imagine_states(live_state, control)
                    dreamed_features = runtime.matching_features(dreamed)
                banks.append(
                    (
                        local_step,
                        control.detach().cpu().numpy().astype(np.float32),
                        dreamed_features.detach().cpu().numpy().astype(np.float32),
                    )
                )
                bank_count += 1
                LOGGER.info(
                    "calibration reconstruction: episode=%d origin=%d/%d bank=%d",
                    episode,
                    local_step,
                    len(rows),
                    bank_count,
                )

        live = np.stack(live_features)
        recorded = np.stack(recorded_states)
        for origin, control, dreamed_features in banks:
            count = min(LEGAL_BANK_HORIZON, len(rows) - origin)
            phases = np.arange(count, dtype=np.int64)
            difference = dreamed_features[:, :count] - live[origin : origin + count][None]
            matching = np.linalg.norm(difference, axis=-1).T
            continuation = recorded_continuation_errors(
                control,
                recorded,
                bank_origin=origin,
                lag=PRIMARY_LAG,
                phases=count,
            )
            meta_episode.append(np.full(count, episode, dtype=np.int64))
            meta_origin.append(np.full(count, origin, dtype=np.int64))
            meta_phase.append(phases)
            meta_step.append(origin + phases)
            matching_blocks.append(matching)
            proxy_blocks.append(continuation)

    matrix = CandidateProxyMatrix(
        episode=np.concatenate(meta_episode),
        bank_origin=np.concatenate(meta_origin),
        phase=np.concatenate(meta_phase),
        control_step=np.concatenate(meta_step),
        matching_distance=np.concatenate(matching_blocks),
        continuation_error=np.concatenate(proxy_blocks),
    )
    metadata = {
        "action_names": list(action_names),
        "policy_chunk_horizon": int(policy.config.chunk_size),
        "matching_feature_width": int(live.shape[1]),
        "banks": int(len(np.unique(matrix.bank_id))),
        "rows": matrix.rows,
        "valid_proxy_rows": int(np.isfinite(matrix.continuation_error).all(axis=1).sum()),
    }
    return matrix, metadata


def calibration_evaluation_split(
    episode_ids: list[int] | tuple[int, ...],
    *,
    calibration_fraction: float = CALIBRATION_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    unique = sorted({int(value) for value in episode_ids})
    if len(unique) < 2:
        raise ValueError("at least two episodes are required")
    permutation = np.random.default_rng(seed).permutation(len(unique))
    count = max(1, min(len(unique) - 1, round(len(unique) * calibration_fraction)))
    return (
        sorted(unique[index] for index in permutation[:count]),
        sorted(unique[index] for index in permutation[count:]),
    )


def phase_zero_candidate_indices(
    bank_id: np.ndarray,
    *,
    seed: int = PHASE_ZERO_SEED,
    rng: PhaseZeroRng = "historical_numpy",
) -> np.ndarray:
    bank_id = np.asarray(bank_id)
    first = np.r_[True, bank_id[1:] != bank_id[:-1]]
    result = np.full(len(bank_id), -1, dtype=np.int64)
    if rng == "historical_numpy":
        draws = np.random.default_rng(seed).integers(0, NUM_CANDIDATES, size=int(first.sum()))
    elif rng == "production_torch":
        generator = torch.Generator().manual_seed(seed)
        draws = torch.randint(0, NUM_CANDIDATES, (int(first.sum()),), generator=generator).numpy()
    else:
        raise ValueError(f"unknown phase-zero RNG {rng!r}")
    result[first] = draws
    return result


def select_memoryless(matching_distance: np.ndarray, bank_first_index: np.ndarray) -> np.ndarray:
    selected = np.argmin(matching_distance, axis=1).astype(np.int64)
    override = bank_first_index >= 0
    selected[override] = bank_first_index[override]
    return selected


def replay_absolute_hysteresis(
    matching_distance: np.ndarray,
    bank_id: np.ndarray,
    *,
    threshold: float,
    bank_first_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay strict absolute hysteresis and reset the incumbent at each bank."""
    distance = np.asarray(matching_distance, dtype=np.float64)
    bank_id = np.asarray(bank_id)
    challenger = np.argmin(distance, axis=1).astype(np.int64)
    selected = np.empty(len(distance), dtype=np.int64)
    advantage = np.zeros(len(distance), dtype=np.float64)
    incumbent = -1
    previous_bank: Any = None
    for row, current_bank in enumerate(bank_id):
        if current_bank != previous_bank:
            incumbent = int(bank_first_index[row])
            if not 0 <= incumbent < distance.shape[1]:
                raise ValueError("bank-first candidate index is missing or invalid")
            previous_bank = current_bank
        else:
            margin = float(distance[row, incumbent] - distance[row, challenger[row]])
            advantage[row] = margin
            if margin > threshold:
                incumbent = int(challenger[row])
        selected[row] = incumbent
    return selected, advantage


def recorded_continuation_regret(
    selected: np.ndarray, continuation_error: np.ndarray
) -> np.ndarray:
    error = np.asarray(continuation_error, dtype=np.float64)
    valid = np.isfinite(error).any(axis=1)
    chosen = error[np.arange(len(error)), selected]
    best = np.full(len(error), np.nan)
    best[valid] = np.nanmin(error[valid], axis=1)
    regret = np.where(valid & np.isfinite(chosen), chosen - best, np.nan)
    finite = regret[np.isfinite(regret)]
    if finite.size and float(finite.min()) < -1e-12:
        raise AssertionError("recorded-continuation regret became negative")
    return regret


def proxy_top1(selected: np.ndarray, continuation_error: np.ndarray) -> np.ndarray:
    error = np.asarray(continuation_error, dtype=np.float64)
    valid = np.isfinite(error).any(axis=1)
    best = np.argmin(np.where(np.isfinite(error), error, np.inf), axis=1)
    return np.where(valid, (selected == best).astype(np.float64), np.nan)


def switch_statistics(selected: np.ndarray, bank_id: np.ndarray) -> SwitchStatistics:
    selected = np.asarray(selected)
    bank_id = np.asarray(bank_id)
    switched = np.zeros(len(selected), dtype=bool)
    reversal = np.zeros(len(selected), dtype=bool)
    rapid = {gap: np.zeros(len(selected), dtype=bool) for gap in (1, 2, 3)}
    previous_bank: Any = None
    last_switch: int | None = None
    for row, current_bank in enumerate(bank_id):
        if current_bank != previous_bank:
            previous_bank = current_bank
            last_switch = None
            continue
        switched[row] = selected[row] != selected[row - 1]
        if switched[row]:
            if (
                row >= 2
                and bank_id[row - 2] == current_bank
                and switched[row - 1]
                and selected[row] == selected[row - 2]
            ):
                reversal[row] = True
            if last_switch is not None:
                distance = row - last_switch
                for gap in rapid:
                    rapid[gap][row] = distance <= gap
            last_switch = row
    return SwitchStatistics(switched, reversal, rapid[1], rapid[2], rapid[3])


def _mean_finite(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def selection_metrics(
    selected: np.ndarray,
    matrix: CandidateProxyMatrix,
    mask: np.ndarray,
) -> dict[str, Any]:
    regret = recorded_continuation_regret(selected, matrix.continuation_error)
    top1 = proxy_top1(selected, matrix.continuation_error)
    stats = switch_statistics(selected, matrix.bank_id)
    selected_mask = np.asarray(mask, dtype=bool)
    return {
        "rows": int(selected_mask.sum()),
        "valid_proxy_rows": int(np.isfinite(regret[selected_mask]).sum()),
        "switch_count": int(stats.switched[selected_mask].sum()),
        "switch_rate": _mean_finite(stats.switched[selected_mask]),
        "recorded_continuation_regret": _mean_finite(regret[selected_mask]),
        "proxy_top1_agreement": _mean_finite(top1[selected_mask]),
        "reversal_aba_rate": _mean_finite(stats.reversal_aba[selected_mask]),
        "rapid_switch_within_1_rate": _mean_finite(stats.rapid_switch_1[selected_mask]),
        "rapid_switch_within_2_rate": _mean_finite(stats.rapid_switch_2[selected_mask]),
        "rapid_switch_within_3_rate": _mean_finite(stats.rapid_switch_3[selected_mask]),
    }


def dense_threshold_grid(advantage: np.ndarray) -> list[float]:
    values = np.asarray(advantage, dtype=np.float64)
    values = values[np.isfinite(values)]
    grid = {0.0}
    if values.size:
        linear_max = float(np.quantile(values, 0.90))
        if linear_max > 0:
            grid.update(float(value) for value in np.linspace(0.0, linear_max, 40))
        for quantile in (0.95, 0.975, 0.99, 0.995, 0.999):
            grid.add(float(np.quantile(values, quantile)))
    return sorted(grid)


def pareto_efficient_indices(
    points: list[dict[str, float]],
    *,
    keys: tuple[str, ...] = ("recorded_continuation_regret", "switch_rate"),
) -> list[int]:
    valid = [
        index for index, point in enumerate(points) if all(np.isfinite(point[key]) for key in keys)
    ]
    efficient: list[int] = []
    for index in valid:
        dominated = any(
            other != index
            and all(points[other][key] <= points[index][key] for key in keys)
            and any(points[other][key] < points[index][key] for key in keys)
            for other in valid
        )
        if not dominated:
            efficient.append(index)
    return efficient


def select_moderate_operating_point(
    rows: list[dict[str, Any]], baseline_regret: float
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["pareto"] and row["recorded_continuation_regret"] <= baseline_regret * 1.05
    ]
    if not eligible:
        raise ValueError("no Pareto point satisfies the moderate +5% regret constraint")
    return min(eligible, key=lambda row: row["switch_rate"])


def paired_bank_bootstrap(
    bank_id: np.ndarray,
    baseline: dict[str, np.ndarray],
    hysteresis: dict[str, np.ndarray],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    unique = np.unique(bank_id)
    rows_by_bank = {bank: np.flatnonzero(bank_id == bank) for bank in unique}
    generator = np.random.default_rng(seed)
    deltas = {key: [] for key in baseline}
    for _ in range(resamples):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_bank[bank] for bank in sampled])
        for key in deltas:
            left = _mean_finite(baseline[key][rows])
            right = _mean_finite(hysteresis[key][rows])
            if left is not None and right is not None:
                deltas[key].append(right - left)
    return {
        key: {
            "bootstrap_mean_delta": float(np.mean(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
        }
        for key, values in deltas.items()
    }


def _paired_metric_arrays(
    selected: np.ndarray, matrix: CandidateProxyMatrix
) -> dict[str, np.ndarray]:
    stats = switch_statistics(selected, matrix.bank_id)
    return {
        "recorded_continuation_regret": recorded_continuation_regret(
            selected, matrix.continuation_error
        ),
        "proxy_top1_agreement": proxy_top1(selected, matrix.continuation_error),
        "switch_rate": stats.switched.astype(np.float64),
        "reversal_aba_rate": stats.reversal_aba.astype(np.float64),
    }


def calibrate_absolute_hysteresis(
    matrix: CandidateProxyMatrix,
    *,
    phase_zero_rng: PhaseZeroRng = "historical_numpy",
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    episodes = sorted({int(value) for value in matrix.episode})
    calibration_episodes, evaluation_episodes = calibration_evaluation_split(episodes)
    calibration_mask = np.isin(matrix.episode, calibration_episodes)
    evaluation_mask = np.isin(matrix.episode, evaluation_episodes)
    if (
        calibration_mask.any()
        and evaluation_mask.any()
        and np.any(calibration_mask & evaluation_mask)
    ):
        raise AssertionError("calibration and evaluation episodes overlap")

    bank_id = matrix.bank_id
    bank_first = phase_zero_candidate_indices(bank_id, rng=phase_zero_rng)
    baseline = select_memoryless(matrix.matching_distance, bank_first)
    baseline_calibration = selection_metrics(baseline, matrix, calibration_mask)

    tau_zero, advantage = replay_absolute_hysteresis(
        matrix.matching_distance,
        bank_id,
        threshold=0.0,
        bank_first_index=bank_first,
    )
    del tau_zero
    sweep_mask = calibration_mask & (matrix.phase > 0)
    thresholds = dense_threshold_grid(advantage[sweep_mask])
    rows: list[dict[str, Any]] = []
    selections: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        selected, _ = replay_absolute_hysteresis(
            matrix.matching_distance,
            bank_id,
            threshold=threshold,
            bank_first_index=bank_first,
        )
        selections[threshold] = selected
        metrics = selection_metrics(selected, matrix, calibration_mask)
        rows.append({"threshold": threshold, **metrics})
    efficient = set(pareto_efficient_indices(rows))
    for index, row in enumerate(rows):
        row["pareto"] = index in efficient
    baseline_regret = baseline_calibration["recorded_continuation_regret"]
    if baseline_regret is None:
        raise ValueError("calibration proxy regret is undefined")
    selected_point = select_moderate_operating_point(rows, baseline_regret)
    threshold = float(selected_point["threshold"])
    frozen = selections[threshold]

    baseline_evaluation = selection_metrics(baseline, matrix, evaluation_mask)
    frozen_evaluation = selection_metrics(frozen, matrix, evaluation_mask)
    baseline_arrays = _paired_metric_arrays(baseline, matrix)
    frozen_arrays = _paired_metric_arrays(frozen, matrix)
    uncertainty = paired_bank_bootstrap(
        bank_id[evaluation_mask],
        {key: value[evaluation_mask] for key, value in baseline_arrays.items()},
        {key: value[evaluation_mask] for key, value in frozen_arrays.items()},
        resamples=bootstrap_resamples,
    )
    for key, values in uncertainty.items():
        baseline_mean = _mean_finite(baseline_arrays[key][evaluation_mask])
        frozen_mean = _mean_finite(frozen_arrays[key][evaluation_mask])
        if baseline_mean is None or frozen_mean is None:
            raise ValueError(f"held-out paired metric {key!r} is undefined")
        values["observed_delta"] = frozen_mean - baseline_mean

    return {
        "calibration_episodes": calibration_episodes,
        "evaluation_episodes": evaluation_episodes,
        "phase_zero_rng": phase_zero_rng,
        "baseline_calibration": baseline_calibration,
        "threshold_sweep": rows,
        "pareto_curve": [row for row in rows if row["pareto"]],
        "moderate_constraint": {
            "maximum_relative_regret_increase": 0.05,
            "maximum_mean_regret": baseline_regret * 1.05,
            "objective": "lowest switch rate among Pareto-efficient eligible points",
        },
        "selected_point": selected_point,
        "selected_tau": threshold,
        "baseline_evaluation": baseline_evaluation,
        "frozen_hysteresis_evaluation": frozen_evaluation,
        "paired_bank_bootstrap": {
            "method": "paired bank-clustered bootstrap",
            "unit": "(episode, bank_origin)",
            "bank_clusters": int(len(np.unique(bank_id[evaluation_mask]))),
            "resamples": bootstrap_resamples,
            "seed": BOOTSTRAP_SEED,
            "confidence_interval": "2.5/97.5 percentile",
            "delta_direction": "frozen absolute hysteresis minus memoryless nearest",
            "metrics": uncertainty,
        },
        "selected_indices": frozen,
        "bank_first_indices": bank_first,
    }


def matrix_diagnostics(matrix: CandidateProxyMatrix) -> dict[str, Any]:
    valid = np.isfinite(matrix.continuation_error).all(axis=1)
    selected = np.argmin(matrix.matching_distance, axis=1)
    regret = recorded_continuation_regret(selected, matrix.continuation_error)
    top1 = proxy_top1(selected, matrix.continuation_error)
    return {
        "rows": matrix.rows,
        "valid_proxy_rows": int(valid.sum()),
        "banks": int(len(np.unique(matrix.bank_id))),
        "episodes": int(len(np.unique(matrix.episode))),
        "memoryless_proxy_top1_agreement": _mean_finite(top1[valid]),
        "memoryless_recorded_continuation_regret": _mean_finite(regret[valid]),
    }


def compare_reference_matrix(matrix: CandidateProxyMatrix, reference_path: Path) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=True) as reference:
        expected = {
            "episode": reference["full_episode"],
            "bank_origin": reference["full_bank_origin"],
            "phase": reference["full_phase"],
            "control_step": reference["full_control_step"],
            "matching_distance": reference["wm__concat"],
            "continuation_error": reference["oracle__lag_1"],
        }
    comparison: dict[str, Any] = {}
    all_equal = True
    for key, historical in expected.items():
        reconstructed = np.asarray(getattr(matrix, key))
        shape_equal = reconstructed.shape == historical.shape
        equal = bool(shape_equal and np.array_equal(reconstructed, historical, equal_nan=True))
        all_equal &= equal
        finite = np.isfinite(reconstructed) & np.isfinite(historical) if shape_equal else None
        maximum = (
            float(np.max(np.abs(reconstructed[finite] - historical[finite])))
            if shape_equal and finite is not None and finite.any()
            else None
        )
        comparison[key] = {
            "shape": list(reconstructed.shape),
            "historical_shape": list(historical.shape),
            "exact_equal": equal,
            "max_abs_difference": maximum,
        }
    return {
        "reference_artifact": "historical final paired_arrays.npz",
        "reference_sha256": sha256_file(reference_path),
        "all_arrays_exact_equal": all_equal,
        "arrays": comparison,
    }


def _expected_tau_verification(
    matrix: CandidateProxyMatrix,
    result: dict[str, Any],
    *,
    expected_tau: float,
) -> dict[str, Any]:
    """Verify an independently selected point against a provenance value."""
    selected_tau = float(result["selected_tau"])
    if selected_tau == expected_tau:
        return {
            "classification": "exact",
            "expected_tau": expected_tau,
            "absolute_difference": 0.0,
            "selector_decisions_identical": True,
        }
    bank_first = result["bank_first_indices"]
    selected, _ = replay_absolute_hysteresis(
        matrix.matching_distance,
        matrix.bank_id,
        threshold=selected_tau,
        bank_first_index=bank_first,
    )
    expected, _ = replay_absolute_hysteresis(
        matrix.matching_distance,
        matrix.bank_id,
        threshold=expected_tau,
        bank_first_index=bank_first,
    )
    identical = bool(np.array_equal(selected, expected))
    return {
        "classification": "numerically_equivalent" if identical else "non_reproduction",
        "expected_tau": expected_tau,
        "absolute_difference": abs(selected_tau - expected_tau),
        "selector_decisions_identical": identical,
    }


def _compact_calibration(result: dict[str, Any]) -> dict[str, Any]:
    public_rng = {
        "historical_numpy": "numpy",
        "production_torch": "torch",
    }[result["phase_zero_rng"]]
    return {
        "phase0_rng": public_rng,
        "selected_tau": result["selected_tau"],
        "calibration": {
            "memoryless_nearest": result["baseline_calibration"],
            "selected_point": result["selected_point"],
        },
        "held_out_evaluation": {
            "memoryless_nearest": result["baseline_evaluation"],
            "frozen_absolute_hysteresis": result["frozen_hysteresis_evaluation"],
            "paired_uncertainty": result["paired_bank_bootstrap"],
        },
        "sweep": {
            "thresholds": [row["threshold"] for row in result["threshold_sweep"]],
            "pareto_curve": result["pareto_curve"],
            "moderate_rule": result["moderate_constraint"],
        },
    }


def build_result_document(
    *,
    matrix: CandidateProxyMatrix,
    generation: dict[str, Any],
    split: EpisodeSplit,
    policy_model_hash: str,
    checkpoint_hash: str,
) -> dict[str, Any]:
    """Build the canonical runtime-faithful calibration record."""
    runtime = calibrate_absolute_hysteresis(matrix, phase_zero_rng="production_torch")
    verification = _expected_tau_verification(matrix, runtime, expected_tau=FINAL_HYSTERESIS_TAU)
    if verification["classification"] == "non_reproduction":
        raise RuntimeError(
            "runtime-faithful Torch calibration did not reproduce its expected threshold: "
            f"{verification}"
        )

    compact: dict[str, Any] = {
        "schema_version": 3,
        "analysis": "absolute-hysteresis stability-reactivity calibration",
        "artifacts": {
            "dataset": {
                "repo_id": DATASET_REPO_ID,
                "revision": DATASET_REVISION,
                "external_bundle_path": "dataset/dreamhandoff-so101-rectangle-on-peg",
                "split_manifest": "configs/data/episode_split.json",
                "split_sha256": split.canonical_sha256,
            },
            "smolvla": {
                "repo_id": POLICY_REPO_ID,
                "revision": POLICY_REVISION,
                "external_bundle_path": "policy/dreamhandoff-smolvla-rectangle-on-peg",
                "model_safetensors_sha256": policy_model_hash,
            },
            "r2": {
                "artifact_id": R2_ARTIFACT_ID,
                "repo_id": R2_REPO_ID,
                "revision": R2_REVISION,
                "external_bundle_path": R2_EXTERNAL_BUNDLE_PATH,
                "checkpoint_sha256": checkpoint_hash,
            },
        },
        "matrix": {**generation, **matrix_diagnostics(matrix)},
        "proxy": {
            "name": "recorded-continuation proxy",
            "definition": (
                "L2 distance between candidate control-space action at phase p and recorded "
                "future observation.state at bank_origin + p + lag"
            ),
            "lag": PRIMARY_LAG,
            "action_order": generation["action_names"],
            "limitation": (
                "Agreement with the recorded continuation does not estimate what would have "
                "physically happened had a counterfactual candidate been executed."
            ),
        },
        "candidate_generation": {
            "candidate_count": NUM_CANDIDATES,
            "policy_horizon": generation["policy_chunk_horizon"],
            "legal_bank_horizon": LEGAL_BANK_HORIZON,
            "bank_origins": "range(0, episode_length, 35)",
            "policy_noise": (
                "one rollout-wide CUDA torch.Generator seeded 0; one [10,50,max_action_dim] "
                "float32 normal draw per bank in sorted episode/origin order"
            ),
            "matching_representation": "R2 concat: flattened stochastic state then deter",
            "observation_state": "recurrent posterior reset once per episode",
        },
        "selector": {
            "rule": "switch iff d(incumbent) - d(nearest) > tau",
            "incumbent_reset": "at every bank activation",
            "runtime_phase_zero": "rollout-wide CPU torch.Generator seeded 0, one draw per bank",
            "tie_handling": "argmin chooses the lowest candidate index",
            "phase_band": [0, LEGAL_BANK_HORIZON - 1],
        },
        "split": {
            "calibration_episodes": runtime["calibration_episodes"],
            "evaluation_episodes": runtime["evaluation_episodes"],
            "calibration_fraction": CALIBRATION_FRACTION,
            "split_seed": SPLIT_SEED,
        },
        "calibration_procedure": {
            "sweep_construction": (
                "40-point linear grid from 0 through the calibration tau=0 advantage q90, "
                "plus q95/q97.5/q99/q99.5/q99.9"
            ),
            "pareto_definition": (
                "non-dominated simultaneous minimization of calibration switch rate and "
                "mean recorded-continuation regret"
            ),
            "moderate_rule": (
                "lowest-switch-rate Pareto point with mean recorded-continuation regret "
                "no more than 5% above the memoryless baseline"
            ),
        },
        "runtime_calibration": {
            "status": "final_operating_point",
            "phase0_rng_semantics": (
                "rollout-wide CPU torch.Generator manual_seed(0), one torch.randint per bank"
            ),
            "verification": verification,
            **_compact_calibration(runtime),
        },
        "final_configuration": {
            "selector": "absolute_hysteresis",
            "tau": FINAL_HYSTERESIS_TAU,
            "calibration_source": "runtime_calibration",
            "phase0_rng": "torch",
            "phase0_seed": PHASE_ZERO_SEED,
            "num_candidates": NUM_CANDIDATES,
            "max_bank_age": LEGAL_BANK_HORIZON,
            "bank_refill_threshold": 15,
            "guidance_horizon": 15,
        },
        "bootstrap": {
            "method": "paired bank-clustered bootstrap",
            "unit": "(episode, bank_origin)",
            "held_out_bank_clusters": int(
                len(
                    np.unique(
                        matrix.bank_id[np.isin(matrix.episode, runtime["evaluation_episodes"])]
                    )
                )
            ),
            "paired": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "2.5/97.5 percentile",
        },
    }
    return compact


def run_reconstruction(
    *,
    dataset_root: Path,
    policy_path: Path,
    checkpoint_path: Path,
    split_path: Path,
    output_path: Path,
    device: str = "cuda",
    reference_matrix: Path | None = None,
    diagnostics_output: Path | None = None,
) -> dict[str, Any]:
    if reference_matrix is not None and diagnostics_output is None:
        raise ValueError("--reference-matrix requires --diagnostics-output")
    started = time.perf_counter()
    split = EpisodeSplit.read(split_path)
    if split.dataset_repo_id != FINAL_DATASET_REPO_ID:
        raise ValueError("episode split does not belong to the final dataset")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != R2_SHA256:
        raise ValueError(f"unexpected R2 checkpoint SHA-256: {checkpoint_hash}")
    policy_model_hash = sha256_file(policy_path / "model.safetensors")
    if policy_model_hash != POLICY_MODEL_SHA256:
        raise ValueError(f"unexpected SmolVLA model SHA-256: {policy_model_hash}")

    matrix, generation = regenerate_candidate_proxy_matrix(
        dataset_root=dataset_root,
        policy_path=policy_path,
        checkpoint_path=checkpoint_path,
        validation_episodes=split.validation_episode_indices,
        device=device,
    )
    compact = build_result_document(
        matrix=matrix,
        generation=generation,
        split=split,
        policy_model_hash=policy_model_hash,
        checkpoint_hash=checkpoint_hash,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")

    if diagnostics_output is not None:
        diagnostics: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "hysteresis calibration reconstruction diagnostics",
            "canonical_result_sha256": sha256_file(output_path),
            "runtime": {
                "seconds": time.perf_counter() - started,
                "device": device,
                "gpu": torch.cuda.get_device_name(torch.device(device))
                if torch.device(device).type == "cuda"
                else None,
            },
        }
        if reference_matrix is not None:
            diagnostics["historical_matrix_parity"] = compare_reference_matrix(
                matrix, reference_matrix
            )
        diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--r2-checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("configs/data/episode_split.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-matrix", type=Path)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="optional run-specific runtime and reference-parity diagnostics JSON",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run_reconstruction(
        dataset_root=args.dataset,
        policy_path=args.policy,
        checkpoint_path=args.r2_checkpoint,
        split_path=args.split,
        output_path=args.output,
        device=args.device,
        reference_matrix=args.reference_matrix,
        diagnostics_output=args.diagnostics_output,
    )
    print(json.dumps({"selected_tau": result["runtime_calibration"]["selected_tau"]}))


if __name__ == "__main__":
    main()

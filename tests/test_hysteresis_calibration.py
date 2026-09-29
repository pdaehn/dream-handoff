from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dream_handoff.analysis import hysteresis_calibration as calibration
from dream_handoff.analysis.hysteresis_calibration import (
    CandidateProxyMatrix,
    calibrate_absolute_hysteresis,
    calibration_evaluation_split,
    pareto_efficient_indices,
    phase_zero_candidate_indices,
    recorded_continuation_errors,
    recorded_continuation_regret,
    replay_absolute_hysteresis,
    select_memoryless,
    select_moderate_operating_point,
)

VALIDATION_EPISODES = (
    79,
    12,
    96,
    101,
    36,
    17,
    64,
    27,
    74,
    45,
    61,
    38,
    106,
    100,
    51,
    62,
    65,
    33,
    5,
    53,
    113,
    97,
    49,
    108,
)


def _synthetic_matrix(episodes: tuple[int, ...] = tuple(range(6))) -> CandidateProxyMatrix:
    episode_rows: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for episode in episodes:
        phase = np.arange(5, dtype=np.int64)
        distance = np.full((5, calibration.NUM_CANDIDATES), 10.0)
        distance[0] = 0.0
        distance[1:, 0] = [0.0, 0.30, 0.0, 0.30]
        distance[1:, 1] = [0.20, 0.0, 0.20, 0.0]
        error = np.full_like(distance, 5.0)
        error[:, 0] = [0.0, 0.0, 1.0, 0.0, 1.0]
        error[:, 1] = [1.0, 1.0, 0.0, 1.0, 0.0]
        episode_rows.append(np.full(5, episode, dtype=np.int64))
        origins.append(np.zeros(5, dtype=np.int64))
        phases.append(phase)
        steps.append(phase.copy())
        distances.append(distance)
        errors.append(error)
    return CandidateProxyMatrix(
        episode=np.concatenate(episode_rows),
        bank_origin=np.concatenate(origins),
        phase=np.concatenate(phases),
        control_step=np.concatenate(steps),
        matching_distance=np.concatenate(distances),
        continuation_error=np.concatenate(errors),
    )


def test_recorded_continuation_proxy_uses_bank_origin_phase_and_lag() -> None:
    actions = np.zeros((calibration.NUM_CANDIDATES, 3, 2), dtype=np.float32)
    actions[:, :, 0] = np.arange(3)
    states = np.stack([np.array([row, 0.0], dtype=np.float32) for row in range(7)])

    error = recorded_continuation_errors(actions, states, bank_origin=1, lag=1, phases=3)

    # phase p is compared with state[bank_origin + p + lag] = state[p + 2].
    np.testing.assert_allclose(error[:, 0], [2.0, 2.0, 2.0])


def test_recorded_continuation_proxy_lag_changes_only_target_index() -> None:
    actions = np.zeros((calibration.NUM_CANDIDATES, 2, 1), dtype=np.float32)
    states = np.arange(5, dtype=np.float32)[:, None]
    lag_zero = recorded_continuation_errors(actions, states, bank_origin=1, lag=0)
    lag_one = recorded_continuation_errors(actions, states, bank_origin=1, lag=1)
    np.testing.assert_allclose(lag_zero[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(lag_one[:, 0], [2.0, 3.0])


def test_proxy_requires_matching_control_space_units_and_order() -> None:
    actions = np.zeros((calibration.NUM_CANDIDATES, 1, 6), dtype=np.float32)
    states = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions/order"):
        recorded_continuation_errors(actions, states, bank_origin=0, lag=1)


def test_recorded_continuation_regret_is_nonnegative() -> None:
    error = np.array([[3.0, 1.0], [0.5, 0.75], [np.nan, np.nan]])
    regret = recorded_continuation_regret(np.array([0, 1, 0]), error)
    np.testing.assert_allclose(regret[:2], [2.0, 0.25])
    assert np.isnan(regret[2])
    assert np.nanmin(regret) >= 0.0


def test_memoryless_baseline_overrides_only_phase_zero() -> None:
    distance = np.array([[0.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    selected = select_memoryless(distance, np.array([1, -1, -1]))
    np.testing.assert_array_equal(selected, [1, 1, 0])


def test_hysteresis_resets_incumbent_at_each_bank_activation() -> None:
    distance = np.full((4, calibration.NUM_CANDIDATES), 5.0)
    distance[:, 0] = 0.0
    bank_id = np.array([0, 0, 1, 1])
    first = np.array([1, -1, 2, -1])
    selected, _ = replay_absolute_hysteresis(
        distance, bank_id, threshold=100.0, bank_first_index=first
    )
    np.testing.assert_array_equal(selected, [1, 1, 2, 2])


def test_absolute_hysteresis_uses_strict_greater_than() -> None:
    distance = np.full((2, calibration.NUM_CANDIDATES), 5.0)
    distance[0] = 0.0
    distance[1, 0] = 0.125
    distance[1, 1] = 0.0
    bank = np.array([0, 0])
    first = np.array([0, -1])
    at_boundary, _ = replay_absolute_hysteresis(
        distance, bank, threshold=0.125, bank_first_index=first
    )
    below_boundary, _ = replay_absolute_hysteresis(
        distance,
        bank,
        threshold=np.nextafter(0.125, 0.0),
        bank_first_index=first,
    )
    np.testing.assert_array_equal(at_boundary, [0, 0])
    np.testing.assert_array_equal(below_boundary, [0, 1])


def test_phase_zero_is_seeded_uniform_and_not_argmin() -> None:
    bank_id = np.repeat(np.arange(20), 2)
    first = phase_zero_candidate_indices(bank_id, seed=0, rng="historical_numpy")
    mask = np.r_[True, bank_id[1:] != bank_id[:-1]]
    assert (first[~mask] == -1).all()
    assert ((first[mask] >= 0) & (first[mask] < calibration.NUM_CANDIDATES)).all()
    assert np.any(first[mask] != 0)


def test_historical_numpy_and_production_torch_phase_zero_streams_differ() -> None:
    bank_id = np.repeat(np.arange(20), 2)
    historical = phase_zero_candidate_indices(bank_id, rng="historical_numpy")
    production = phase_zero_candidate_indices(bank_id, rng="production_torch")
    assert not np.array_equal(historical, production)


def test_final_calibration_and_evaluation_episode_sets_are_disjoint() -> None:
    calibration_episodes, evaluation_episodes = calibration_evaluation_split(VALIDATION_EPISODES)
    assert calibration_episodes == [17, 33, 38, 53, 61, 97, 106, 108]
    assert evaluation_episodes == [5, 12, 27, 36, 45, 49, 51, 62, 64, 65, 74, 79, 96, 100, 101, 113]
    assert set(calibration_episodes).isdisjoint(evaluation_episodes)


def test_pareto_efficiency_excludes_dominated_points() -> None:
    points = [
        {"recorded_continuation_regret": 1.0, "switch_rate": 0.5},
        {"recorded_continuation_regret": 1.1, "switch_rate": 0.4},
        {"recorded_continuation_regret": 1.2, "switch_rate": 0.6},
        {"recorded_continuation_regret": 0.9, "switch_rate": 0.7},
    ]
    assert pareto_efficient_indices(points) == [0, 1, 3]


def test_moderate_rule_chooses_lowest_switch_rate_within_five_percent() -> None:
    rows = [
        {"pareto": True, "recorded_continuation_regret": 1.00, "switch_rate": 0.5},
        {"pareto": True, "recorded_continuation_regret": 1.05, "switch_rate": 0.3},
        {"pareto": True, "recorded_continuation_regret": 1.051, "switch_rate": 0.2},
        {"pareto": False, "recorded_continuation_regret": 1.01, "switch_rate": 0.1},
    ]
    assert select_moderate_operating_point(rows, 1.0) is rows[1]


def test_held_out_rows_cannot_change_selected_tau() -> None:
    matrix = _synthetic_matrix()
    first = calibrate_absolute_hysteresis(matrix, bootstrap_resamples=5)
    _, evaluation = calibration_evaluation_split(tuple(range(6)))
    held_out = np.isin(matrix.episode, evaluation)
    changed_distance = matrix.matching_distance.copy()
    changed_error = matrix.continuation_error.copy()
    changed_distance[held_out] = np.flip(changed_distance[held_out], axis=1)
    changed_error[held_out] = np.flip(changed_error[held_out], axis=1)
    changed = CandidateProxyMatrix(
        episode=matrix.episode,
        bank_origin=matrix.bank_origin,
        phase=matrix.phase,
        control_step=matrix.control_step,
        matching_distance=changed_distance,
        continuation_error=changed_error,
    )
    second = calibrate_absolute_hysteresis(changed, bootstrap_resamples=5)
    assert first["selected_tau"] == second["selected_tau"]
    assert first["threshold_sweep"] == second["threshold_sweep"]


def test_selection_is_deterministic_under_fixed_inputs() -> None:
    matrix = _synthetic_matrix()
    first = calibrate_absolute_hysteresis(matrix, bootstrap_resamples=5)
    second = calibrate_absolute_hysteresis(matrix, bootstrap_resamples=5)
    assert first["selected_tau"] == second["selected_tau"]
    np.testing.assert_array_equal(first["selected_indices"], second["selected_indices"])
    assert first["paired_bank_bootstrap"] == second["paired_bank_bootstrap"]


def test_selected_tau_is_derived_from_inputs_not_deployed_constant(monkeypatch) -> None:
    matrix = _synthetic_matrix()
    first = calibrate_absolute_hysteresis(matrix, bootstrap_resamples=5)
    before = first["selected_tau"]
    monkeypatch.setattr(calibration, "HISTORICAL_TAU", 12345.0)
    monkeypatch.setattr(calibration, "FINAL_HYSTERESIS_TAU", 54321.0)
    after = calibrate_absolute_hysteresis(matrix, bootstrap_resamples=5)["selected_tau"]
    assert after == before
    assert after in [row["threshold"] for row in first["threshold_sweep"]]
    assert after != 12345.0


def test_checked_in_result_records_only_runtime_faithful_calibration() -> None:
    result_path = Path(__file__).parents[1] / "results/hysteresis_calibration.json"
    result = json.loads(result_path.read_text())
    runtime = result["runtime_calibration"]

    assert result["schema_version"] == 3
    assert "selected_tau" not in result
    assert "calibration" not in result
    assert "historical_reconstruction" not in result
    assert "phase_zero_rng_audit" not in result
    assert "0.027628261595964432" not in json.dumps(result)
    assert "numpy" not in json.dumps(result).lower()
    assert runtime["status"] == "final_operating_point"
    assert runtime["selected_tau"] == calibration.FINAL_HYSTERESIS_TAU
    assert result["final_configuration"] == {
        "selector": "absolute_hysteresis",
        "tau": calibration.FINAL_HYSTERESIS_TAU,
        "calibration_source": "runtime_calibration",
        "phase0_rng": "torch",
        "phase0_seed": 0,
        "num_candidates": 10,
        "max_bank_age": 35,
        "bank_refill_threshold": 15,
        "guidance_horizon": 15,
    }
    baseline_regret = runtime["calibration"]["memoryless_nearest"]["recorded_continuation_regret"]
    independently_selected = select_moderate_operating_point(
        runtime["sweep"]["pareto_curve"], baseline_regret
    )
    assert independently_selected["threshold"] == runtime["selected_tau"]
    assert "runtime" not in result
    assert "historical_matrix_parity" not in result
    assert result["proxy"]["name"] == "recorded-continuation proxy"
    assert "selected_indices" not in result
    assert "candidate_tensors" not in result

    bootstrap = result["bootstrap"]
    assert bootstrap == {
        "method": "paired bank-clustered bootstrap",
        "unit": "(episode, bank_origin)",
        "held_out_bank_clusters": 228,
        "paired": True,
        "resamples": 2000,
        "seed": 0,
        "interval": "2.5/97.5 percentile",
    }
    uncertainty = runtime["held_out_evaluation"]["paired_uncertainty"]
    assert uncertainty["method"] == "paired bank-clustered bootstrap"
    assert uncertainty["unit"] == "(episode, bank_origin)"
    assert uncertainty["bank_clusters"] == 228
    regret = uncertainty["metrics"]["recorded_continuation_regret"]
    assert regret["observed_delta"] == pytest.approx(0.050190291717157276)
    assert regret["bootstrap_mean_delta"] == pytest.approx(0.05012974181340478)
    assert regret["observed_delta"] != regret["bootstrap_mean_delta"]
    assert regret["ci95_low"] == pytest.approx(0.029672905379086362)
    assert regret["ci95_high"] == pytest.approx(0.07190693670297318)


def test_optional_forensic_parity_cannot_change_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = _synthetic_matrix()
    historical = calibrate_absolute_hysteresis(
        matrix, phase_zero_rng="historical_numpy", bootstrap_resamples=5
    )
    corrected = calibrate_absolute_hysteresis(
        matrix, phase_zero_rng="production_torch", bootstrap_resamples=5
    )
    monkeypatch.setattr(calibration, "HISTORICAL_TAU", historical["selected_tau"])
    monkeypatch.setattr(calibration, "FINAL_HYSTERESIS_TAU", corrected["selected_tau"])
    monkeypatch.setattr(calibration, "BOOTSTRAP_RESAMPLES", 5)
    monkeypatch.setattr(
        calibration,
        "EpisodeSplit",
        SimpleNamespace(
            read=lambda _: SimpleNamespace(
                dataset_repo_id=calibration.FINAL_DATASET_REPO_ID,
                validation_episode_indices=tuple(range(6)),
                canonical_sha256="synthetic-split",
            )
        ),
    )
    monkeypatch.setattr(
        calibration,
        "sha256_file",
        lambda path: (
            calibration.POLICY_MODEL_SHA256
            if Path(path).name == "model.safetensors"
            else calibration.R2_SHA256
        ),
    )
    monkeypatch.setattr(
        calibration,
        "regenerate_candidate_proxy_matrix",
        lambda **_: (
            matrix,
            {
                "action_names": [f"joint_{index}" for index in range(6)],
                "policy_chunk_horizon": 50,
                "matching_feature_width": 4,
                "banks": 6,
                "rows": matrix.rows,
                "valid_proxy_rows": matrix.rows,
            },
        ),
    )
    monkeypatch.setattr(
        calibration,
        "compare_reference_matrix",
        lambda *_: {"all_arrays_exact_equal": True},
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    diagnostics = tmp_path / "diagnostics.json"
    common = {
        "dataset_root": tmp_path / "dataset",
        "policy_path": tmp_path / "policy",
        "checkpoint_path": tmp_path / "latest.pt",
        "split_path": tmp_path / "split.json",
        "device": "cpu",
    }
    calibration.run_reconstruction(output_path=first, **common)
    calibration.run_reconstruction(
        output_path=second,
        reference_matrix=tmp_path / "paired_arrays.npz",
        diagnostics_output=diagnostics,
        **common,
    )

    assert first.read_bytes() == second.read_bytes()
    assert "runtime" not in json.loads(first.read_text())
    assert "historical_matrix_parity" not in json.loads(first.read_text())
    forensic = json.loads(diagnostics.read_text())
    assert "runtime" in forensic
    assert forensic["historical_matrix_parity"]["all_arrays_exact_equal"] is True

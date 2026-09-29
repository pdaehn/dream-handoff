from __future__ import annotations

import os

import numpy as np
import pytest

from dream_handoff.analysis.handoff_prediction import (
    action_path_metrics,
    comparator_metrics,
    reconstruct_events,
    switch_stratum,
)
from dream_handoff.analysis.hysteresis import (
    V1_ABSOLUTE_TAU,
    replay_absolute_hysteresis,
    selection_statistics,
)
from dream_handoff.analysis.prefix_conditioning import (
    array_replay_error,
    condition_schedule,
    crop_like_engine,
    prefix_adherence,
)
from dream_handoff.analysis.rollout_characterization import (
    _phase_diversity,
    action_difference_comparison,
    candidate_matching_geometry,
)
from dream_handoff.analysis.statistics import episode_clustered_bootstrap
from dream_handoff.capture import RequestRecord


class TinyCapture:
    """Small reader-shaped activation handoff fixture."""

    def __init__(self) -> None:
        old = np.asarray(
            [
                [[0.0], [0.1], [0.2], [0.3], [0.4]],
                [[1.0], [1.1], [1.2], [1.3], [1.4]],
            ],
            dtype=np.float32,
        )
        new = np.asarray(
            [
                [[2.0], [2.1], [2.2], [np.nan], [np.nan]],
                [[3.0], [3.1], [3.2], [np.nan], [np.nan]],
            ],
            dtype=np.float32,
        )
        self.values = {
            "control_step": np.asarray([2, 3, 4]),
            "episode_id": np.asarray([0, 0, 0]),
            "bank_origin_action_count": np.asarray([0, 0, 4]),
            "phase": np.asarray([2, 3, 0]),
            "selected_candidate_index": np.asarray([0, 1, 1]),
            "candidate_switched": np.asarray([False, True, False]),
            "final_executed_action": np.asarray([[0.2], [1.3], [3.0]], dtype=np.float32),
            "final_executed_policy_action": np.asarray([[0.2], [1.3], [3.0]], dtype=np.float32),
            "command_sent_to_robot": np.asarray([[0.2], [1.3], [3.0]], dtype=np.float32),
            "measured_joint_position": np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32),
            "live_world_model_stoch": np.ones((3, 1, 2), dtype=np.float32),
            "live_world_model_deter": np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32),
            "candidate_bank_episode_id": np.asarray([0, 0]),
            "candidate_bank_origin_action_count": np.asarray([0, 4]),
            "candidate_bank_horizon": np.asarray([5, 3]),
            "candidate_bank_control_actions": np.stack([old, new]),
            "candidate_bank_policy_actions": np.stack([old, new]),
            "async_request_activation_phase": np.asarray([0, 2, -1]),
        }
        self._requests = (
            RequestRecord(0, 0, 0, 0, 0, 0, 0, False, "", "plain", -1, -1, 0, 5),
            RequestRecord(1, 0, 1, 0, 2, 4, 2, False, "", "plain", -1, 2, 3, 3),
            RequestRecord(2, 0, 2, 0, 5, -1, -1, True, "reset", "plain", -1, 5, 3, -1),
        )

    def array(self, name: str) -> np.ndarray:
        return self.values[name]

    def requests(self) -> tuple[RequestRecord, ...]:
        return self._requests


def test_rollout_action_differences_are_action_space_and_episode_aligned() -> None:
    actions = np.asarray([[0.0], [1.0], [3.0], [100.0], [102.0]])
    bank = np.asarray([0, 0, 2, 0, 0])
    phase = np.asarray([0, 1, 0, 0, 1])
    episode = np.asarray([0, 0, 0, 1, 1])

    result = action_difference_comparison(
        actions,
        bank,
        phase,
        episode,
        action_names=("joint",),
        policy_chunk_size=4,
        interior_margin=0,
    )

    assert result["first_difference"]["bank_boundary"]["vector_l2_norm"]["mean"] == 2.0
    assert result["first_difference"]["all_within_bank"]["vector_l2_norm"]["count"] == 2
    assert "not physical velocity" in result["definition"]["space"]


def test_rollout_boundary_ratios_distinguish_all_within_from_strict_interior() -> None:
    actions = np.asarray([[0.0], [10.0], [11.0], [31.0], [32.0], [34.0]])
    bank = np.asarray([0, 0, 0, 3, 3, 3])
    phase = np.asarray([0, 1, 2, 0, 1, 2])
    episode = np.zeros(6, dtype=np.int64)

    first = action_difference_comparison(
        actions,
        bank,
        phase,
        episode,
        action_names=("joint",),
        policy_chunk_size=4,
        interior_margin=1,
    )["first_difference"]

    # The synthetic differences are boundary [20], within [10, 1, 1, 2],
    # and strict interior [1, 2]. Keep expected ratios independent of the output.
    assert first["bank_boundary"]["vector_l2_norm"]["p95"] == 20.0
    assert first["all_within_bank"]["vector_l2_norm"]["p95"] == pytest.approx(8.8)
    assert first["strict_bank_interior"]["vector_l2_norm"]["p95"] == pytest.approx(1.95)
    assert first["boundary_to_all_within_bank_p95_l2_ratio"] == pytest.approx(20 / 8.8)
    assert first["boundary_to_strict_interior_p95_l2_ratio"] == pytest.approx(20 / 1.95)
    assert first["legacy_frozen_parity"] == {
        "field_name": "boundary_to_within_p95_l2_ratio",
        "actual_denominator": "all_within_bank.vector_l2_norm.p95",
        "value": pytest.approx(20 / 8.8),
    }


def test_phase_diversity_is_mean_unordered_pair_l2() -> None:
    values = np.asarray([[[0.0], [1.0]], [[2.0], [5.0]], [[4.0], [9.0]]])
    np.testing.assert_allclose(_phase_diversity(values), [8.0 / 3.0, 16.0 / 3.0])


def test_candidate_matching_geometry_sorts_distances_and_excludes_phase_zero() -> None:
    distances = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.2, 0.5],
            [0.7, 0.9, 0.1],
        ]
    )
    result = candidate_matching_geometry(distances, np.asarray([0, 1, 2]))

    assert result["operational_row_count"] == 2
    assert result["d1"]["count"] == 2
    assert result["d1"]["mean"] == pytest.approx(0.15)
    assert result["margin_12"]["mean"] == pytest.approx(0.45)
    assert result["definition"]["operational_rows"].startswith("phase > 0")
    assert result == candidate_matching_geometry(distances, np.asarray([0, 1, 2]))


def test_candidate_matching_geometry_requires_two_finite_candidates() -> None:
    with pytest.raises(ValueError, match="at least two"):
        candidate_matching_geometry(np.asarray([[0.1]]), np.asarray([1]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        candidate_matching_geometry(np.asarray([[0.1, np.nan]]), np.asarray([1]))


def test_absolute_hysteresis_uses_strict_margin_and_resets_by_bank() -> None:
    distances = np.asarray([[0.0, 0.0], [0.04, 0.0], [0.027628261595964432, 0.0], [0.0, 0.0]])
    bank_id = np.asarray([0, 0, 0, 1])
    bank_first = np.asarray([0, -1, -1, 1])

    selected = replay_absolute_hysteresis(
        distances,
        bank_id,
        threshold=V1_ABSOLUTE_TAU,
        bank_first_index=bank_first,
    )

    np.testing.assert_array_equal(selected, [0, 1, 1, 1])
    stats = selection_statistics(selected, bank_id)
    assert stats.switch_count == 1
    np.testing.assert_array_equal(stats.dwell_lengths, [1, 2, 1])


def test_successive_switches_and_candidate_reversals_are_distinct() -> None:
    successive = selection_statistics(np.asarray([0, 1, 2]), np.asarray([0, 0, 0]))
    reversal = selection_statistics(np.asarray([0, 1, 0]), np.asarray([0, 0, 0]))

    assert int(successive.rapid_successive_switch_within_3_rows.sum()) == 1
    assert int(successive.reversal_aba.sum()) == 0
    assert int(reversal.rapid_successive_switch_within_3_rows.sum()) == 1
    assert int(reversal.reversal_aba.sum()) == 1


def test_handoff_reconstruction_accounts_for_initial_stale_and_aligned_event() -> None:
    events, counts = reconstruct_events(TinyCapture())  # type: ignore[arg-type]

    assert counts["requests_total"] == 3
    assert counts["included_events"] == 1
    assert counts["excluded_initial_inline_build"] == 1
    assert counts["excluded_stale_result"] == 1
    event = events[0]
    assert event.latency_length == 2
    assert event.request_phase == 2
    assert event.n_switches == 1
    np.testing.assert_allclose(event.gt_prefix[:, 0], [0.2, 1.3])


@pytest.mark.parametrize(("switches", "expected"), [(0, "0"), (1, "1"), (2, ">=2")])
def test_switch_strata(switches: int, expected: str) -> None:
    assert switch_stratum(switches) == expected


def test_handoff_comparators_use_target_and_action_path_selection_independently() -> None:
    factual = np.asarray([[0.0], [0.0]])
    static = np.full((10, 2, 1), 20.0)
    static[0] = np.asarray([[0.0], [1.0]])
    static[7] = np.asarray([[10.0], [0.0]])

    action = action_path_metrics(static, factual)
    assert action["path_best_candidate"] == 0
    comparison = comparator_metrics(
        np.asarray([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 0.2, 1.5, 1.6]),
        0.5,
        action_path_nearest_candidate=action["path_best_candidate"],
    )

    assert comparison["target_selected_best_of_n_static_wm_error"] == 0.2
    assert comparison["target_selected_best_of_n_static_minus_exact_history"] == pytest.approx(-0.3)
    assert comparison["action_path_nearest_static_wm_error"] == 0.8
    assert comparison["action_path_nearest_static_minus_exact_history"] == pytest.approx(0.3)


def test_zero_switch_target_selected_minimum_does_not_imply_exact_history_harm() -> None:
    factual = np.asarray([[1.0], [2.0]])
    static = np.asarray([[[0.0], [0.0]], factual])
    action = action_path_metrics(static, factual)
    assert action["path_best_candidate"] == 1
    assert action["path_matching_candidates"] == [1]

    comparison = comparator_metrics(
        np.asarray([0.2, 0.7]),
        0.7,
        action_path_nearest_candidate=action["path_best_candidate"],
    )
    assert comparison["target_selected_best_of_n_static_minus_exact_history"] <= 0
    assert comparison["action_path_nearest_static_minus_exact_history"] == 0


def test_clustered_bootstrap_is_deterministic_and_resamples_episode_clusters() -> None:
    rows = [
        {"episode_id": 0, "value": 0.0},
        {"episode_id": 0, "value": 2.0},
        {"episode_id": 1, "value": 10.0},
    ]
    first = episode_clustered_bootstrap(rows, value_key="value", num_resamples=100, seed=0)
    second = episode_clustered_bootstrap(rows, value_key="value", num_resamples=100, seed=0)

    assert first == second
    assert first["episodes"] == 2
    assert first["mean"] == 4.0


def test_clustered_bootstrap_duplicates_whole_episode_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class FirstEpisodeTwice:
        def choice(
            self,
            values: np.ndarray,
            *,
            size: int,
            replace: bool,
        ) -> np.ndarray:
            np.testing.assert_array_equal(values, [0, 1])
            assert size == 2
            assert replace
            return np.asarray([0, 0])

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: FirstEpisodeTwice())
    rows = [
        {"episode_id": 0, "value": 0.0},
        {"episode_id": 0, "value": 2.0},
        {"episode_id": 1, "value": 10.0},
    ]
    result = episode_clustered_bootstrap(rows, value_key="value", num_resamples=1, seed=0)

    assert result["mean"] == 4.0
    assert result["mean_ci95_low"] == 1.0
    assert result["mean_ci95_high"] == 1.0


def test_prefix_schedules_and_no_synthetic_gt_live() -> None:
    assert condition_schedule(
        "static", latency_length=8, guidance_horizon=15, predicted_delay=7
    ) == {"prefix_length": 8, "inference_delay": 8, "execution_horizon": 8}
    assert condition_schedule(
        "static_live", latency_length=8, guidance_horizon=15, predicted_delay=7
    ) == {"prefix_length": 15, "inference_delay": 7, "execution_horizon": 15}
    with pytest.raises(ValueError, match="unknown final prefix condition"):
        condition_schedule("gt_live", latency_length=8, guidance_horizon=15, predicted_delay=7)


def test_paired_noise_replay_crop_and_prefix_adherence() -> None:
    generated = np.arange(12, dtype=np.float32).reshape(2, 6, 1)
    persisted = generated[:, 2:5].copy()

    replay = crop_like_engine(generated, elapsed=2, horizon=3)
    assert array_replay_error(persisted, replay)["exact_match"]
    np.testing.assert_array_equal(prefix_adherence(generated, generated[:, :2]), [0.0, 0.0])


@pytest.mark.diagnostic_artifact
def test_external_rollout_and_hysteresis_parity() -> None:
    artifact = os.environ.get("DREAMHANDOFF_DIAGNOSTIC_ARTIFACT")
    if artifact is None:
        pytest.skip("set DREAMHANDOFF_DIAGNOSTIC_ARTIFACT")
    from dream_handoff.analysis.hysteresis import run as run_hysteresis
    from dream_handoff.analysis.rollout_characterization import run as run_rollout

    rollout = run_rollout(artifact)
    hysteresis = run_hysteresis(artifact)
    assert rollout["input"]["episodes"] == 24
    assert rollout["input"]["discarded_capture_episode_ids"] == [4]
    assert rollout["candidate_selection"]["switch_rate"] == 0.07210031347962383
    assert rollout["candidate_diversity"]["action_policy_space"] == 0.09263924884980883
    first = rollout["action_differences"]["first_difference"]
    second = rollout["action_differences"]["second_difference"]
    assert first["boundary_to_all_within_bank_p95_l2_ratio"] == 3.2211292926634845
    assert first["boundary_to_strict_interior_p95_l2_ratio"] == 3.143051208681689
    assert first["legacy_frozen_parity"]["actual_denominator"] == (
        "all_within_bank.vector_l2_norm.p95"
    )
    assert second["boundary_to_all_within_bank_p95_l2_ratio"] == 4.41318705515582
    assert second["boundary_to_strict_interior_p95_l2_ratio"] == 4.398977454719156
    assert hysteresis["factual_selection_match_rate"] == 1.0
    assert hysteresis["tau"] == 0.06851652264595032


@pytest.mark.inference_artifacts
def test_external_handoff_prediction_parity() -> None:
    artifact = os.environ.get("DREAMHANDOFF_DIAGNOSTIC_ARTIFACT")
    checkpoint = os.environ.get("DREAMHANDOFF_R2_CHECKPOINT")
    if artifact is None or checkpoint is None:
        pytest.skip("set DREAMHANDOFF_DIAGNOSTIC_ARTIFACT and DREAMHANDOFF_R2_CHECKPOINT")
    from dream_handoff.analysis.handoff_prediction import run

    result = run(artifact, checkpoint)
    inclusion = result["event_inclusion"]
    assert inclusion["capture_requests_total"] == 475
    assert inclusion["requests_total"] == 464
    assert inclusion["included_events"] == 430
    assert inclusion["excluded_discarded_rerecord_requests"] == 11
    assert inclusion["excluded_initial_inline_build"] == 24
    assert inclusion["excluded_stale_result"] == 10
    assert result["switch_counts"] == {"0": 255, "1": 119, ">=2": 56}
    expected_path_mismatch = {
        "0": 0.0,
        "1": 0.7638355563180237,
        ">=2": 1.1566828686880617,
        "all": 0.36202481825203786,
    }
    for stratum, expected in expected_path_mismatch.items():
        assert result["numerical_summary"][stratum]["path_error"]["mean"] == expected
    summary = result["numerical_summary"]["all"]
    assert summary["target_selected_best_of_n_static_wm_error"]["mean"] == 1.7739435715399363
    assert summary["action_path_nearest_static_wm_error"]["mean"] == 1.8137840770719904
    assert summary["exact_history_wm_error"]["mean"] == 1.7960092202175495
    assert summary["target_selected_best_of_n_static_minus_exact_history"]["mean"] == (
        -0.022065648677613053
    )
    assert summary["action_path_nearest_static_minus_exact_history"]["mean"] == (
        0.01777485685444096
    )


@pytest.mark.inference_artifacts
def test_external_prefix_generated_bank_parity() -> None:
    artifact = os.environ.get("DREAMHANDOFF_DIAGNOSTIC_ARTIFACT")
    generated = os.environ.get("DREAMHANDOFF_EXP2_GENERATED_BANKS")
    if artifact is None or generated is None:
        pytest.skip("set DREAMHANDOFF_DIAGNOSTIC_ARTIFACT and DREAMHANDOFF_EXP2_GENERATED_BANKS")
    from dream_handoff.analysis.prefix_conditioning import run

    result = run(artifact, generated)
    means = result["numerical_summary"]["by_condition"]
    assert means["plain"]["command_seam"]["mean"] == 4.1021269656744135
    assert means["static_live"]["command_seam"]["mean"] == 2.09665589274794
    assert result["gate_0"]["passed"]


@pytest.mark.prefix_policy_replay
def test_external_real_policy_prefix_replay_gate() -> None:
    artifact = os.environ.get("DREAMHANDOFF_DIAGNOSTIC_ARTIFACT")
    policy = os.environ.get("DREAMHANDOFF_POLICY")
    if artifact is None or policy is None:
        pytest.skip("set DREAMHANDOFF_DIAGNOSTIC_ARTIFACT and DREAMHANDOFF_POLICY")
    from dream_handoff.analysis.prefix_conditioning import run_real_policy_replay_gate

    result = run_real_policy_replay_gate(artifact, policy)
    assert result["events"] == 430
    assert result["passed"]


@pytest.mark.prefix_policy_replay
def test_external_real_policy_conditioned_replay_gate() -> None:
    artifact = os.environ.get("DREAMHANDOFF_DIAGNOSTIC_ARTIFACT")
    policy = os.environ.get("DREAMHANDOFF_POLICY")
    generated = os.environ.get("DREAMHANDOFF_EXP2_GENERATED_BANKS")
    if artifact is None or policy is None or generated is None:
        pytest.skip(
            "set DREAMHANDOFF_DIAGNOSTIC_ARTIFACT, DREAMHANDOFF_POLICY, and "
            "DREAMHANDOFF_EXP2_GENERATED_BANKS"
        )
    from dream_handoff.analysis.prefix_conditioning import (
        run_conditioned_real_policy_replay_gate,
    )

    result = run_conditioned_real_policy_replay_gate(artifact, policy, generated)
    assert {event["switch_stratum"] for event in result["events"]} == {"0", "1", ">=2"}
    assert len({event["actual_elapsed_delay"] for event in result["events"]}) > 1
    assert result["passed"]

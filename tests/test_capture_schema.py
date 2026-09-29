from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from dream_handoff.capture import (
    DIAGNOSTIC_FORMAT_VERSION,
    DREAMHANDOFF_DIAGNOSTIC_FORMAT,
    RolloutCapture,
    load_diagnostic_capture,
    sha256_file,
)
from dream_handoff.capture.schema import WP6_REQUIRED_FIELDS

FIXTURE = Path(__file__).parent / "fixtures" / "wp5_format16_events.json"
EXPECTED_FINAL_SHA256 = "c2c709ca5fbbaa673d8c31d71683fb5b63756596083e2b5f3d00f61975170bba"


def _configure(capture: RolloutCapture) -> None:
    capture.configure(
        action_keys=["joint_a", "joint_b"],
        fps=30.0,
        interpolation_multiplier=1,
        policy_chunk_size=3,
        max_action_dim=4,
        num_candidates=2,
        bank_refill_threshold=1,
        guidance_horizon=1,
        guidance_mode="plain",
        selector_mode="absolute_hysteresis",
        hysteresis_tau=0.25,
        phase0_seed=0,
        task="synthetic task",
        robot_type="fake_robot",
    )


def _write_fixture_capture(path: Path) -> tuple[Path, dict]:
    fixture = json.loads(FIXTURE.read_text())
    capture = RolloutCapture(path)
    _configure(capture)
    capture.event_sink("episode_started", {"epoch": fixture["request"]["epoch"]})

    request_image = np.asarray(fixture["request_image"], dtype=np.uint8)
    activation_image = np.asarray(fixture["activation_image"], dtype=np.uint8)
    request_state = np.asarray(fixture["request_state"], dtype=np.float32)
    activation_state = np.asarray(fixture["activation_state"], dtype=np.float32)
    request = fixture["request"]
    capture.event_sink(
        "bank_requested",
        {
            "request_id": request["request_id"],
            "epoch": request["epoch"],
            "request_action_count": request["origin_action_count"],
            "request_phase": request["request_phase"],
            "old_bank_origin_action_count": request["old_bank_origin_action_count"],
            "guidance_mode": "plain",
            "guidance_horizon": -1,
            "predicted_delay": request["predicted_delay"],
            "prefix_actions": None,
            "request_observation": {
                "observation.images.context": request_image,
                "observation.state": request_state,
            },
        },
    )

    policy = torch.tensor(fixture["policy_actions"], dtype=torch.float32)
    control = torch.tensor(fixture["control_actions"], dtype=torch.float32)
    noise = torch.tensor(fixture["noise"], dtype=torch.float32)
    capture.event_sink(
        "generation_ready",
        {
            "request_id": request["request_id"],
            "epoch": request["epoch"],
            "ready_action_count": 1,
            "generated_policy_actions": policy,
            "generated_control_actions": control,
            "sampled_noise": noise,
        },
    )

    live_stoch = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    live_deter = torch.tensor([[0.5]], dtype=torch.float32)
    capture.event_sink(
        "bank_activated",
        {
            "request_id": request["request_id"],
            "epoch": request["epoch"],
            "request_action_count": request["origin_action_count"],
            "activation_action_count": 1,
            "actual_elapsed": 1,
            "usable_horizon": 2,
            "bank_origin_action_count": 1,
            "bank_policy_actions": policy[:, 1:],
            "bank_control_actions": control[:, 1:],
            "bank_dreamed_matching_features": torch.tensor(
                fixture["dreamed_matching_features"], dtype=torch.float32
            ),
            "sampled_noise": noise[:, 1:],
            "activation_observation": {
                "observation.images.context": activation_image,
                "observation.state": activation_state,
            },
        },
    )

    capture.event_sink(
        "action_served",
        {
            "action_count": 1,
            "bank_origin": 1,
            "phase": 0,
            "selected_candidate_index": 0,
            "challenger_candidate_index": 0,
            "incumbent_candidate_index": None,
            "hysteresis_advantage": None,
            "candidate_distances": torch.tensor([0.0, 0.0]),
            "candidate_switched": False,
            "selection_rule": "seeded_uniform",
            "selected_policy_action": policy[0, 1],
            "control_action": control[0, 1],
            "live_world_model_stoch": live_stoch,
            "live_world_model_deter": live_deter,
        },
    )
    capture.record_observation({"joint_a": 9.0, "joint_b": 8.0})
    capture.record_action_processor(
        {"joint_a": 10.2, "joint_b": 10.3},
        {"joint_a": 10.25, "joint_b": 10.35},
    )
    capture.record_command_sent(
        {"joint_a": 10.25, "joint_b": 10.35},
        dataset_episode_index=4,
        dataset_frame_index=12,
    )
    # Event-boundary snapshots must not retain later-mutated runtime storage.
    request_image.fill(255)
    activation_image.fill(254)
    policy.fill_(-1)
    control.fill_(-2)
    noise.fill_(-3)
    return capture.save(), fixture


def test_format16_roundtrip_preserves_exact_replay_facts(tmp_path: Path) -> None:
    path, fixture = _write_fixture_capture(tmp_path / "fixture.npz")

    with load_diagnostic_capture(path) as capture:
        assert capture.format_version == DIAGNOSTIC_FORMAT_VERSION
        assert capture.metadata["format"] == DREAMHANDOFF_DIAGNOSTIC_FORMAT
        assert capture.control_step_count == 1
        assert capture.episode_count == 1
        assert capture.recorded_episode_ids == (0,)
        assert capture.discarded_episode_ids == ()
        assert capture.candidate_bank_count == 1
        assert len(capture.requests()) == 1
        assert len(capture.handoffs()) == 1
        capture.require_exact_replay()

        request = capture.requests()[0]
        request_observation = capture.sparse_observation(
            episode_id=request.episode_id,
            generation=request.generation,
            request_id=request.request_id,
            event_role="request_origin",
        )
        activation_observation = capture.sparse_observation(
            episode_id=request.episode_id,
            generation=request.generation,
            request_id=request.request_id,
            event_role="activation",
        )
        assert request_observation["observation.images.context"].dtype == np.uint8
        np.testing.assert_array_equal(
            request_observation["observation.images.context"],
            np.asarray(fixture["request_image"], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            activation_observation["observation.images.context"],
            np.asarray(fixture["activation_image"], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            capture.array("async_request_generation_noise")[0],
            np.asarray(fixture["noise"], dtype=np.float32),
        )
        assert capture.array("candidate_bank_control_actions").dtype == np.float32
        assert capture.array("command_sent_to_robot").shape == (1, 2)
        assert capture.array("async_request_generated_policy_actions").shape == (1, 2, 3, 2)


def test_empty_capture_is_valid_but_not_exact_replay(tmp_path: Path) -> None:
    capture = RolloutCapture(tmp_path / "empty.npz")
    _configure(capture)
    path = capture.save()

    with load_diagnostic_capture(path) as loaded:
        assert loaded.control_step_count == 0
        assert loaded.candidate_bank_count == 0
        assert loaded.request_count == 0
        assert loaded.metadata["capture_complete"] is False
        with pytest.raises(ValueError, match="generation noise|sparse exact"):
            loaded.require_exact_replay()


def test_final_tau_is_serialized_in_capture_metadata(tmp_path: Path) -> None:
    from dream_handoff.inference import FINAL_HYSTERESIS_TAU, DreamHandoffInferenceConfig

    config = DreamHandoffInferenceConfig(r2_checkpoint=Path("external/latest.pt"))
    capture = RolloutCapture(tmp_path / "capture-metadata.npz")
    capture.configure(
        action_keys=["joint_a", "joint_b"],
        fps=30.0,
        interpolation_multiplier=1,
        policy_chunk_size=50,
        max_action_dim=6,
        num_candidates=config.num_candidates,
        bank_refill_threshold=config.bank_refill_threshold,
        guidance_horizon=config.guidance_horizon,
        guidance_mode=config.async_bank_guidance,
        selector_mode=config.selector_mode,
        hysteresis_tau=config.hysteresis_tau,
        phase0_seed=config.phase0_seed,
        task="synthetic task",
        robot_type="fake_robot",
    )
    path = capture.save()

    with load_diagnostic_capture(path) as loaded:
        assert loaded.metadata["controller"]["hysteresis_tau"] == FINAL_HYSTERESIS_TAU
        assert loaded.metadata["controller"]["selector_mode"] == "absolute_hysteresis"


def test_genuinely_incomplete_stale_request_keeps_explicit_missing_rows(
    tmp_path: Path,
) -> None:
    capture = RolloutCapture(tmp_path / "incomplete.npz")
    _configure(capture)
    capture.event_sink("episode_started", {"epoch": 1})
    capture.event_sink(
        "bank_requested",
        {
            "request_id": 7,
            "epoch": 1,
            "request_action_count": 3,
            "request_phase": 2,
            "old_bank_origin_action_count": 1,
            "guidance_mode": "plain",
            "guidance_horizon": -1,
            "predicted_delay": 1,
            "prefix_actions": None,
            "request_observation": {
                "observation.images.context": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation.state": np.zeros(2, dtype=np.float32),
            },
        },
    )
    capture.event_sink(
        "stale_generation",
        {
            "request_id": 7,
            "epoch": 1,
            "request_action_count": 3,
            "stale_reason": "generation never completed",
        },
    )

    with load_diagnostic_capture(capture.save()) as loaded:
        request = loaded.requests()[0]
        assert request.stale and not request.generation_completed
        assert np.isnan(loaded.array("async_request_generation_noise")[0]).all()
        generated_policy = loaded.array("async_request_generated_policy_actions")
        generated_control = loaded.array("async_request_generated_control_actions")
        assert generated_policy.shape == generated_control.shape == (1, 2, 3, 2)
        assert np.isnan(generated_policy[0]).all()
        assert np.isnan(generated_control[0]).all()
        with pytest.raises(ValueError, match="no completed request generation"):
            loaded.require_exact_replay()


def test_existing_capture_destination_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "capture.npz"
    path.touch()
    with pytest.raises(FileExistsError, match="already exists"):
        RolloutCapture(path)


@pytest.mark.diagnostic_artifact
def test_frozen_final_format16_external_gate() -> None:
    artifact = os.environ.get("DREAMHANDOFF_DIAGNOSTIC_ARTIFACT")
    if artifact is None:
        pytest.skip("set DREAMHANDOFF_DIAGNOSTIC_ARTIFACT to run the 1.8 GB format-16 gate")
    path = Path(artifact)
    assert sha256_file(path) == EXPECTED_FINAL_SHA256

    with load_diagnostic_capture(path) as capture:
        assert capture.metadata["format"] == "dream-handoff-rollout-diagnostic"
        assert capture.format_version == 16
        assert capture.episode_count == 25
        assert capture.recorded_episode_count == 24
        assert capture.discarded_episode_ids == (4,)
        assert capture.recorded_episode_ids == (1, 2, 3, *range(5, 26))
        assert capture.control_step_count == 12_552
        assert capture.candidate_bank_count == 465
        assert len(capture.handoffs()) == 440
        assert (
            sum(
                handoff.request.episode_id in capture.recorded_episode_ids
                for handoff in capture.handoffs()
            )
            == 430
        )
        assert WP6_REQUIRED_FIELDS.issubset(capture.array_info)
        capture.require_exact_replay()
        sparse = capture.metadata["sparse_exact_observations"]["keys"]
        assert [item["dtype"] for item in sparse] == ["uint8", "uint8", "float32"]

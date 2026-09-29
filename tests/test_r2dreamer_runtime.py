from __future__ import annotations

import numpy as np
import pytest
import torch

from dream_handoff.r2dreamer import (
    R2DreamerActionNormalizer,
    R2DreamerModelConfig,
    R2DreamerObservationPreprocessor,
    R2DreamerPreprocessingConfig,
    R2DreamerRuntime,
    R2State,
    R2Trajectory,
)
from dream_handoff.r2dreamer.model import R2DreamerWorldModel


def _runtime(device: str = "cpu") -> R2DreamerRuntime:
    torch.manual_seed(7)
    config = R2DreamerModelConfig(
        action_dim=2,
        image_channels=0,
        state_dim=3,
        image_size=(16, 16),
        stoch=4,
        deter=16,
        hidden=8,
        discrete=4,
        blocks=4,
        encoder_depth=2,
        encoder_mults=(1, 2),
        vector_layers=1,
        vector_units=8,
    )
    preprocessing = R2DreamerPreprocessingConfig(
        image_keys=(), image_size=(16, 16), state_key="observation.state"
    )
    return R2DreamerRuntime(
        R2DreamerWorldModel(config),
        R2DreamerObservationPreprocessor(preprocessing, device=device),
        R2DreamerActionNormalizer(
            action_names=("joint_a", "joint_b"),
            q01=(0.0, 10.0),
            q99=(10.0, 50.0),
        ),
        device=device,
    )


def _observation(device: str = "cpu") -> dict[str, torch.Tensor]:
    return {"observation.state": torch.tensor([1.0, 2.0, 3.0], device=device)}


def test_preprocessing_preserves_final_image_shape_range_order_and_dtype() -> None:
    preprocessor = R2DreamerObservationPreprocessor(R2DreamerPreprocessingConfig())
    context = np.zeros((48, 80, 3), dtype=np.uint8)
    wrist = np.full((3, 48, 80), 255, dtype=np.uint8)

    processed = preprocessor(
        {
            "observation.images.context": context,
            "observation.images.wrist": wrist,
            "observation.state": np.arange(6, dtype=np.float32),
        }
    )

    assert processed["images"].shape == (64, 64, 6)
    assert processed["images"].dtype == torch.float32
    assert processed["state"].shape == (6,)
    torch.testing.assert_close(processed["images"][..., :3], torch.zeros(64, 64, 3))
    torch.testing.assert_close(processed["images"][..., 3:], torch.ones(64, 64, 3))


def test_action_q01_q99_normalization_broadcasts_uses_epsilon_and_does_not_clip() -> None:
    normalizer = R2DreamerActionNormalizer(
        action_names=("a", "fixed"),
        q01=(10.0, 4.0),
        q99=(30.0, 4.0),
        epsilon=0.5,
    )
    actions = torch.tensor([[[10.0, 4.0], [20.0, 4.25], [40.0, 3.75]]])

    normalized = normalizer.normalize(actions)

    torch.testing.assert_close(
        normalized,
        torch.tensor([[[-1.0, -1.0], [0.0, 0.0], [2.0, -2.0]]]),
    )


def test_observe_is_recurrent_and_reset_restores_first_tick_state() -> None:
    runtime = _runtime()
    first = runtime.observe(_observation(), previous_action=None)
    first_copy = first.detached_clone()
    second = runtime.observe(_observation(), previous_action=torch.tensor([8.0, 45.0]))

    assert first.stoch.shape == (1, 4, 4)
    assert first.deter.shape == (1, 16)
    assert not torch.equal(second.deter, first.deter)
    assert runtime.live_state is second

    runtime.reset()
    assert runtime.live_state is None
    after_reset = runtime.observe(_observation(), previous_action=torch.tensor([8.0, 45.0]))
    torch.testing.assert_close(after_reset.stoch, first_copy.stoch)
    torch.testing.assert_close(after_reset.deter, first_copy.deter)


def test_imagination_is_deterministic_pre_action_aligned_and_non_mutating() -> None:
    runtime = _runtime()
    live = runtime.observe(_observation(), previous_action=None)
    live_copy = live.detached_clone()
    actions = torch.tensor(
        [
            [[0.0, 10.0], [5.0, 30.0], [10.0, 50.0]],
            [[10.0, 50.0], [5.0, 30.0], [0.0, 10.0]],
        ]
    )

    trajectory = runtime.imagine_states(live, actions)
    repeated = runtime.imagine_states(live, actions)
    normalized = runtime.action_normalizer.normalize(actions)
    expanded_stoch = live.stoch.expand(2, -1, -1).clone()
    expanded_deter = live.deter.expand(2, -1).clone()
    post_stoch, post_deter = runtime.model.rssm.imagine_with_action(
        expanded_stoch, expanded_deter, normalized
    )

    assert trajectory.leading_shape == (2, 3)
    assert trajectory.stoch.shape == (2, 3, 4, 4)
    assert trajectory.deter.shape == (2, 3, 16)
    torch.testing.assert_close(trajectory.stoch[:, 0], expanded_stoch)
    torch.testing.assert_close(trajectory.deter[:, 0], expanded_deter)
    torch.testing.assert_close(trajectory.stoch[:, 1:], post_stoch[:, :-1])
    torch.testing.assert_close(trajectory.deter[:, 1:], post_deter[:, :-1])
    torch.testing.assert_close(trajectory.stoch, repeated.stoch)
    torch.testing.assert_close(trajectory.deter, repeated.deter)
    assert trajectory.deter.shape[1] == actions.shape[1]
    assert runtime.live_state is live
    torch.testing.assert_close(runtime.live_state.stoch, live_copy.stoch)
    torch.testing.assert_close(runtime.live_state.deter, live_copy.deter)


def test_final_state_consumes_the_complete_action_prefix() -> None:
    runtime = _runtime()
    live = runtime.observe(_observation(), previous_action=None)
    actions = torch.tensor(
        [
            [[0.0, 10.0], [5.0, 30.0], [10.0, 50.0]],
            [[10.0, 50.0], [5.0, 30.0], [0.0, 10.0]],
        ]
    )
    normalized = runtime.action_normalizer.normalize(actions)
    post_stoch, post_deter = runtime.model.rssm.imagine_with_action(
        live.stoch.expand(2, -1, -1).clone(),
        live.deter.expand(2, -1).clone(),
        normalized,
    )

    final = runtime.imagine_final_state(live, actions)

    torch.testing.assert_close(final.stoch, post_stoch[:, -1])
    torch.testing.assert_close(final.deter, post_deter[:, -1])
    assert runtime.live_state is live


def test_matching_feature_flattens_categorical_state_then_appends_deter() -> None:
    runtime = _runtime()
    state = R2State(
        stoch=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        deter=torch.arange(16, 32, dtype=torch.float32).reshape(1, 16),
    )
    trajectory = R2Trajectory(stoch=state.stoch[:, None], deter=state.deter[:, None])
    expected = torch.arange(32, dtype=torch.float32).reshape(1, 32)

    torch.testing.assert_close(runtime.matching_feature(state), expected)
    torch.testing.assert_close(runtime.matching_features(trajectory), expected[:, None])


def test_cpu_compile_request_keeps_eager_parity() -> None:
    runtime = _runtime()
    live = runtime.observe(_observation(), previous_action=None)
    actions = torch.tensor([[[1.0, 20.0], [2.0, 25.0]]])
    eager = runtime.imagine_states(live, actions)

    assert runtime.compile_imagination(1, 2) == 0.0
    after_compile_request = runtime.imagine_states(live, actions)
    torch.testing.assert_close(after_compile_request.stoch, eager.stoch)
    torch.testing.assert_close(after_compile_request.deter, eager.deter)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_imagination_matches_eager() -> None:
    runtime = _runtime("cuda")
    live = runtime.observe(_observation("cuda"), previous_action=None)
    actions = torch.tensor([[[1.0, 20.0], [2.0, 25.0]]], device="cuda")
    eager = runtime.imagine_states(live, actions)

    runtime.compile_imagination(1, 2)
    compiled = runtime.imagine_states(live, actions)

    torch.testing.assert_close(compiled.stoch, eager.stoch, rtol=0, atol=0)
    torch.testing.assert_close(compiled.deter, eager.deter, rtol=1e-5, atol=1e-6)

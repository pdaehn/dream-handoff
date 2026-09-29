from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dream_handoff.inference.sampling import (
    SmolVLACandidateSampler,
    configure_rtc_prefix_guidance,
)


class RecordingPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chunk_size=5,
            max_action_dim=4,
            use_amp=False,
            rtc_config=SimpleNamespace(enabled=True),
        )
        self.rtc_processor = object()
        self.calls = 0
        self.batch = None
        self.noise = None
        self.kwargs = None

    def predict_action_chunk(self, batch, noise=None, **kwargs):
        self.calls += 1
        self.batch = batch
        self.noise = noise
        self.kwargs = kwargs
        return noise[:, :, :2].clone()


class RecordingProcessor:
    def __init__(self, *, offset: float = 0.0, mutate: bool = False) -> None:
        self.offset = offset
        self.mutate = mutate
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, value):
        self.calls += 1
        if torch.is_tensor(value):
            self.shapes.append(tuple(value.shape))
            if self.mutate:
                return value.add_(self.offset)
            return value + self.offset
        return value


def make_sampler():
    policy = RecordingPolicy()
    preprocessor = RecordingProcessor()
    postprocessor = RecordingProcessor(offset=10.0, mutate=True)
    sampler = SmolVLACandidateSampler(policy, preprocessor, postprocessor, device="cpu")
    return sampler, policy, preprocessor, postprocessor


def test_sampling_uses_one_n_batched_call_and_explicit_independent_noise() -> None:
    sampler, policy, preprocessor, postprocessor = make_sampler()
    policy_actions, control_actions = sampler.sample_candidate_bank(
        {"observation.state": torch.zeros(1, 2)}, 4, generator=torch.Generator().manual_seed(3)
    )
    assert policy.calls == preprocessor.calls == postprocessor.calls == 1
    assert policy.batch["observation.state"].shape == (4, 2)
    assert policy.noise.shape == (4, 5, 4)
    assert policy.noise.dtype == torch.float32
    assert not torch.equal(policy.noise[0], policy.noise[1])
    assert postprocessor.shapes == [(4, 5, 2)]
    torch.testing.assert_close(control_actions, policy_actions + 10.0)
    torch.testing.assert_close(policy_actions, policy.noise[:, :, :2])
    torch.testing.assert_close(sampler.last_noise, policy.noise)


def test_static_rtc_passes_same_index_policy_prefix_and_timing_without_expansion() -> None:
    sampler, policy, _, _ = make_sampler()
    prefix = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    sampler.sample_candidate_bank(
        {"observation.state": torch.zeros(1, 2)},
        4,
        noise=torch.zeros(4, 5, 4),
        prefix_actions=prefix,
        inference_delay=2,
        execution_horizon=3,
    )
    assert policy.batch["observation.state"].shape[0] == 4
    assert policy.kwargs == {
        "prev_chunk_left_over": prefix,
        "inference_delay": 2,
        "execution_horizon": 3,
    }


def test_sampling_rejects_malformed_noise_and_prefixes() -> None:
    sampler, _, _, _ = make_sampler()
    observation = {"observation.state": torch.zeros(1, 2)}
    with pytest.raises(ValueError, match="noise must have shape"):
        sampler.sample_candidate_bank(observation, 2, noise=torch.zeros(2, 4, 4))
    with pytest.raises(ValueError, match="execution_horizon"):
        sampler.sample_candidate_bank(
            observation,
            2,
            noise=torch.zeros(2, 5, 4),
            prefix_actions=torch.zeros(2, 3, 2),
            inference_delay=1,
            execution_horizon=2,
        )


def test_rtc_configuration_uses_pinned_lerobot_linear_schedule() -> None:
    policy = RecordingPolicy()

    def initialize() -> None:
        policy.rtc_processor = SimpleNamespace(config=policy.config.rtc_config)

    policy.init_rtc_processor = initialize
    config = configure_rtc_prefix_guidance(policy)
    assert config.enabled is True
    assert config.max_guidance_weight == 10.0
    assert config.execution_horizon == policy.config.chunk_size
    assert policy.rtc_processor.config is config

from __future__ import annotations

import json
from pathlib import Path

import torch

from dream_handoff.r2dreamer import R2DreamerRuntime
from dream_handoff.r2dreamer.action_normalization import R2DreamerActionNormalizer
from dream_handoff.r2dreamer.checkpoint import FINAL_R2_ARCHITECTURE_SIGNATURE
from dream_handoff.r2dreamer.config import (
    R2DreamerLossConfig,
    R2DreamerModelConfig,
    R2DreamerPreprocessingConfig,
    architecture_signature,
)
from dream_handoff.r2dreamer.losses import barlow_twins_loss, world_model_loss
from dream_handoff.r2dreamer.model import (
    R2DreamerModelOutput,
    R2DreamerWorldModel,
    R2State,
)
from dream_handoff.r2dreamer.optimization import LaProp, clip_grad_agc_
from dream_handoff.r2dreamer.replay import EpisodeReplayBatch
from dream_handoff.r2dreamer.training import R2Trainer, final_model_config, set_deterministic_seed


def tiny_config() -> R2DreamerModelConfig:
    return R2DreamerModelConfig(
        action_dim=2,
        image_channels=0,
        state_dim=3,
        image_size=(16, 16),
        stoch=4,
        deter=16,
        hidden=8,
        discrete=4,
        blocks=4,
        obs_layers=1,
        img_layers=2,
        dyn_layers=1,
        unimix_ratio=0.01,
        encoder_depth=2,
        encoder_mults=(1, 2),
        encoder_kernel_size=5,
        vector_layers=1,
        vector_units=8,
        activation="SiLU",
    )


def normalizer() -> R2DreamerActionNormalizer:
    return R2DreamerActionNormalizer(
        action_names=("joint_a", "joint_b"), q01=(-1.0, -1.0), q99=(1.0, 1.0)
    )


def fixed_batch() -> EpisodeReplayBatch:
    return EpisodeReplayBatch(
        observation={"state": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 17},
        previous_actions=torch.tensor(
            [[[0.0, 0.0], [0.1, -0.2], [0.3, 0.4]], [[0.0, 0.0], [-0.5, 0.6], [0.7, -0.8]]]
        ),
        is_first=torch.tensor([[True, False, False], [True, False, False]]),
        initial_state=R2State(torch.zeros(2, 4, 4), torch.zeros(2, 16)),
        row_indices=torch.arange(6).reshape(2, 3),
    )


def test_final_training_model_is_the_wp2a_checkpoint_architecture() -> None:
    config = final_model_config()
    model = R2DreamerWorldModel(config)
    assert architecture_signature(config) == FINAL_R2_ARCHITECTURE_SIGNATURE
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_678_944


def test_frozen_stochastic_forward_and_loss_parity() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/r2_training_loss_parity.json").read_text()
    )
    torch.manual_seed(fixture["model_seed"])
    model = R2DreamerWorldModel(tiny_config())
    batch = fixed_batch()
    torch.manual_seed(fixture["sample_seed"])
    output = model(
        batch.observation,
        batch.previous_actions,
        batch.is_first,
        initial_state=batch.initial_state,
        deterministic=False,
    )
    losses = world_model_loss(model, output, R2DreamerLossConfig())

    assert output.posterior_stoch.argmax(-1).tolist() == fixture["posterior_modes"]
    torch.testing.assert_close(
        output.embed.flatten()[:8], torch.tensor(fixture["embed_prefix"]), rtol=1e-6, atol=1e-6
    )
    for name in ("dynamics", "representation", "barlow", "total"):
        torch.testing.assert_close(
            getattr(losses, name), torch.tensor(fixture[name]), rtol=1e-6, atol=1e-6
        )


def test_kl_and_barlow_stop_gradients() -> None:
    model = R2DreamerWorldModel(tiny_config())
    posterior = torch.randn(2, 3, 4, 4, requires_grad=True)
    prior = torch.randn(2, 3, 4, 4, requires_grad=True)
    dynamics, representation = model.rssm.kl_loss(posterior, prior, free_bits=0.0)
    dynamics.sum().backward(retain_graph=True)
    assert posterior.grad is None and prior.grad is not None
    posterior.grad, prior.grad = None, None
    representation.sum().backward()
    assert posterior.grad is not None and prior.grad is None

    features = torch.randn(6, model.rssm.feat_size, requires_grad=True)
    embed = torch.randn(6, model.encoder.out_dim, requires_grad=True)
    output = R2DreamerModelOutput(
        embed,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        features,
    )
    barlow_twins_loss(model, output, 5e-4).backward()
    assert features.grad is not None and embed.grad is None


def test_laprop_and_agc_match_final_update_semantics() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    parameter.grad = torch.tensor([3.0])
    optimizer = LaProp([parameter], lr=0.1)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([1.9]))

    large = torch.nn.Parameter(torch.tensor([10.0]))
    small = torch.nn.Parameter(torch.tensor([0.0]))
    large.grad = torch.tensor([100.0])
    small.grad = torch.tensor([10.0])
    clip_grad_agc_([large, small])
    torch.testing.assert_close(large.grad, torch.tensor([3.0]))
    torch.testing.assert_close(small.grad, torch.tensor([3e-4]))


def make_trainer(seed: int) -> R2Trainer:
    set_deterministic_seed(seed)
    return R2Trainer(
        R2DreamerWorldModel(tiny_config()),
        preprocessing_config=R2DreamerPreprocessingConfig(
            image_keys=(), image_size=(16, 16), state_key="observation.state"
        ),
        action_normalizer=normalizer(),
        loss_config=R2DreamerLossConfig(),
        device="cpu",
        learning_rate=1e-3,
        warmup_steps=2,
    )


def test_checkpoint_runtime_roundtrip_and_interrupted_resume_parity(tmp_path: Path) -> None:
    uninterrupted = make_trainer(9)
    uninterrupted.step(fixed_batch())
    uninterrupted.step(fixed_batch())
    uninterrupted_result = uninterrupted.step(fixed_batch())

    interrupted = make_trainer(9)
    interrupted.step(fixed_batch())
    interrupted.step(fixed_batch())
    checkpoint = interrupted.save(
        tmp_path / "arbitrary-name.pt",
        metadata={
            "architecture_signature": architecture_signature(tiny_config()),
            "action_metadata": {"names": list(normalizer().action_names)},
            "action_normalization": normalizer().to_dict(),
        },
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert {"optimizer_state_dict", "scheduler_state_dict", "trainer_state"} <= payload.keys()
    assert R2DreamerRuntime.from_checkpoint(checkpoint, "cpu").action_names == (
        "joint_a",
        "joint_b",
    )

    resumed = make_trainer(1234)
    resumed.resume(checkpoint)
    resumed_result = resumed.step(fixed_batch())

    assert resumed.step_number == uninterrupted.step_number == 3
    assert resumed.sequences_seen == uninterrupted.sequences_seen == 6
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert resumed_result.metrics == uninterrupted_result.metrics

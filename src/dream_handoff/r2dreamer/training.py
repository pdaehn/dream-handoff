"""Focused offline trainer for the final R2-Dreamer s12-r64 world model."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .action_normalization import R2DreamerActionNormalizer
from .checkpoint import (
    R2_UPSTREAM_COMMIT,
    read_r2dreamer_checkpoint,
    save_r2dreamer_checkpoint,
)
from .config import (
    R2DreamerLossConfig,
    R2DreamerModelConfig,
    R2DreamerPreprocessingConfig,
    R2DreamerTrainingConfig,
    architecture_signature,
)
from .data import (
    FINAL_DATASET_REPO_ID,
    fit_action_normalizer,
    load_final_lerobot_datasets,
    read_final_episode_split,
)
from .losses import R2DreamerLosses, world_model_loss
from .model import R2DreamerModelOutput, R2DreamerWorldModel
from .optimization import LaProp, clip_grad_agc_
from .replay import EpisodeReplayBatch, EpisodeTensorStore, load_or_materialize_episode_stores

OPEN_LOOP_VALIDATION_HORIZONS = (1, 2, 4, 8, 16, 32)
FINAL_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
FINAL_PREPROCESSING_CONFIG = R2DreamerPreprocessingConfig()
FINAL_LOSS_CONFIG = R2DreamerLossConfig()


def final_model_config() -> R2DreamerModelConfig:
    """Return the exact checkpoint architecture established in WP2A."""
    return R2DreamerModelConfig(
        action_dim=6,
        image_channels=6,
        state_dim=6,
        image_size=(64, 64),
        stoch=32,
        deter=2048,
        hidden=256,
        discrete=16,
        blocks=8,
        obs_layers=1,
        img_layers=2,
        dyn_layers=1,
        unimix_ratio=0.01,
        encoder_depth=16,
        encoder_mults=(2, 3, 4, 4),
        encoder_kernel_size=5,
        vector_layers=3,
        vector_units=256,
        activation="SiLU",
    )


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingStepResult:
    losses: R2DreamerLosses
    output: R2DreamerModelOutput
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class WorldModelTrainingResult:
    latest_checkpoint: Path
    metrics_path: Path
    resolved_config_path: Path
    final_metrics: dict[str, float | int]


class R2Trainer:
    """One concrete LaProp trainer shared by production and deterministic tests."""

    def __init__(
        self,
        model: R2DreamerWorldModel,
        *,
        preprocessing_config: R2DreamerPreprocessingConfig,
        action_normalizer: R2DreamerActionNormalizer,
        loss_config: R2DreamerLossConfig,
        device: torch.device | str,
        learning_rate: float = 4e-5,
        warmup_steps: int = 1000,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        self.model = model.to(self.device)
        self.preprocessing_config = preprocessing_config
        self.action_normalizer = action_normalizer
        self.loss_config = loss_config
        self.parameters = list(self.model.parameters())
        self.optimizer = LaProp(
            self.parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-20,
            weight_decay=0.0,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0,
        )
        # The final run had AMP disabled. Keeping the disabled scaler preserves
        # the historical step ordering and gives checkpoints one resume shape.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        self.step_number = 0
        self.sequences_seen = 0
        self._resume_rng_state: dict[str, Any] | None = None

    def step(self, batch: EpisodeReplayBatch) -> TrainingStepResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(
            batch.observation,
            batch.previous_actions,
            batch.is_first,
            initial_state=batch.initial_state,
            deterministic=False,
        )
        losses = world_model_loss(self.model, output, self.loss_config)
        if not bool(torch.isfinite(losses.total)):
            raise FloatingPointError(f"non-finite R2 loss at step {self.step_number + 1}")
        self.scaler.scale(losses.total).backward()
        self.scaler.unscale_(self.optimizer)
        gradients = [parameter.grad for parameter in self.parameters if parameter.grad is not None]
        gradient_norm = torch.nn.utils.get_total_norm(gradients)
        clip_grad_agc_(self.parameters, clip=0.3, pmin=1e-3)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step_number += 1
        self.sequences_seen += int(batch.previous_actions.shape[0])
        metrics: dict[str, float | int] = {
            "step": self.step_number,
            **losses.detached_metrics(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "gradient_norm": float(gradient_norm.detach()),
            "sequences": self.sequences_seen,
        }
        return TrainingStepResult(losses, output, metrics)

    def _trainer_state(self, replay: EpisodeTensorStore | None) -> dict[str, Any]:
        return {
            "sequences_seen": self.sequences_seen,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "scaler_state_dict": self.scaler.state_dict(),
            "replay_sampler_state": None if replay is None else replay.sampler_state_dict(),
        }

    def save(
        self,
        path: Path | str,
        *,
        metadata: dict[str, Any],
        replay: EpisodeTensorStore | None = None,
    ) -> Path:
        return save_r2dreamer_checkpoint(
            path,
            model=self.model,
            preprocessing_config=self.preprocessing_config,
            action_normalizer=self.action_normalizer,
            step=self.step_number,
            metadata=metadata,
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            trainer_state=self._trainer_state(replay),
        )

    def resume(self, path: Path | str, *, replay: EpisodeTensorStore | None = None) -> None:
        payload = read_r2dreamer_checkpoint(path, map_location=self.device)
        if R2DreamerModelConfig.from_dict(payload["model_config"]) != self.model.config:
            raise ValueError("resume checkpoint model config differs from the training model")
        if (
            R2DreamerPreprocessingConfig.from_dict(payload["preprocessing_config"])
            != self.preprocessing_config
        ):
            raise ValueError("resume checkpoint preprocessing config differs")
        if (
            R2DreamerActionNormalizer.from_dict(payload["action_normalization"])
            != self.action_normalizer
        ):
            raise ValueError("resume checkpoint action normalization differs")
        if "optimizer_state_dict" not in payload or "scheduler_state_dict" not in payload:
            raise ValueError(
                "resume requires a full latest checkpoint with optimizer and scheduler"
            )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.step_number = int(payload["step"])
        state = payload.get("trainer_state")
        if state is None:
            # Historical latest.pt retained model/optimizer/scheduler only.
            self.sequences_seen = int(
                payload.get("metadata", {}).get("metrics", {}).get("sequences", 0)
            )
            return
        self.sequences_seen = int(state["sequences_seen"])
        self.scaler.load_state_dict(state.get("scaler_state_dict", {}))
        self._resume_rng_state = state
        self.restore_rng_state()
        replay_state = state.get("replay_sampler_state")
        if replay is not None and replay_state is not None:
            replay.load_sampler_state_dict(replay_state)

    def restore_rng_state(self) -> None:
        """Restore saved sampling RNG after any resume-time cache rebuild."""
        if self._resume_rng_state is None:
            return
        torch.set_rng_state(self._resume_rng_state["torch_rng_state"].cpu())
        cuda_states = self._resume_rng_state.get("cuda_rng_state_all", [])
        if cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


@torch.no_grad()
def evaluate_episode_store(
    model: R2DreamerWorldModel,
    store: EpisodeTensorStore,
    loss_config: R2DreamerLossConfig,
    *,
    batch_size: int,
    max_batches: int,
) -> dict[str, float]:
    refresh = store.refresh_latent_cache(model, deterministic=True)
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in store.evaluation_batches(batch_size=batch_size, max_batches=max_batches):
        output = model(
            batch.observation,
            batch.previous_actions,
            batch.is_first,
            initial_state=batch.initial_state,
            deterministic=True,
        )
        for name, value in world_model_loss(model, output, loss_config).detached_metrics().items():
            metric = f"validation/{name.removeprefix('loss/')}"
            totals[metric] = totals.get(metric, 0.0) + value
        count += 1
    if not count:
        raise ValueError("validation store produced no batches")
    open_loop = store.open_loop_feature_errors(
        model, horizons=OPEN_LOOP_VALIDATION_HORIZONS, batch_size=batch_size
    )
    return {
        **{name: total / count for name, total in totals.items()},
        **{
            f"validation/open_loop_feature_error_h{horizon}": value
            for horizon, value in open_loop.items()
        },
        "validation/cache_refresh_seconds": refresh["cache/refresh_seconds"],
    }


def _feature_shape(features: dict[str, Any], key: str) -> tuple[int, ...]:
    try:
        return tuple(features[key]["shape"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"dataset feature {key!r} has no usable shape") from exc


def _write_resolved_config(
    path: Path,
    *,
    training: R2DreamerTrainingConfig,
    action_normalizer: R2DreamerActionNormalizer,
) -> None:
    payload = {
        "source": "DreamHandoff final R2 training configuration",
        "dataset_repo_id": FINAL_DATASET_REPO_ID,
        "model": final_model_config().to_dict(),
        "preprocessing": FINAL_PREPROCESSING_CONFIG.to_dict(),
        "loss": FINAL_LOSS_CONFIG.to_dict(),
        "training": {
            **training.to_dict(),
            "optimizer": "laprop",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-20,
            "weight_decay": 0.0,
            "agc_clip": 0.3,
            "agc_pmin": 1e-3,
            "gradient_clip_norm": None,
            "amp": False,
        },
        "action_normalization": action_normalizer.to_dict(),
        "architecture_signature": architecture_signature(final_model_config()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_world_model(config: R2DreamerTrainingConfig) -> WorldModelTrainingResult:
    """Run the final offline path against the immutable episode split."""
    split = read_final_episode_split(config.split_path)
    train_dataset, validation_dataset, dataset = load_final_lerobot_datasets(
        dataset_root=config.dataset_root,
        split=split,
        sequence_length=config.sequence_length,
        image_keys=FINAL_PREPROCESSING_CONFIG.image_keys,
        image_size=FINAL_PREPROCESSING_CONFIG.image_size,
        state_key=FINAL_PREPROCESSING_CONFIG.state_key,
        video_backend=config.video_backend,
    )
    features = dict(dataset.features)
    if _feature_shape(features, "action") != (6,):
        raise ValueError("final R2 dataset action feature must have shape [6]")
    if _feature_shape(features, "observation.state") != (6,):
        raise ValueError("final R2 dataset state feature must have shape [6]")
    action_names = tuple(features["action"].get("names") or ())
    if action_names != FINAL_ACTION_NAMES:
        raise ValueError("final R2 dataset canonical action names or order differ")
    normalizer = fit_action_normalizer(train_dataset, action_names=action_names)

    set_deterministic_seed(config.seed)
    model = R2DreamerWorldModel(final_model_config())
    trainer = R2Trainer(
        model,
        preprocessing_config=FINAL_PREPROCESSING_CONFIG,
        action_normalizer=normalizer,
        loss_config=FINAL_LOSS_CONFIG,
        device=config.device,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
    )
    cache_dir = config.cache_dir or config.output_dir / "cache"
    train_store, validation_store, cache_path, cache_hit = load_or_materialize_episode_stores(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        preprocessing_config=FINAL_PREPROCESSING_CONFIG,
        action_normalizer=normalizer,
        model_config=model.config,
        device=config.device,
        storage_device=config.replay_storage_device,
        seed=config.seed,
        dataset_root=config.dataset_root,
        cache_dir=cache_dir,
        cache_preprocessed=config.cache_preprocessed_dataset,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = config.output_dir / "resolved_config.json"
    _write_resolved_config(resolved_config_path, training=config, action_normalizer=normalizer)

    if config.resume_path is not None:
        trainer.resume(config.resume_path, replay=train_store)
        if trainer.step_number >= config.steps:
            raise ValueError("resume checkpoint step must be below the configured maximum steps")
        # Frozen checkpoints did not retain the large posterior cache. Rebuild it
        # with the resumed model, then restore training RNG so Gumbel samples continue.
        train_store.refresh_latent_cache(model, deterministic=False)
        trainer.restore_rng_state()
    else:
        train_store.refresh_latent_cache(model, deterministic=False)

    latest_path = config.output_dir / "checkpoints" / "latest.pt"
    metrics_path = config.output_dir / "metrics.jsonl"
    metadata = {
        "dataset_repo_id": FINAL_DATASET_REPO_ID,
        "dataset_features": features,
        "action_key": "action",
        "action_metadata": features["action"],
        "action_normalization": normalizer.to_dict(),
        "train_episode_indices": list(split.train_episode_indices),
        "validation_episode_indices": list(split.validation_episode_indices),
        "episode_split_sha256": split.split_sha256,
        "episode_metadata_sha256": split.episode_metadata_sha256,
        "loss_config": FINAL_LOSS_CONFIG.to_dict(),
        "architecture_signature": architecture_signature(model.config),
        "r2_upstream_commit": R2_UPSTREAM_COMMIT,
        "data_pipeline": {
            "mode": "preloaded_episode_tensor_store",
            "requested_num_workers": config.num_workers,
            "effective_num_workers": 0,
            "window_weighting": "uniform_valid_start_timestep",
            "latent_context": "preceding_cached_posterior",
            "latent_full_refresh": "initial_and_each_sampler_epoch",
            "preprocessed_cache_enabled": cache_path is not None,
            "preprocessed_cache_hit": cache_hit,
        },
    }
    final_metrics: dict[str, float | int] = {}
    pending_cache_metrics: dict[str, float] = {}
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        while trainer.step_number < config.steps:
            batch = train_store.next_batch(config.batch_size)
            result = trainer.step(batch)
            train_store.update_latents(batch.row_indices, result.output)
            if batch.epoch_finished:
                pending_cache_metrics = train_store.refresh_latent_cache(model, deterministic=False)
            metrics = {
                **result.metrics,
                **pending_cache_metrics,
                "cache/age_steps": train_store.cache_age_steps,
                "cache/epochs_completed": train_store.epochs_completed,
            }
            pending_cache_metrics = {}
            if trainer.step_number % config.validation_interval == 0 or (
                trainer.step_number == config.steps
            ):
                metrics.update(
                    evaluate_episode_store(
                        model,
                        validation_store,
                        FINAL_LOSS_CONFIG,
                        batch_size=config.batch_size,
                        max_batches=config.validation_batches,
                    )
                )
            if trainer.step_number % config.checkpoint_interval == 0 or (
                trainer.step_number == config.steps
            ):
                trainer.save(
                    latest_path,
                    metadata={**metadata, "metrics": metrics},
                    replay=train_store,
                )
            metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            print(json.dumps(metrics, sort_keys=True), flush=True)
            final_metrics = metrics
    return WorldModelTrainingResult(
        latest_checkpoint=latest_path,
        metrics_path=metrics_path,
        resolved_config_path=resolved_config_path,
        final_metrics=final_metrics,
    )

"""Episode-local offline replay with the final R2 cached-posterior semantics."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .action_normalization import R2DreamerActionNormalizer
from .config import R2DreamerModelConfig, R2DreamerPreprocessingConfig
from .data import EpisodeSequenceDataset
from .model import R2DreamerModelOutput, R2DreamerWorldModel, R2State
from .preprocessing import R2DreamerObservationPreprocessor

_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class EpisodeReplayBatch:
    observation: dict[str, torch.Tensor]
    previous_actions: torch.Tensor
    is_first: torch.Tensor
    initial_state: R2State
    row_indices: torch.Tensor
    epoch_finished: bool = False


class EpisodeTensorStore:
    """Preprocessed episode frames and cached posterior states on chosen devices."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        sequence_length: int,
        model_config: R2DreamerModelConfig,
        device: torch.device | str,
        storage_device: torch.device | str,
        seed: int,
    ) -> None:
        self.device = self._concrete_device(device)
        self.storage_device = self._concrete_device(storage_device)
        self.sequence_length = sequence_length
        self.model_config = model_config
        self.observations = {
            key: torch.as_tensor(value).to(self.storage_device)
            for key, value in payload["observations"].items()
        }
        self.previous_actions = torch.as_tensor(
            payload["previous_actions"], dtype=torch.float32, device=self.device
        )
        self.is_first = torch.as_tensor(payload["is_first"], dtype=torch.bool, device=self.device)
        self.episode_index = torch.as_tensor(
            payload["episode_index"], dtype=torch.long, device=self.device
        )
        self.frame_index = torch.as_tensor(
            payload["frame_index"], dtype=torch.long, device=self.device
        )
        self.episode_ids = tuple(int(value) for value in payload["episode_ids"])
        self.episode_lengths = tuple(int(value) for value in payload["episode_lengths"])
        if len(self.episode_ids) != len(self.episode_lengths):
            raise ValueError("episode ids and lengths must have equal sizes")
        self.frame_count = sum(self.episode_lengths)
        if self.frame_count <= 0:
            raise ValueError("episode tensor store cannot be empty")
        values = [
            *self.observations.values(),
            self.previous_actions,
            self.is_first,
            self.episode_index,
            self.frame_index,
        ]
        if any(value.shape[0] != self.frame_count for value in values):
            raise ValueError("all episode tensor fields must share the frame dimension")

        slices: list[tuple[int, int]] = []
        starts: list[int] = []
        offset = 0
        for episode_id, length in zip(self.episode_ids, self.episode_lengths, strict=True):
            if length < sequence_length:
                raise ValueError(f"episode {episode_id} is shorter than sequence length")
            stop = offset + length
            slices.append((offset, stop))
            starts.extend(range(offset, stop - sequence_length + 1))
            if not bool(self.is_first[offset]) or bool(self.is_first[offset + 1 : stop].any()):
                raise ValueError(f"episode {episode_id} must mark only its first frame")
            if not bool((self.episode_index[offset:stop] == episode_id).all()):
                raise ValueError(f"episode metadata is inconsistent for episode {episode_id}")
            offset = stop
        self.episode_slices = tuple(slices)
        self.window_starts = torch.tensor(starts, dtype=torch.long, device=self.device)
        self.window_count = len(starts)
        self.cached_stoch = torch.zeros(
            self.frame_count,
            model_config.stoch,
            model_config.discrete,
            device=self.device,
        )
        self.cached_deter = torch.zeros(self.frame_count, model_config.deter, device=self.device)
        self.latent_valid = torch.zeros(self.frame_count, dtype=torch.bool, device=self.device)
        generator_device = self.device if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(seed)
        self.evaluation_seed = seed + 10_000
        self._permutation: torch.Tensor | None = None
        self._cursor = 0
        self.epochs_completed = 0
        self.cache_age_steps = 0

    @staticmethod
    def _concrete_device(value: torch.device | str) -> torch.device:
        device = torch.device(value)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def sampler_state_dict(self) -> dict[str, Any]:
        """Return the small replay ordering state retained on resume."""
        return {
            "generator_state": self.generator.get_state().cpu(),
            "permutation": None if self._permutation is None else self._permutation.cpu(),
            "cursor": self._cursor,
            "epochs_completed": self.epochs_completed,
        }

    def load_sampler_state_dict(self, state: dict[str, Any]) -> None:
        self.generator.set_state(torch.as_tensor(state["generator_state"]).cpu())
        permutation = state.get("permutation")
        self._permutation = (
            None if permutation is None else torch.as_tensor(permutation, device=self.device)
        )
        self._cursor = int(state["cursor"])
        self.epochs_completed = int(state["epochs_completed"])
        if self._permutation is not None and not 0 <= self._cursor <= self.window_count:
            raise ValueError("resumed replay cursor is outside its permutation")

    def _new_permutation(self) -> None:
        self._permutation = torch.randperm(
            self.window_count, generator=self.generator, device=self.device
        )
        self._cursor = 0

    def next_batch(self, batch_size: int) -> EpisodeReplayBatch:
        """Sample valid start timesteps uniformly without replacement per epoch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._permutation is None:
            self._new_permutation()
        assert self._permutation is not None
        stop = min(self._cursor + batch_size, self.window_count)
        indices = self._permutation[self._cursor : stop]
        self._cursor = stop
        finished = stop == self.window_count
        batch = self.batch_from_window_indices(indices, epoch_finished=finished)
        if finished:
            self._permutation = None
            self.epochs_completed += 1
        return batch

    def batch_from_window_indices(
        self, window_indices: torch.Tensor, *, epoch_finished: bool = False
    ) -> EpisodeReplayBatch:
        indices = torch.as_tensor(window_indices, dtype=torch.long, device=self.device).reshape(-1)
        if not indices.numel():
            raise ValueError("window indices must not be empty")
        if bool((indices < 0).any()) or bool((indices >= self.window_count).any()):
            raise IndexError("window index is outside the episode tensor store")
        starts = self.window_starts[indices]
        rows = starts[:, None] + torch.arange(self.sequence_length, device=self.device)[None, :]
        episode_first = self.is_first[starts]
        preceding = starts - 1
        mid_episode = ~episode_first
        if bool(mid_episode.any()) and not bool(self.latent_valid[preceding[mid_episode]].all()):
            raise RuntimeError("mid-episode window lacks its preceding cached posterior")
        batch_size = len(starts)
        initial_stoch = torch.zeros(
            batch_size,
            self.model_config.stoch,
            self.model_config.discrete,
            device=self.device,
        )
        initial_deter = torch.zeros(batch_size, self.model_config.deter, device=self.device)
        if bool(mid_episode.any()):
            context = preceding[mid_episode]
            initial_stoch[mid_episode] = self.cached_stoch[context]
            initial_deter[mid_episode] = self.cached_deter[context]
        storage_rows = rows.to(self.storage_device)
        return EpisodeReplayBatch(
            observation={
                key: value[storage_rows].to(self.device) for key, value in self.observations.items()
            },
            previous_actions=self.previous_actions[rows],
            is_first=self.is_first[rows],
            initial_state=R2State(initial_stoch.detach(), initial_deter.detach()),
            row_indices=rows,
            epoch_finished=epoch_finished,
        )

    def evaluation_batches(
        self, *, batch_size: int, max_batches: int
    ) -> Iterator[EpisodeReplayBatch]:
        limit = min(self.window_count, batch_size * max_batches)
        indices = self._representative_evaluation_indices(limit)
        for start in range(0, limit, batch_size):
            yield self.batch_from_window_indices(indices[start : start + batch_size])

    def _representative_evaluation_indices(self, limit: int) -> torch.Tensor:
        if limit >= self.window_count:
            return torch.arange(self.window_count, device=self.device)
        counts = [length - self.sequence_length + 1 for length in self.episode_lengths]
        ideal = [limit * count / self.window_count for count in counts]
        quotas = [int(value) for value in ideal]
        remaining = limit - sum(quotas)
        order = sorted(
            range(len(counts)),
            key=lambda index: (ideal[index] - quotas[index], counts[index]),
            reverse=True,
        )
        for index in order[:remaining]:
            quotas[index] += 1
        if limit >= len(counts):
            for empty in (index for index, quota in enumerate(quotas) if quota == 0):
                donors = [index for index, quota in enumerate(quotas) if quota > 1]
                if not donors:
                    break
                donor = max(donors, key=lambda index: quotas[index] - ideal[index])
                quotas[donor] -= 1
                quotas[empty] = 1
        selected: list[int] = []
        offset = 0
        for count, quota in zip(counts, quotas, strict=True):
            selected.extend(
                offset + min(count - 1, ((2 * phase + 1) * count) // (2 * quota))
                for phase in range(quota)
            )
            offset += count
        generator = torch.Generator().manual_seed(self.evaluation_seed)
        shuffled = torch.tensor(selected)[torch.randperm(len(selected), generator=generator)]
        return shuffled.to(self.device)

    @torch.no_grad()
    def update_latents(self, row_indices: torch.Tensor, output: R2DreamerModelOutput) -> None:
        """Update overlaps deterministically, preferring the greatest window phase."""
        batch_size, sequence_length = row_indices.shape
        flat_rows = row_indices.reshape(-1)
        batch = torch.arange(batch_size, device=self.device)[:, None]
        phase = torch.arange(sequence_length, device=self.device)[None, :]
        priorities = (phase * batch_size + batch).reshape(-1)
        selected = torch.full((self.frame_count,), -1, dtype=torch.long, device=self.device)
        selected.scatter_reduce_(0, flat_rows, priorities, reduce="amax", include_self=True)
        target = torch.nonzero(selected >= 0, as_tuple=False).squeeze(-1)
        chosen = selected[target]
        chosen_batch = torch.remainder(chosen, batch_size)
        chosen_phase = torch.div(chosen, batch_size, rounding_mode="floor")
        self.cached_stoch[target] = output.posterior_stoch[chosen_batch, chosen_phase].detach()
        self.cached_deter[target] = output.posterior_deter[chosen_batch, chosen_phase].detach()
        self.latent_valid[target] = True
        self.cache_age_steps += 1

    @torch.no_grad()
    def refresh_latent_cache(
        self,
        model: R2DreamerWorldModel,
        *,
        deterministic: bool,
        chunk_length: int = 256,
    ) -> dict[str, float]:
        """Recompute every episode posterior chronologically."""
        was_training = model.training
        model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        valid_before = self.latent_valid.clone()
        for start, stop in self.episode_slices:
            stoch, deter = model.rssm.initial(1, device=self.device)
            initial = R2State(stoch, deter)
            for chunk_start in range(start, stop, chunk_length):
                chunk_stop = min(chunk_start + chunk_length, stop)
                rows = slice(chunk_start, chunk_stop)
                output = model(
                    {
                        key: value[rows].to(self.device).unsqueeze(0)
                        for key, value in self.observations.items()
                    },
                    self.previous_actions[rows].unsqueeze(0),
                    self.is_first[rows].unsqueeze(0),
                    initial_state=initial,
                    deterministic=deterministic,
                )
                self.cached_stoch[rows] = output.posterior_stoch[0]
                self.cached_deter[rows] = output.posterior_deter[0]
                self.latent_valid[rows] = True
                initial = R2State(
                    output.posterior_stoch[0, -1:].detach(),
                    output.posterior_deter[0, -1:].detach(),
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if was_training:
            model.train()
        self.cache_age_steps = 0
        return {
            "cache/refresh_seconds": time.perf_counter() - started,
            "cache/valid_fraction_before_refresh": float(valid_before.float().mean()),
        }

    @torch.no_grad()
    def open_loop_feature_errors(
        self,
        model: R2DreamerWorldModel,
        *,
        horizons: tuple[int, ...],
        batch_size: int,
    ) -> dict[int, float]:
        maximum = max(horizons)
        origins = torch.tensor(
            [
                origin
                for start, stop in self.episode_slices
                for origin in range(start, stop - maximum)
            ],
            device=self.device,
        )
        if not origins.numel():
            return {}
        if not bool(self.latent_valid.all()):
            raise RuntimeError("posterior cache must be complete before open-loop validation")
        sums = {
            horizon: torch.zeros((), dtype=torch.float64, device=self.device)
            for horizon in horizons
        }
        offsets = torch.arange(1, maximum + 1, device=self.device)
        for start in range(0, len(origins), batch_size):
            selected = origins[start : start + batch_size]
            action_rows = selected[:, None] + offsets[None, :]
            stoch, deter = model.rssm.imagine_with_action(
                self.cached_stoch[selected],
                self.cached_deter[selected],
                self.previous_actions[action_rows],
                deterministic=True,
            )
            imagined = model.rssm.get_feat(stoch, deter)
            for horizon in horizons:
                target = model.rssm.get_feat(
                    self.cached_stoch[selected + horizon], self.cached_deter[selected + horizon]
                )
                sums[horizon] += (
                    torch.linalg.vector_norm(imagined[:, horizon - 1] - target, dim=-1)
                    .double()
                    .sum()
                )
        return {horizon: float(total / len(origins)) for horizon, total in sums.items()}


def materialize_episode_dataset(
    dataset: EpisodeSequenceDataset,
    preprocessing_config: R2DreamerPreprocessingConfig,
    action_normalizer: R2DreamerActionNormalizer,
) -> dict[str, Any]:
    """Decode and preprocess each selected frame once."""
    preprocessor = R2DreamerObservationPreprocessor(preprocessing_config, device="cpu")
    episode_ids = dataset.episode_indices
    episode_lengths = tuple(len(dataset.episode_rows[index]) for index in episode_ids)
    frame_count = sum(episode_lengths)
    lookup = {
        (window.episode_index, window.start_offset): index
        for index, window in enumerate(dataset.windows)
    }
    observations: dict[str, torch.Tensor] | None = None
    previous_actions: torch.Tensor | None = None
    is_first = torch.empty(frame_count, dtype=torch.bool)
    episode_index = torch.empty(frame_count, dtype=torch.long)
    frame_index = torch.empty(frame_count, dtype=torch.long)
    destination_offset = 0
    for episode_id, episode_length in zip(episode_ids, episode_lengths, strict=True):
        copy_start = 0
        while copy_start < episode_length:
            sample_start = min(copy_start, episode_length - dataset.sequence_length)
            sample = dataset[lookup[(episode_id, sample_start)]]
            prepared = {
                key: value.cpu().contiguous() for key, value in preprocessor(sample).items()
            }
            if observations is None:
                observations = {
                    key: torch.empty(frame_count, *value.shape[1:], dtype=value.dtype)
                    for key, value in prepared.items()
                }
                previous_actions = torch.empty(
                    frame_count, *sample["previous_action"].shape[1:], dtype=torch.float32
                )
            local_start = copy_start - sample_start
            count = min(dataset.sequence_length - local_start, episode_length - copy_start)
            source = slice(local_start, local_start + count)
            destination = slice(
                destination_offset + copy_start, destination_offset + copy_start + count
            )
            assert observations is not None and previous_actions is not None
            for key, value in prepared.items():
                observations[key][destination] = value[source]
            previous_actions[destination] = sample["previous_action"][source]
            is_first[destination] = sample["is_first"][source]
            episode_index[destination] = episode_id
            frame_index[destination] = sample["frame_index"][source]
            copy_start += count
        destination_offset += episode_length
    if observations is None or previous_actions is None:
        raise ValueError("cannot materialize an empty episode dataset")
    previous_actions = action_normalizer.normalize(previous_actions)
    previous_actions[is_first] = 0
    return {
        "observations": observations,
        "previous_actions": previous_actions,
        "is_first": is_first,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "episode_ids": episode_ids,
        "episode_lengths": episode_lengths,
    }


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for directory_name in ("meta", "data", "videos"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            stat = path.stat()
            digest.update(str(path.relative_to(root)).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def load_or_materialize_episode_stores(
    *,
    train_dataset: EpisodeSequenceDataset,
    validation_dataset: EpisodeSequenceDataset,
    preprocessing_config: R2DreamerPreprocessingConfig,
    action_normalizer: R2DreamerActionNormalizer,
    model_config: R2DreamerModelConfig,
    device: torch.device | str,
    storage_device: torch.device | str,
    seed: int,
    dataset_root: Path,
    cache_dir: Path,
    cache_preprocessed: bool,
) -> tuple[EpisodeTensorStore, EpisodeTensorStore, Path | None, bool]:
    """Load or rebuild the optional path-independent preprocessing cache."""
    fingerprint_data = {
        "schema": _CACHE_SCHEMA,
        "source_files": _source_fingerprint(dataset_root),
        "sequence_length": train_dataset.sequence_length,
        "preprocessing": preprocessing_config.to_dict(),
        "action_normalization": action_normalizer.to_dict(),
        "train_episodes": train_dataset.episode_indices,
        "validation_episodes": validation_dataset.episode_indices,
    }
    canonical = json.dumps(
        fingerprint_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    path = cache_dir / f"r2-preprocessed-v{_CACHE_SCHEMA}-{fingerprint[:16]}.pt"
    payload: dict[str, Any] = {}
    hit = False
    if cache_preprocessed and path.is_file():
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        if candidate.get("schema") == _CACHE_SCHEMA and candidate.get("fingerprint") == fingerprint:
            payload = candidate
            hit = True
    if not payload:
        payload = {
            "schema": _CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "train": materialize_episode_dataset(
                train_dataset, preprocessing_config, action_normalizer
            ),
            "validation": materialize_episode_dataset(
                validation_dataset, preprocessing_config, action_normalizer
            ),
        }
        if cache_preprocessed:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
    common = {
        "sequence_length": train_dataset.sequence_length,
        "model_config": model_config,
        "device": device,
        "storage_device": storage_device,
    }
    train = EpisodeTensorStore(payload["train"], seed=seed, **common)
    validation = EpisodeTensorStore(payload["validation"], seed=seed + 1, **common)
    return train, validation, path if cache_preprocessed else None, hit

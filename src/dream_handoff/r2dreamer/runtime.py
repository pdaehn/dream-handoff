"""Concrete recurrent runtime for the final DreamHandoff R2 world model."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .action_normalization import R2DreamerActionNormalizer
from .checkpoint import load_r2dreamer_checkpoint
from .model import R2DreamerWorldModel, R2State, R2Trajectory
from .preprocessing import R2DreamerObservationPreprocessor


class R2DreamerRuntime:
    """Maintain one recurrent posterior and branch deterministic imagination from it."""

    def __init__(
        self,
        model: R2DreamerWorldModel,
        preprocessor: R2DreamerObservationPreprocessor,
        action_normalizer: R2DreamerActionNormalizer,
        *,
        device: torch.device | str,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor
        if action_normalizer.action_dim != model.config.action_dim:
            raise ValueError("action normalization dimension does not match the R2 model")
        self.action_normalizer = action_normalizer
        self._action_q01 = torch.tensor(
            action_normalizer.q01, dtype=torch.float32, device=self.device
        )
        q99 = torch.tensor(action_normalizer.q99, dtype=torch.float32, device=self.device)
        denominator = q99 - self._action_q01
        self._action_denominator = torch.where(
            denominator == 0,
            denominator.new_full((), action_normalizer.epsilon),
            denominator,
        )
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self._live_state: R2State | None = None
        self._compiled_imagine: Any | None = None
        self._compiled_imagine_shape: tuple[int, int] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        device: torch.device | str,
    ) -> R2DreamerRuntime:
        """Build the runtime directly from an explicitly supplied compatible checkpoint."""
        loaded = load_r2dreamer_checkpoint(path, device=device)
        return cls(
            loaded.model,
            loaded.preprocessor,
            loaded.action_normalizer,
            device=device,
            checkpoint_metadata=loaded.metadata,
        )

    @property
    def live_state(self) -> R2State | None:
        return self._live_state

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.action_normalizer.action_names

    @property
    def observation_keys(self) -> tuple[str, ...]:
        config = self.preprocessor.config
        state = () if config.state_key is None else (config.state_key,)
        return (*config.image_keys, *state)

    def reset(self) -> None:
        """Clear the recurrent posterior at an episode boundary."""
        self.preprocessor.reset()
        self._live_state = None

    def compile_imagination(self, num_candidates: int, horizon: int) -> float:
        """Compile and warm one fixed CUDA imagination shape before control starts.

        CPU keeps the eager implementation because it has no CUDA launch overhead.
        A shape other than the warmed shape also remains eager, avoiding compilation
        in the control loop.
        """
        if num_candidates <= 0 or horizon <= 0:
            raise ValueError(
                "compiled imagination shape must be positive, got "
                f"num_candidates={num_candidates}, horizon={horizon}"
            )
        if self.device.type != "cuda":
            return 0.0
        started_at = time.perf_counter()
        rssm = self.model.rssm

        def imagine(stoch: Tensor, deter: Tensor, actions: Tensor):
            return rssm.imagine_with_action(stoch, deter, actions)

        compiled = torch.compile(imagine, dynamic=False)
        with torch.inference_mode():
            warm_stoch, warm_deter = rssm.initial(num_candidates, device=self.device)
            warm_actions = torch.zeros(
                num_candidates,
                horizon,
                self.model.config.action_dim,
                dtype=torch.float32,
                device=self.device,
            )
            compiled(warm_stoch, warm_deter, warm_actions)
        torch.cuda.synchronize(self.device)
        self._compiled_imagine = compiled
        self._compiled_imagine_shape = (int(num_candidates), int(horizon))
        return time.perf_counter() - started_at

    @torch.inference_mode()
    def observe(
        self,
        observation: Mapping[str, Any],
        previous_action: Tensor | None,
    ) -> R2State:
        """Advance the observation-conditioned posterior by one control tick."""
        processed = self._runtime_batch(self.preprocessor(observation))
        embed = self.model.encoder(processed)
        if embed.ndim != 2 or embed.shape[0] != 1:
            raise ValueError(f"live R2 encoder must produce [1,E], got {tuple(embed.shape)}")

        if self._live_state is None:
            stoch, deter = self.model.rssm.initial(1, device=self.device)
            reset = torch.ones(1, dtype=torch.bool, device=self.device)
        else:
            stoch = self._live_state.stoch.to(self.device)
            deter = self._live_state.deter.to(self.device)
            reset = torch.zeros(1, dtype=torch.bool, device=self.device)

        if previous_action is None:
            action = torch.zeros(
                1,
                self.model.config.action_dim,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            action = previous_action.detach().to(device=self.device, dtype=torch.float32)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            if tuple(action.shape) != (1, self.model.config.action_dim):
                raise ValueError(
                    "previous canonical action must have shape [A] or [1,A], got "
                    f"{tuple(action.shape)}"
                )
            action = self._normalize_actions(action)

        stoch, deter, _ = self.model.rssm.obs_step(stoch, deter, action, embed, reset)
        self._live_state = R2State(stoch=stoch.detach(), deter=deter.detach())
        return self._live_state

    @torch.inference_mode()
    def imagine_states(self, start_state: R2State, actions: Tensor) -> R2Trajectory:
        """Imagine candidates and return H states aligned before their H actions."""
        start_stoch, start_deter, post_stoch, post_deter = self._post_action_states(
            start_state, actions
        )
        if actions.shape[1] == 1:
            aligned_stoch = start_stoch[:, None]
            aligned_deter = start_deter[:, None]
        else:
            aligned_stoch = torch.cat([start_stoch[:, None], post_stoch[:, :-1]], dim=1)
            aligned_deter = torch.cat([start_deter[:, None], post_deter[:, :-1]], dim=1)
        return R2Trajectory(stoch=aligned_stoch, deter=aligned_deter)

    @torch.inference_mode()
    def imagine_final_state(self, start_state: R2State, actions: Tensor) -> R2State:
        """Return the deterministic prior state after consuming every action."""
        _, _, post_stoch, post_deter = self._post_action_states(start_state, actions)
        return R2State(stoch=post_stoch[:, -1], deter=post_deter[:, -1])

    def _post_action_states(
        self, start_state: R2State, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(start_state, R2State):
            raise TypeError("start_state must be an R2State")
        if not torch.is_tensor(actions) or actions.ndim != 3:
            raise ValueError("R2 actions must have shape [N,H,A]")
        if actions.shape[-1] != self.model.config.action_dim:
            raise ValueError("R2 action dimension does not match checkpoint")
        if actions.shape[0] == 0 or actions.shape[1] == 0:
            raise ValueError("R2 actions must contain at least one candidate and action")
        actions = self._normalize_actions(
            actions.detach().to(device=self.device, dtype=torch.float32)
        )
        candidates = int(actions.shape[0])
        start_stoch = self._single_state(start_state.stoch, state_dimensions=2)
        start_deter = self._single_state(start_state.deter, state_dimensions=1)
        start_stoch = start_stoch.to(self.device).expand(candidates, -1, -1).clone()
        start_deter = start_deter.to(self.device).expand(candidates, -1).clone()
        shape = candidates, int(actions.shape[1])
        if self._compiled_imagine is not None and self._compiled_imagine_shape == shape:
            post_stoch, post_deter = self._compiled_imagine(start_stoch, start_deter, actions)
        else:
            post_stoch, post_deter = self.model.rssm.imagine_with_action(
                start_stoch, start_deter, actions
            )
        return start_stoch, start_deter, post_stoch, post_deter

    def matching_feature(self, state: R2State) -> Tensor:
        """Flatten categorical stochastic state, then append deterministic state."""
        if not isinstance(state, R2State):
            raise TypeError("state must be an R2State")
        return self.model.rssm.get_feat(state.stoch, state.deter)

    def matching_features(self, trajectory: R2Trajectory) -> Tensor:
        """Return matching features with leading shape ``[N,H]``."""
        if not isinstance(trajectory, R2Trajectory):
            raise TypeError("trajectory must be an R2Trajectory")
        return self.model.rssm.get_feat(trajectory.stoch, trajectory.deter)

    def _runtime_batch(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        if "images" in observation:
            images = observation["images"]
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim != 4 or images.shape[0] != 1:
                raise ValueError("live R2 images must represent one observation")
            if images.shape[-1] != self.model.config.image_channels:
                raise ValueError("live R2 image channels do not match checkpoint")
            result["images"] = images
        if "state" in observation:
            state = observation["state"]
            if state.ndim == 1:
                state = state.unsqueeze(0)
            if state.ndim != 2 or state.shape[0] != 1:
                raise ValueError("live R2 state must represent one observation")
            if state.shape[-1] != self.model.config.state_dim:
                raise ValueError("live R2 vector-state dimension does not match checkpoint")
            result["state"] = state
        return result

    def _normalize_actions(self, actions: Tensor) -> Tensor:
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("canonical R2 actions contain NaN or Inf")
        return 2.0 * (actions - self._action_q01) / self._action_denominator - 1.0

    @staticmethod
    def _single_state(value: Tensor, *, state_dimensions: int) -> Tensor:
        if value.ndim == state_dimensions:
            return value.unsqueeze(0)
        if value.ndim == state_dimensions + 1 and value.shape[0] == 1:
            return value
        raise ValueError("start_state must contain one live R2 posterior")

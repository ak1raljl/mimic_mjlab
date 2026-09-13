"""MLP model helpers for frame-stacked tracking observations."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState


class FrameStackMLPModel(MLPModel):
    """Flatten frame-major observations before normalization and the actor MLP.

    MjLab keeps an unflattened observation history in ``[batch, history, features]``
    order. Flattening the trailing dimensions here produces
    ``[frame_oldest, ..., frame_newest]``, which matches the deployment buffer.

    The inherited JIT and ONNX exporters already consume a pre-flattened vector,
    so exported policies keep the conventional ``[batch, history * features]``
    input shape.
    """

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Build the normalized MLP input from frame-major observation groups."""
        del masks, hidden_state
        return self.obs_normalizer(self._flatten_obs_groups(obs))

    def update_normalization(self, obs: TensorDict) -> None:
        """Update normalization statistics with flattened frame histories."""
        if self.obs_normalization:
            self.obs_normalizer.update(self._flatten_obs_groups(obs))

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        """Return the flattened size of all non-batch observation dimensions."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for group in active_obs_groups:
            if obs[group].ndim < 2:
                raise ValueError(
                    "FrameStackMLPModel expects observations with a batch dimension, "
                    f"got shape {obs[group].shape} for '{group}'."
                )
            obs_dim += math.prod(obs[group].shape[1:])
        return active_obs_groups, obs_dim

    def _flatten_obs_groups(self, obs: TensorDict) -> torch.Tensor:
        """Flatten trailing dimensions and concatenate configured groups."""
        obs_list = [obs[group].flatten(start_dim=1) for group in self.obs_groups]
        return torch.cat(obs_list, dim=-1)

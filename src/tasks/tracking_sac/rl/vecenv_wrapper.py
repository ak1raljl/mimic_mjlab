"""VecEnv wrapper for FlashSAC training on mjlab environments.

Ported from the FlashSAC Isaac Lab wrapper (``rsl_rl_flashsac/isaaclab_flashsac/wrapper.py``,
BSD-3-Clause, Holiday Robotics) to mjlab's ``RslRlVecEnvWrapper``.

Two responsibilities:

1. Affine action scaling. The FlashSAC actor, replay buffer and critic all operate on
   normalized [-1, 1] tanh-space actions; this wrapper maps them to environment action
   units at the env boundary (``bias + scale * clamp(a, -1, 1)``). The same mapping is
   baked into the exported ONNX policy, so deployment does not need this wrapper.
2. Terminal-observation caching. mjlab resets done environments inside ``step()``, so the
   returned observation of a done env is already the post-reset frame. FlashSAC
   bootstraps truncated (timed-out) episodes from the true final observation; this
   wrapper caches it by intercepting ``_reset_idx`` — the last point where the simulator
   still holds the terminal state — and republishes it as ``extras["time_outs_obs"]``.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper


class TrackingSacVecEnvWrapper(RslRlVecEnvWrapper):
    """RslRlVecEnvWrapper exposing ``time_outs_obs`` and action bounds for FlashSAC.

    ``action_bias``/``action_scale`` are consumed by ``FlashSAC.construct_algorithm``
    (temperature target derivation) and applied to the policy's normalized actions in
    ``step()``.
    """

    action_bias: torch.Tensor | None
    action_scale: torch.Tensor | None

    def __init__(
        self,
        env: Any,
        clip_actions: float | None = None,
        action_bound: str = "scalar",
        action_bound_scale: float = 3.0,
    ) -> None:
        super().__init__(env, clip_actions)

        self._final_obs_buf: dict[str, torch.Tensor] | None = None
        base_env = self.unwrapped
        if hasattr(base_env, "_reset_idx") and hasattr(base_env, "observation_manager"):
            base_env._reset_idx = partial(self._reset_idx_with_final_obs, base_env, base_env._reset_idx)
        else:
            warnings.warn(
                "TrackingSacVecEnvWrapper: the environment exposes no _reset_idx/observation_manager;"
                " extras['time_outs_obs'] will not be provided and truncated episodes bootstrap"
                " from post-reset observations.",
                stacklevel=2,
            )

        self.action_bias = None
        self.action_scale = None
        self.configure_action_scaling(action_bound, action_bound_scale)

    def configure_action_scaling(self, action_bound: str, action_bound_scale: float) -> None:
        """(Re)compute the affine action scaling; the runner re-applies the rl-cfg values."""
        if action_bound == "scalar":
            # +-action_bound_scale bounds.
            self.action_bias = torch.zeros(self.num_actions, device=self.device)
            self.action_scale = torch.full((self.num_actions,), float(action_bound_scale), device=self.device)
            return
        if action_bound != "joint_limit":
            raise ValueError(f"Unknown action_bound '{action_bound}' (expected 'scalar' or 'joint_limit').")
        bias, scale = self._compute_joint_limit_scaling()
        if bias is None or scale is None:
            warnings.warn(
                "TrackingSacVecEnvWrapper: joint-limit action scaling unavailable;"
                f" falling back to scalar bounds (±{action_bound_scale}).",
                stacklevel=2,
            )
            bias = torch.zeros(self.num_actions, device=self.device)
            scale = torch.full((self.num_actions,), float(action_bound_scale), device=self.device)
        self.action_bias, self.action_scale = bias, scale

    def _compute_joint_limit_scaling(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Zero-bias symmetric per-joint range from the soft joint position limits.

        The range is the max distance from the default pose to the soft limits, divided by
        the action manager's action scale, so that tanh(0) is exactly the default pose and
        ±1 reaches the farther soft limit (the nearer one can be overshot).
        """
        base_env = self.unwrapped
        scene = getattr(base_env, "scene", None)
        if scene is None:
            return None, None
        try:
            robot = scene["robot"]
        except Exception:
            return None, None
        data = robot.data
        lower_limits = data.soft_joint_pos_limits[0, :, 0]
        upper_limits = data.soft_joint_pos_limits[0, :, 1]
        default_pos = data.default_joint_pos[0]
        if default_pos.numel() != self.num_actions:
            return None, None

        # Action scale from the action manager: scalar, or per-joint (dict cfg ->
        # the term's applied ``_scale`` tensor, scattered onto robot joint order).
        term_action_scale: float | torch.Tensor = 1.0
        action_manager = getattr(base_env, "action_manager", None)
        if action_manager is not None:
            for term in action_manager._terms.values():
                if not hasattr(term.cfg, "scale"):
                    continue
                if isinstance(term.cfg.scale, (float, int)):
                    term_action_scale = float(term.cfg.scale)
                    break
                applied = getattr(term, "_scale", None)
                if isinstance(applied, torch.Tensor):
                    vec = torch.ones_like(default_pos)
                    target_ids = getattr(term, "_target_ids", None)
                    row = applied[0].to(device=default_pos.device, dtype=default_pos.dtype)
                    if target_ids is None or isinstance(target_ids, slice):
                        vec = row.clone()
                    else:
                        vec[torch.as_tensor(list(target_ids), device=default_pos.device)] = row
                    term_action_scale = vec
                    break

        upper = torch.abs(upper_limits - default_pos) / term_action_scale
        lower = torch.abs(lower_limits - default_pos) / term_action_scale
        action_bias = torch.zeros_like(upper)
        action_scale = torch.maximum(upper, lower)

        # Joints without finite soft limits (e.g. continuous joints) fall back to identity.
        finite = torch.isfinite(action_scale) & (action_scale > 0)
        if not bool(finite.all()):
            warnings.warn(
                f"TrackingSacVecEnvWrapper: {int((~finite).sum())} joint(s) have no finite soft"
                " position limits; using identity action scaling for them.",
                stacklevel=2,
            )
            action_scale = torch.where(finite, action_scale, torch.ones_like(action_scale))
        return action_bias, action_scale

    def _reset_idx_with_final_obs(self, base_env: Any, orig_reset_idx: Any, env_ids: Any, *args: Any, **kwargs: Any) -> Any:
        """Cache the terminal observations of the environments about to be reset."""
        if env_ids is None:
            env_ids = slice(None)
        terminal = base_env.observation_manager.compute()
        # Only flat (concatenated) observation groups are cached; nested term dicts are skipped.
        terminal = {key: value for key, value in terminal.items() if isinstance(value, torch.Tensor)}
        if self._final_obs_buf is None:
            self._final_obs_buf = {key: torch.zeros_like(value) for key, value in terminal.items()}
        for key, value in terminal.items():
            self._final_obs_buf[key][env_ids] = value[env_ids]
        return orig_reset_idx(env_ids, *args, **kwargs)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment, adding ``time_outs_obs`` to the extras when available.

        The policy emits normalized [-1, 1] actions; the affine scaling is applied here,
        at the env boundary, as ``bias + scale * clamp(a, -1, 1)``.
        """
        if (
            self.action_bias is not None
            and self.action_scale is not None
            and self.action_scale.numel() == actions.shape[-1]
        ):
            actions = self.action_bias + self.action_scale * torch.clamp(actions, -1.0, 1.0)
        obs, rew, dones, extras = super().step(actions)
        if "time_outs" in extras and self._final_obs_buf is not None:
            extras["time_outs_obs"] = TensorDict(
                {key: value.clone() for key, value in self._final_obs_buf.items()},
                batch_size=[self.num_envs],
            )
        return obs, rew, dones, extras

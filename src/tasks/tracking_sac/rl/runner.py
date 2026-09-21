"""Motion-tracking runner for FlashSAC (off-policy).

Extends the vendored ``rsl_rl.runners.off_policy_runner.OffPolicyRunner`` with the mjlab
tracking extras: env-state persistence (``common_step_counter``), ``upload_model`` gating,
and ONNX export of the policy plus the reference-motion bundle on every save.
"""

import os
from typing import cast

import torch
from torch import nn

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.tracking.mdp import MotionCommand

from rsl_rl.runners.off_policy_runner import OffPolicyRunner

from .vecenv_wrapper import TrackingSacVecEnvWrapper


class _OnnxMotionModel(nn.Module):
    """ONNX-exportable model that wraps the policy and bundles motion reference data."""

    def __init__(self, actor, motion):
        super().__init__()
        self.policy = actor.as_onnx(verbose=False)
        self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
        self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
        self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
        self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
        self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
        self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
        self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.policy(x),
            self.joint_pos[time_step_clamped],  # type: ignore[index]
            self.joint_vel[time_step_clamped],  # type: ignore[index]
            self.body_pos_w[time_step_clamped],  # type: ignore[index]
            self.body_quat_w[time_step_clamped],  # type: ignore[index]
            self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
            self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
        )


class MotionTrackingOffPolicyRunner(OffPolicyRunner):
    """Off-policy runner with mjlab tracking-specific save/load extras."""

    #: Env wrapper class selected by the ``*_wrapper_cls`` hook in train.py / play.py.
    env_wrapper_cls = TrackingSacVecEnvWrapper

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Apply the cfg's action scaling to the wrapper before the algorithm reads
        # env.action_bias/action_scale in construct_algorithm (via super().__init__).
        if isinstance(env, TrackingSacVecEnvWrapper):
            env.configure_action_scaling(
                train_cfg.get("action_bound", "scalar"),
                train_cfg.get("action_bound_scale", 3.0),
            )
        super().__init__(env, train_cfg, log_dir, device)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export policy to ONNX format using legacy export path (dynamo=False)."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
            dynamic_axes={},
            dynamo=False,
        )

    def export_motion_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        os.makedirs(path, exist_ok=True)
        cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
        model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
        model.to("cpu")
        model.eval()
        obs = torch.zeros(1, model.policy.input_size)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            model,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
            dynamo=False,
        )

    def save(self, path: str, infos: dict | None = None) -> None:
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["wall_time"] = self._elapsed_wall_time()
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        self.export_motion_policy_to_onnx(policy_path, filename)
        self.export_policy_to_onnx(policy_path, "policy.onnx")
        metadata = get_base_metadata(self.env.unwrapped, "local")
        motion_term = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
        metadata.update(
            {
                "anchor_body_name": motion_term.cfg.anchor_body_name,
                "body_names": list(motion_term.cfg.body_names),
            }
        )
        attach_metadata_to_onnx(os.path.join(policy_path, filename), metadata)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos

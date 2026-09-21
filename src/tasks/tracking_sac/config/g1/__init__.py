"""Unitree G1 tracking task trained with FlashSAC (off-policy)."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.tracking_sac.rl import MotionTrackingOffPolicyRunner

from .env_cfgs import unitree_g1_flat_tracking_sac_env_cfg
from .rl_cfg import unitree_g1_tracking_sac_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-Tracking-No-State-Estimation-SAC",
    env_cfg=unitree_g1_flat_tracking_sac_env_cfg(has_state_estimation=False),
    play_env_cfg=unitree_g1_flat_tracking_sac_env_cfg(has_state_estimation=False, play=True),
    rl_cfg=unitree_g1_tracking_sac_runner_cfg(),
    runner_cls=MotionTrackingOffPolicyRunner,
)

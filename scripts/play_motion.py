"""Motion playback script for validating NPZ files."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass
class RobotConfig:
    env_cfg_task_id: str
    joint_names: list[str]

ROBOT_CONFIGS = {
    "g1": RobotConfig(
        env_cfg_task_id="Unitree-G1-Tracking",
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    ),
    # "h1": RobotConfig(
    #     env_cfg_task_id="Unitree-H1-Tracking",
    #     joint_names=[...],
    # ),
}


@dataclass(frozen=True)
class PlayConfig:
    robot: str
    motion_file: str
    loop: bool = True
    viewer: Literal["auto", "native", "viser"] = "auto"
    device: str | None = None

class NPZMotionLoader:
    """Loads and manages motion data from NPZ files."""
    def __init__(self, motion_file: str, device: str, loop: bool = True):
        self.motion_file = motion_file
        self.device = device
        self.loop = loop
        self.current_frame = 0

        data = np.load(motion_file)

        self.fps = float(data["fps"][0])
        self.joint_pos = torch.from_numpy(data["joint_pos"]).to(device).float()
        self.joint_vel = torch.from_numpy(data["joint_vel"]).to(device).float()
        self.body_pos_w = torch.from_numpy(data["body_pos_w"]).to(device).float()
        self.body_quat_w = torch.from_numpy(data["body_quat_w"]).to(device).float()

        self.num_frames = self.joint_pos.shape[0]
        self.duration = self.num_frames / self.fps

        print(f"[INFO] Loaded motion: {self.num_frames} frames @ {self.fps} Hz ({self.duration:.2f}s)")

    def get_frame(self, frame_idx: int | None = None):
        if frame_idx is None:
            frame_idx = self.current_frame

        root_pos = self.body_pos_w[frame_idx, 0]  # (3,)
        root_quat = self.body_quat_w[frame_idx, 0]  # (4,) in wxyz

        joint_pos = self.joint_pos[frame_idx]  # (num_joints,)
        joint_vel = self.joint_vel[frame_idx]  # (num_joints,)

        return root_pos, root_quat, joint_pos, joint_vel

    def step(self) -> bool:
        self.current_frame += 1

        if self.current_frame >= self.num_frames:
            if self.loop:
                self.current_frame = 0
                return False
            else:
                self.current_frame = self.num_frames - 1
                return True  # Signal end

        return False

    def reset(self):
        self.current_frame = 0


class MotionPlaybackEnv:
    """Lightweight environment for motion playback that mimics RL env interface."""
    def __init__(
        self, 
        scene: Scene, 
        sim: Simulation, 
        robot: Entity,  
        motion: NPZMotionLoader, 
        robot_joint_indices: torch.Tensor,
        decimation: int = 4
    ):
        self.scene = scene
        self.sim = sim
        self.robot = robot
        self.motion = motion
        self.robot_joint_indices = robot_joint_indices
        self.paused = False
        self.single_step_mode = False
        self.decimation = decimation

        self._decimation_counter = 0

        self.num_envs = 1
        self.unwrapped = self 
        self.step_dt = self.sim.mj_model.opt.timestep
        from mjlab.viewer.viewer_config import ViewerConfig
        from dataclasses import dataclass

        @dataclass
        class MinimalCfg:
            viewer: ViewerConfig
            decimation: int = 1

        self.cfg = MinimalCfg(viewer=ViewerConfig())

        class DummyRewardManager:
            def get_active_iterable_terms(self, env_idx=None):
                return []

        self.reward_manager = DummyRewardManager()

        class DummyCommandManager:
            active_terms: list = []

            def create_gui(self, server, env_idx_getter=None):
                pass

            def create_debug_vis_gui(self, server):
                pass

        self.command_manager = DummyCommandManager()

    def step(self, action=None):
        """Execute one simulation step with motion data."""
        if self.paused and not self.single_step_mode:
            return None, 0.0, False, False, {}

        self.single_step_mode = False

        self._decimation_counter += 1
        should_update_frame = (self._decimation_counter >= self.decimation)

        if should_update_frame:
            self._decimation_counter = 0
            should_end = self.motion.step()
        else:
            should_end = False
        
        root_pos, root_quat, joint_pos, joint_vel = self.motion.get_frame()

        root_states = self.robot.data.default_root_state.clone()
        root_states[:, 0:3] = root_pos.unsqueeze(0)
        root_states[:, :2] += self.scene.env_origins[:, :2]
        root_states[:, 3:7] = root_quat.unsqueeze(0)
        self.robot.write_root_state_to_sim(root_states)

        joint_pos_full = self.robot.data.default_joint_pos.clone()
        joint_vel_full = self.robot.data.default_joint_vel.clone()
        joint_pos_full[:, self.robot_joint_indices] = joint_pos.unsqueeze(0)
        joint_vel_full[:, self.robot_joint_indices] = joint_vel.unsqueeze(0)
        self.robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

        self.sim.forward()
        self.scene.update(self.sim.mj_model.opt.timestep)
        return None, 0.0, should_end, False, {}

    def reset(self, seed=None, options=None):
        self.motion.reset()
        self.scene.reset()
        return None, {}

    def toggle_pause(self):
        self.paused = not self.paused

    def single_step(self):
        if self.paused:
            self.single_step_mode = True

    def get_observations(self):
        return None

    def close(self):
        pass


def resolve_viewer(cfg: PlayConfig) -> str:
    """Resolve viewer type based on config and environment."""
    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        return "native" if has_display else "viser"
    return cfg.viewer


def create_scene_and_sim(robot_cfg: RobotConfig, device: str):
    """Create scene and simulation for the robot."""
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(robot_cfg.env_cfg_task_id, play=True)

    env_cfg.scene.num_envs = 1
    decimation = env_cfg.decimation
    scene = Scene(env_cfg.scene, device=device)
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = env_cfg.sim.mujoco.timestep

    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    return scene, sim, decimation


class DummyPolicy:
    def __call__(self, obs):
        return None


def run_play_motion(cfg: PlayConfig):
    """Main execution function for motion playback."""
    configure_torch_backends()

    if cfg.robot not in ROBOT_CONFIGS:
        available = ", ".join(ROBOT_CONFIGS.keys())
        raise ValueError(f"Unsupported robot: {cfg.robot}. Available: {available}")

    motion_path = Path(cfg.motion_file)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {cfg.motion_file}")

    robot_cfg = ROBOT_CONFIGS[cfg.robot]
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    scene, sim, decimation = create_scene_and_sim(robot_cfg, device)
    robot: Entity = scene["robot"]

    robot_joint_indices = robot.find_joints(robot_cfg.joint_names, preserve_order=True)[0]

    print("[INFO] Loading motion data...")
    motion = NPZMotionLoader(cfg.motion_file, device=device, loop=cfg.loop)
    env = MotionPlaybackEnv(scene, sim, robot, motion, robot_joint_indices, decimation=decimation)

    env.reset()
    policy = DummyPolicy()

    viewer_type = resolve_viewer(cfg)
    try:
        if viewer_type == "native":
            viewer = NativeMujocoViewer(env, policy)
            viewer.run()
        elif viewer_type == "viser":
            print("Web viewer started at http://localhost:8080")
            print("Use the web interface to control playback")
            print("Press Ctrl+C to exit")
            viewer = ViserPlayViewer(env, policy)
            viewer.run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {viewer_type}")
    finally:
        env.close()
        print("[INFO] Viewer closed")


def main():
    # Import tasks to populate the registry
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    cfg = tyro.cli(
        PlayConfig,
        description="Motion playback for NPZ validation",
    )

    run_play_motion(cfg)


if __name__ == "__main__":
    main()

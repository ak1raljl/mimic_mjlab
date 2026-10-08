"""Record synchronized PPO/SAC tracking with MuJoCo-native error figures.

Uses the same task registry, runners, actor loading, and action wrappers as play.py.
Both policies start at reference frame zero under fixed physics, without resets.
Rollouts are saved before rendering so the video and CSV use identical samples.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tracking_comparison_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tracking_comparison_cache")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from mjlab.viewer.native.viewer import make_empty_figure


TASK = "Unitree-G1-Tracking-No-State-Estimation"
METRICS = (
    ("error_anchor_pos", "Torso position error (m)", 0.10),
    ("error_body_pos", "Body position error (m)", 0.10),
    ("error_body_rot", "Body orientation error (rad)", 0.50),
    ("error_joint_pos", "Joint position L2 error (rad)", 1.00),
)
COLORS = ((0.20, 0.75, 1.00), (1.00, 0.60, 0.20))


def make_playback(task_id: str, checkpoint: Path, motion: Path, device: str, seed: int):
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = seed
    env_cfg.events = {}
    env_cfg.terminations = {}
    env_cfg.episode_length_s = int(1e9)
    env_cfg.observations["actor"].enable_corruption = False
    motion_cfg = env_cfg.commands["motion"]
    motion_cfg.motion_file = str(motion)
    motion_cfg.sampling_mode = "start"
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)

    # Replay storage and training compilation are unused during actor inference.
    if hasattr(agent_cfg.algorithm, "replay_buffer_size"):
        agent_cfg.algorithm.replay_buffer_size = 64
        agent_cfg.algorithm.buffer_min_length = 32
        agent_cfg.algorithm.mini_batch_size = 16
        agent_cfg.algorithm.use_compile = False

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        runner_cls = load_runner_cls(task_id)
        wrapper_cls = getattr(runner_cls, "env_wrapper_cls", RslRlVecEnvWrapper)
        env = wrapper_cls(base_env, clip_actions=agent_cfg.clip_actions)
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        command = base_env.command_manager.get_term("motion")
        # Populate aligned reference poses at frame zero without advancing time.
        command.time_steps.sub_(1)
        command._update_command()
        obs = env.get_observations()
        print(f"[READY] {task_id}: actor input {tuple(obs['actor'].shape)}", flush=True)
        return env, policy, obs
    except BaseException:
        base_env.close()
        raise


def collect_rollouts(playbacks, frame_count: int):
    tracks = []
    for env, _, _ in playbacks:
        base = env.unwrapped
        tracks.append({
            "qpos": np.empty((frame_count, base.sim.mj_model.nq), dtype=np.float32),
            "qvel": np.empty((frame_count, base.sim.mj_model.nv), dtype=np.float32),
            "metrics": np.empty((frame_count, len(METRICS)), dtype=np.float64),
        })
    torch.testing.assert_close(
        playbacks[0][0].unwrapped.sim.data.qpos,
        playbacks[1][0].unwrapped.sim.data.qpos,
        rtol=0.0,
        atol=1e-6,
    )
    observations = [p[2] for p in playbacks]
    start = time.monotonic()
    with torch.inference_mode():
        for frame in range(frame_count):
            for side, (env, policy, _) in enumerate(playbacks):
                base = env.unwrapped
                command = base.command_manager.get_term("motion")
                if int(command.time_steps[0]) != frame:
                    raise RuntimeError(f"Reference lost synchronization at {frame}, side {side}")
                # CommandManager computes metrics before advancing the reference.
                # Refresh them here to pair each rendered state with its own frame.
                command._update_metrics()
                tracks[side]["qpos"][frame] = base.sim.data.qpos[0].cpu().numpy()
                tracks[side]["qvel"][frame] = base.sim.data.qvel[0].cpu().numpy()
                tracks[side]["metrics"][frame] = [float(command.metrics[k][0]) for k, _, _ in METRICS]
                if not np.isfinite(tracks[side]["metrics"][frame]).all():
                    raise RuntimeError(f"Non-finite tracking error at frame {frame}, side {side}")
            if frame % 500 == 0 or frame == frame_count - 1:
                print(f"[ROLLOUT] {frame + 1}/{frame_count}; elapsed {time.monotonic() - start:.1f}s", flush=True)
            if frame + 1 < frame_count:
                for side, (env, policy, _) in enumerate(playbacks):
                    observations[side], _, dones, _ = env.step(policy(observations[side]))
                    if bool(dones.any()):
                        raise RuntimeError("Unexpected reset during synchronized recording")
    return tracks


def save_measurements(output: Path, tracks, dt: float, args):
    count = len(tracks[0]["qpos"])
    with (output / "tracking_errors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "reference_frame"] + [f"{label}_{k}" for label in ("PPO", "SAC") for k, _, _ in METRICS])
        for i in range(count):
            writer.writerow([i * dt, i, *tracks[0]["metrics"][i], *tracks[1]["metrics"][i]])
    summary = {
        "motion": str(args.motion_file.resolve()),
        "checkpoints": {"PPO": str(args.ppo_checkpoint.resolve()), "SAC": str(args.sac_checkpoint.resolve())},
        "tasks": {"PPO": TASK, "SAC": TASK + "-SAC"},
        "seed": args.seed,
        "frames": count,
        "fps": 1.0 / dt,
        "video_duration_s": count * dt,
        "protocol": "Reference frame zero, identical initial state, nominal physics, no observation noise, pushes, startup randomization, or automatic resets. Deterministic actor inference with each task's original action wrapper.",
        "definitions": {
            "error_anchor_pos": "Euclidean world position error of torso_link (m).",
            "error_body_pos": "Mean Euclidean error over 14 tracked bodies after the task's reference alignment: robot anchor x/y and relative yaw; reference anchor z (m).",
            "error_body_rot": "Mean quaternion angular distance over 14 tracked bodies with the same yaw alignment (rad).",
            "error_joint_pos": "L2 norm of the 29 joint position errors (rad), not per-joint RMSE.",
        },
        "mean_errors": {label: {k: float(tracks[s]["metrics"][:, j].mean()) for j, (k, _, _) in enumerate(METRICS)} for s, label in enumerate(("PPO", "SAC"))},
        "plot_history_s": args.history_seconds,
        "plot_scales": "Both algorithms share the same y range per metric at every frame; ranges only expand and never clip a previous sample.",
        "camera": ("Same camera angle and distance following each actual robot. The ghost uses the task's torso x/y and yaw alignment; world drift remains in the torso-position plot."
            if args.camera_mode == "robot" else "Common world-space camera framing the reference and both actual robots together."),
    }
    (output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "rollouts.npz", dt=dt, **{f"{label}_{key}": value for label, track in zip(("PPO", "SAC"), tracks) for key, value in track.items()})
    return summary


def render_video(output: Path, tracks, base_env, dt: float, args):
    width, height = args.width, args.height
    panel = width // 2
    header, footer = 78, 32
    chart_height = int(height * 0.40)
    chart_bottom = footer
    scene_bottom = chart_bottom + chart_height
    scene_height = height - scene_bottom - header
    if scene_height < 200:
        raise ValueError("Video height is too small for the scene and figures")

    model = copy.deepcopy(base_env.sim.mj_model)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.stat.extent = 5.0
    model.vis.global_.fovy = 45.0
    ghost_model = copy.deepcopy(model)
    ghost_model.geom_rgba[:] = (0.40, 0.85, 0.50, 0.25)
    ghost_model.geom_contype[:] = 0
    ghost_model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene["robot"]
    free_adr = robot.indexing.free_joint_q_adr.cpu().numpy()
    joint_adr = robot.indexing.joint_q_adr.cpu().numpy()
    ref_pos = command.motion.body_pos_w[:, 0].cpu().numpy()
    ref_quat = command.motion.body_quat_w[:, 0].cpu().numpy()
    ref_joint = command.motion.joint_pos.cpu().numpy()
    anchor_index = command.motion_anchor_body_index
    ref_anchor_pos = command.motion.body_pos_w[:, anchor_index].cpu().numpy()
    ref_anchor_quat = command.motion.body_quat_w[:, anchor_index].cpu().numpy()
    anchor_body_id = robot.indexing.bodies[command.robot_anchor_body_index].id
    tracked_body_ids = [robot.indexing.bodies[j].id for j in command.body_indexes.cpu().tolist()]
    origin = base_env.scene.env_origins[0].cpu().numpy()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    figures = [[make_empty_figure(title, (5, 3), (0.0, minimum), 2, 1.0) for _, title, minimum in METRICS] for _ in range(2)]
    for side in range(2):
        for fig in figures[side]:
            fig.flg_extend = 0
            fig.flg_legend = 0
            fig.flg_ticklabel[:] = 1
            fig.figurergba[:] = (0.035, 0.045, 0.065, 1.0)
            fig.panergba[:] = (0.06, 0.075, 0.10, 1.0)
            fig.gridrgb[:] = (0.22, 0.26, 0.30)
            fig.linergb[0] = COLORS[side]
            fig.linewidth = 2.0
            fig.xlabel = "Time (s)"
            fig.xformat = "%.1f"
            fig.yformat = "%.2f"
    count = len(tracks[0]["qpos"])
    history = max(2, round(args.history_seconds / dt))
    limits = np.array([m[2] for m in METRICS])
    frame_buffer = np.empty((height, width, 3), dtype=np.uint8)
    viewport = mujoco.MjrRect(0, 0, width, height)
    video_path = output / "ppo_left_sac_right.mp4"
    start = time.monotonic()
    with mujoco.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
        video_path, fps=1.0 / dt, codec="libx264", quality=8,
        macro_block_size=2, ffmpeg_params=["-preset", "fast", "-movflags", "+faststart"],
    ) as writer:
        context = renderer._mjr_context
        mujoco.mjr_changeFont(mujoco.mjtFontScale.mjFONTSCALE_100, context)
        data.qpos[:] = tracks[0]["qpos"][0]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        forward = np.array(renderer.scene.camera[0].forward, dtype=np.float64)
        up = np.array(renderer.scene.camera[0].up, dtype=np.float64)
        right = np.cross(forward, up)
        forward /= np.linalg.norm(forward)
        up /= np.linalg.norm(up)
        right /= np.linalg.norm(right)
        tangent_y = math.tan(math.radians(model.vis.global_.fovy) / 2)
        tangent_x = tangent_y * (panel - 8) / scene_height
        # Conservative body bounds around each pelvis include feet and extended arms.
        offsets = np.array([(x, y, z) for x in (-0.75, 0.75)
            for y in (-0.75, 0.75) for z in (-0.9, 0.9)])
        for i in range(count):
            mujoco.mjr_rectangle(viewport, 0.025, 0.035, 0.055, 1.0)
            limits = np.maximum(limits, np.maximum(tracks[0]["metrics"][i], tracks[1]["metrics"][i]) * 1.12)
            first = max(0, i - history + 1)
            # MuJoCo figures support at most 1001 points per line.
            indices = np.unique(np.linspace(first, i, min(i - first + 1, 1001), dtype=int))
            roots = np.stack((ref_pos[i] + origin,
                tracks[0]["qpos"][i, free_adr[:3]], tracks[1]["qpos"][i, free_adr[:3]]))
            target = (roots.min(axis=0) + roots.max(axis=0)) / 2
            target[2] = 0.85
            bounds = (roots[:, None, :] + offsets[None, :, :]).reshape(-1, 3) - target
            depth = bounds @ forward
            required_distance = max(args.camera_distance,
                float(np.max(np.abs(bounds @ right) / tangent_x - depth)),
                float(np.max(np.abs(bounds @ up) / tangent_y - depth))) + 0.10
            camera.lookat[:] = target
            camera.distance = max(required_distance, 0.98 * camera.distance + 0.02 * required_distance)
            for side, label in enumerate(("PPO", "SAC")):
                data.qpos[:] = tracks[side]["qpos"][i]
                data.qvel[:] = tracks[side]["qvel"][i]
                mujoco.mj_forward(model, data)
                if args.camera_mode == "robot":
                    camera.lookat[:] = tracks[side]["qpos"][i, free_adr[:3]]
                    camera.lookat[2] = 0.85
                    camera.distance = args.camera_distance
                renderer.update_scene(data, camera=camera)
                visualizer = MujocoNativeDebugVisualizer(renderer.scene, model, env_idx=0)
                ghost_qpos = np.zeros(model.nq)
                ghost_qpos[free_adr[:3]] = ref_pos[i] + origin
                ghost_qpos[free_adr[3:7]] = ref_quat[i]
                ghost_qpos[joint_adr] = ref_joint[i]
                if args.camera_mode == "robot":
                    # Match MotionCommand's alignment used by the body-error plots.
                    inverse_ref = np.empty(4)
                    delta_quat = np.empty(4)
                    mujoco.mju_negQuat(inverse_ref, ref_anchor_quat[i])
                    mujoco.mju_mulQuat(delta_quat, data.xquat[anchor_body_id], inverse_ref)
                    w, x, y, z = delta_quat
                    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    yaw_quat = np.array((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, yaw_quat)
                    anchor = data.xpos[anchor_body_id].copy()
                    anchor[2] = ref_anchor_pos[i, 2]
                    ghost_qpos[free_adr[:3]] = anchor + rotation.reshape(3, 3) @ (ref_pos[i] - ref_anchor_pos[i])
                    aligned_quat = np.empty(4)
                    mujoco.mju_mulQuat(aligned_quat, yaw_quat, ref_quat[i])
                    ghost_qpos[free_adr[3:7]] = aligned_quat
                visualizer.add_ghost_mesh(ghost_qpos, ghost_model)
                if args.camera_mode == "robot" and i in (0, min(count - 1, round(5 / dt)), count // 2, count - 1):
                    # Independently verify rendered ghost alignment against GPU metrics.
                    rendered_error = np.linalg.norm(
                        visualizer._viz_data.xpos[tracked_body_ids] - data.xpos[tracked_body_ids], axis=-1).mean()
                    np.testing.assert_allclose(rendered_error, tracks[side]["metrics"][i, 1], rtol=1e-4, atol=2e-5)
                mujoco.mjr_render(mujoco.MjrRect(side * panel + 4, scene_bottom, panel - 8, scene_height), renderer.scene, context)
                title_rect = mujoco.MjrRect(side * panel + 12, height - header, panel - 24, header)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, title_rect,
                    label, "", context)
                for j, (_, title, _) in enumerate(METRICS):
                    fig = figures[side][j]
                    fig.title = f"{title}: {tracks[side]['metrics'][i, j]:.3f}"
                    fig.range[0] = (max(0.0, i * dt - args.history_seconds), max(args.history_seconds, i * dt))
                    fig.range[1] = (0.0, limits[j])
                    fig.linepnt[0] = len(indices)
                    fig.linedata[0, :2 * len(indices):2] = indices * dt
                    fig.linedata[0, 1:2 * len(indices):2] = tracks[side]["metrics"][indices, j]
                    row, column = divmod(j, 2)
                    rect = mujoco.MjrRect(side * panel + column * (panel // 2) + 4,
                        chart_bottom + (1 - row) * (chart_height // 2), panel // 2 - 8, chart_height // 2 - 4)
                    mujoco.mjr_figure(rect, fig, context)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    mujoco.MjrRect(side * panel + 12, 0, panel - 24, footer),
                    ("Green = aligned reference | World drift in torso plot | No resets"
                        if args.camera_mode == "robot" else "Green = world reference | Shared axes | No resets"), "", context)
            mujoco.mjr_readPixels(frame_buffer, None, viewport, context)
            frame = frame_buffer[::-1].copy()
            writer.append_data(frame)
            if i in (0, min(count - 1, round(5 / dt)), count // 2, count - 1):
                imageio.imwrite(output / f"preview_{i:05d}.png", frame)
            if i % 250 == 0 or i == count - 1:
                print(f"[RENDER] {i + 1}/{count}; elapsed {time.monotonic() - start:.1f}s", flush=True)
    return video_path


def save_static_plot(output: Path, tracks, dt: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for j, (_, title, _) in enumerate(METRICS):
        for side, label in enumerate(("PPO", "SAC")):
            axes[j].plot(np.arange(len(tracks[side]["metrics"])) * dt, tracks[side]["metrics"][:, j],
                label=label, color=COLORS[side], linewidth=0.9)
        axes[j].set_ylabel(title)
        axes[j].set_ylim(bottom=0)
        axes[j].grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[0].set_title("G1 motion tracking: PPO vs SAC")
    axes[-1].set_xlabel("Reference time (s)")
    fig.savefig(output / "tracking_errors.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--sac-checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracking_comparison"))
    parser.add_argument("--duration", type=float, help="Seconds to record; default is the complete motion.")
    parser.add_argument("--reuse-rollouts", action="store_true", help="Render the saved rollouts again without rerunning the simulation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--history-seconds", type=float, default=10.0)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--camera-azimuth", type=float, default=120.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--camera-mode", choices=("robot", "shared"), default="robot",
        help="Follow each robot with an aligned ghost, or fit all world positions into a common view.")
    args = parser.parse_args()
    for path in (args.ppo_checkpoint, args.sac_checkpoint, args.motion_file):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if args.width % 4 or args.height % 2 or args.history_seconds <= 0 or (args.duration is not None and args.duration <= 0):
        parser.error("Width must be divisible by 4, height must be even, durations must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with GPU device access enabled")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    configure_torch_backends()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    playbacks = []
    try:
        for task_id, checkpoint in ((TASK, args.ppo_checkpoint), (TASK + "-SAC", args.sac_checkpoint)):
            playbacks.append(make_playback(task_id, checkpoint, args.motion_file, args.device, args.seed))
        base = playbacks[0][0].unwrapped
        dt = base.step_dt
        command = base.command_manager.get_term("motion")
        with np.load(args.motion_file) as motion_data:
            reference_fps = float(motion_data["fps"].item())
        if not math.isclose(dt, playbacks[1][0].unwrapped.step_dt) or not math.isclose(1.0 / dt, reference_fps):
            raise ValueError("Both policies and the reference must share a timestep")
        frame_count = command.motion.time_step_total
        if args.duration is not None:
            frame_count = min(frame_count, math.ceil(args.duration / dt))
        if args.reuse_rollouts:
            previous = json.loads((args.output_dir / "metadata.json").read_text())
            expected_checkpoints = {"PPO": str(args.ppo_checkpoint.resolve()), "SAC": str(args.sac_checkpoint.resolve())}
            if (previous["checkpoints"] != expected_checkpoints or previous["motion"] != str(args.motion_file.resolve())
                or previous["seed"] != args.seed or previous["frames"] != frame_count or previous["fps"] != 1.0 / dt):
                raise ValueError("Saved rollout provenance does not match the requested recording")
            with np.load(args.output_dir / "rollouts.npz") as saved:
                tracks = [{key: saved[f"{label}_{key}"].copy() for key in ("qpos", "qvel", "metrics")}
                    for label in ("PPO", "SAC")]
            print(f"[REUSE] Loaded {frame_count} synchronized frames", flush=True)
        else:
            tracks = collect_rollouts(playbacks, frame_count)
        summary = save_measurements(args.output_dir, tracks, dt, args)
        save_static_plot(args.output_dir, tracks, dt)
        video = render_video(args.output_dir, tracks, base, dt, args)
        print(f"[DONE] {video.resolve()}\n{json.dumps(summary['mean_errors'], indent=2)}", flush=True)
    finally:
        for env, _, _ in playbacks:
            env.close()


if __name__ == "__main__":
    main()

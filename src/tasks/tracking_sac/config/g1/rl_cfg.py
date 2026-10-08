"""RL configuration for the Unitree G1 tracking task trained with FlashSAC (off-policy).

Hyperparameters follow the official FlashSAC IsaacLab benchmark defaults
(``rsl_rl_flashsac/isaaclab_flashsac/rl_cfg.py``), with the tracking-tuned
``mini_batch_size=8192`` and a smaller, bf16 replay buffer as the only deviations
(the reference 10M fp32 buffer does not fit on an 8 GB GPU).

Class names use qualified ``module:Class`` paths so that no ``__init__.py`` exports are
needed inside the vendored ``rsl_rl`` package (deleting the ported FlashSAC files then
leaves the rest of ``rsl_rl`` — and PPO — untouched).
"""

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class FlashSACActorCfg:
    """Configuration for the FlashSAC actor model."""

    class_name: str = "rsl_rl.models.flash_sac_model:FlashSACActor"
    num_blocks: int = 2
    """The number of residual FlashSAC blocks in the trunk."""
    hidden_dim: int = 128
    """The hidden dimension of the trunk."""
    log_std_min: float = -10.0
    """Lower bound of the Tanh-normalized log standard deviation."""
    log_std_max: float = 2.0
    """Upper bound of the Tanh-normalized log standard deviation."""


@dataclass
class FlashSACCriticCfg:
    """Configuration for the FlashSAC critic model."""

    class_name: str = "rsl_rl.models.flash_sac_model:FlashSACCritic"
    num_blocks: int = 2
    """The number of residual FlashSAC blocks in the trunk."""
    hidden_dim: int = 256
    """The hidden dimension of the trunk."""
    num_bins: int = 101
    """The number of bins of the categorical value distribution."""
    min_v: float = -5.0
    """Minimum value of the categorical support. Keep at -normalized_G_max."""
    max_v: float = 5.0
    """Maximum value of the categorical support. Keep at +normalized_G_max."""


@dataclass
class FlashSACAlgorithmCfg:
    """Configuration for the FlashSAC algorithm (official benchmark defaults)."""

    class_name: str = "rsl_rl.algorithms.flash_sac:FlashSAC"

    replay_buffer_size: int = 1_000_000
    """Max replay buffer size in total transitions across all environments. The reference
    uses 10M; reduced here to fit the replay buffer in 8 GB of VRAM."""
    buffer_min_length: int = 10_000
    """Minimum number of transitions before updates start."""
    buffer_optimize_memory_usage: bool = True
    """Store observations once and reconstruct next observations by index."""
    buffer_device: str | None = None
    """Device for the replay buffer storage. None uses the training device."""
    buffer_obs_dtype: str | None = "bfloat16"
    """Optional torch dtype for observation storage (halves replay-buffer memory)."""

    num_learning_epochs: int = 1
    """Number of gradient epochs per update step."""
    num_mini_batches: int = 2
    """Gradient updates per iteration (with num_steps_per_env=1: updates per env step)."""
    mini_batch_size: int = 8192
    """Mini-batch size drawn from the replay buffer (tracking-tuned value)."""

    learning_rate_init: float = 3.0e-4
    """Learning rate at the start of the warmup."""
    learning_rate_peak: float = 3.0e-4
    """Learning rate after warmup (optimizer base learning rate)."""
    learning_rate_end: float = 1.5e-4
    """Learning rate at the end of the cosine decay."""
    learning_rate_warmup_steps: int = 0
    """Number of update steps for the linear warmup."""
    learning_rate_decay_steps: int | None = None
    """Total schedule length in update steps. None resolves it from max_iterations."""

    actor_bc_alpha: float = 0.0
    """BC regularization coefficient. 0 disables BC regularization."""
    actor_noise_zeta_mu: float = 2.0
    """Zeta distribution exponent for exploration noise repeat lengths."""
    actor_noise_zeta_max: int = 16
    """Maximum exploration noise repeat length."""
    actor_update_period: int = 2
    """Actor/temperature update period relative to critic updates."""
    critic_target_update_tau: float = 0.01
    """EMA coefficient for the target critic."""

    temp_initial_value: float = 0.01
    """Initial temperature value."""
    temp_target_sigma: float = 0.15
    """Target per-dimension action std used to derive the target entropy."""
    temp_target_entropy: float | None = None
    """Explicit target entropy. None derives it from temp_target_sigma."""

    gamma: float = 0.99
    """The discount factor."""
    n_steps: int = 3
    """Number of steps for n-step returns."""
    normalize_reward: bool = True
    """Whether to normalize rewards with the running return scale."""
    normalized_G_max: float = 5.0
    """Maximum magnitude of the normalized return (match the critic min_v/max_v)."""

    use_compile: bool = True
    """Whether to torch.compile the network forward passes and update helpers."""
    compile_mode: str = "auto"
    """torch.compile mode. 'auto' picks autotuned kernels without CUDA graphs."""
    use_amp: bool = True
    """Whether to use fp16 automatic mixed precision for actor/critic updates."""

    rnd_cfg: dict | None = None
    """Not supported by FlashSAC; must stay None."""
    symmetry_cfg: dict | None = None
    """Symmetry data augmentation config, or None to disable (default)."""


@dataclass
class RslRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Runner configuration for off-policy (FlashSAC) training on mjlab tasks."""

    class_name: str = "rsl_rl.runners.off_policy_runner:OffPolicyRunner"
    num_steps_per_env: int = 1
    """The number of environment steps per iteration (1 for off-policy)."""
    start_training: int = 0
    """Extra update-free iterations; updates are already gated by buffer_min_length."""
    log_interval: int = 20
    """The number of iterations between logging the training statistics."""
    check_for_nan: bool = True
    """Whether to check environment outputs for NaN values."""
    action_bound: str = "scalar"
    """How the affine action scaling is computed: 'scalar' (+-action_bound_scale) or
    'joint_limit' (per-joint range from the soft joint position limits)."""
    action_bound_scale: float = 3.0
    """Half-width of the scalar action bounds (matches the official FlashSAC tracking
    task). Also the fallback when joint-limit scaling is unavailable."""

    actor: FlashSACActorCfg = field(default_factory=FlashSACActorCfg)
    """The actor model configuration."""
    critic: FlashSACCriticCfg = field(default_factory=FlashSACCriticCfg)
    """The critic model configuration."""
    algorithm: FlashSACAlgorithmCfg = field(default_factory=FlashSACAlgorithmCfg)
    """The algorithm configuration."""


def unitree_g1_tracking_sac_runner_cfg() -> RslRlOffPolicyRunnerCfg:
    """Create the FlashSAC runner configuration for the Unitree G1 tracking task."""
    return RslRlOffPolicyRunnerCfg(
        algorithm=FlashSACAlgorithmCfg(
            # Slightly higher exploration target than the reference (0.15): with the
            # relaxed anchor termination, keep the temperature from collapsing before
            # the policy escapes the early "stand still and die" local optimum.
            temp_target_sigma=0.2,
        ),
        experiment_name="g1_tracking_sac",
        logger="tensorboard",
        save_interval=5000,
        max_iterations=50_000,
    )

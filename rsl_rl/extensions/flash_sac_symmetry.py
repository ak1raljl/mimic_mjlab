# Copyright (c) 2021-2026, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from RSL-RL Project (BSD-3-Clause license),
# with modifications by Holiday Robotics (BSD-3-Clause license).

"""Minimal symmetry data-augmentation surface for FlashSAC.

The vendored rsl_rl does not ship the upstream ``rsl_rl.extensions.Symmetry`` class, so this
module provides the subset of its interface that FlashSAC consumes: data augmentation only.
The mirror loss is not supported by FlashSAC.
"""

from __future__ import annotations

from collections.abc import Callable

from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_callable


class FlashSacSymmetryAugmentation:
    """Symmetry data augmentation (drop-in for the upstream ``rsl_rl.extensions.Symmetry`` kwargs).

    Only data augmentation is supported: ``use_mirror_loss=True`` raises ``NotImplementedError``
    (the mirror loss needs the actor's mean-only forward, which FlashSAC's stochastic actor does
    not expose).
    """

    def __init__(
        self,
        env: VecEnv,
        data_augmentation_func: str | Callable,
        use_data_augmentation: bool = False,
        use_mirror_loss: bool = False,
        mirror_loss_coeff: float = 0.0,
    ) -> None:
        """Initialize the symmetry data augmentation.

        Args:
            env: Environment object. Passed to the data augmentation function for handling
                different observation terms.
            data_augmentation_func: Callable that generates mirrored observations / actions,
                called as ``func(env=env, obs=obs, actions=actions) -> (obs, actions)``. Resolved
                using :func:`~rsl_rl.utils.utils.resolve_callable`.
            use_data_augmentation: Whether to append mirrored samples to every mini-batch.
            use_mirror_loss: Not supported; must be False.
            mirror_loss_coeff: Unused (mirror loss is not supported); kept for config parity.
        """
        if use_mirror_loss:
            raise NotImplementedError(
                "FlashSAC does not support the symmetry mirror loss; set use_mirror_loss=False "
                "(use_data_augmentation is supported)."
            )

        self.env = env
        self.use_data_augmentation = use_data_augmentation
        self.use_mirror_loss = use_mirror_loss
        self.mirror_loss_coeff = mirror_loss_coeff

        # Resolve the augmentation function
        self.data_augmentation_func = resolve_callable(data_augmentation_func)

        # Inform the user if symmetry is configured only for logging
        if not use_data_augmentation:
            print("Symmetry not used for learning. We will use it for logging instead.")

"""Motion-tracking task family trained with FlashSAC (off-policy SAC).

Fully self-contained: the env factory, MDP terms, runner, config and registration all
live in this package (the env/MDP code is a copy of the PPO-based ``tracking`` family,
which remains untouched). Either family can be deleted without affecting the other;
shared dependencies are only mjlab and the vendored ``rsl_rl`` package.
"""

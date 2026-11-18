"""Hand-written RL agents. One readable file per algorithm."""

from kelvin_rl.agents.common import PPOConfig, SyncEnvs, compute_gae
from kelvin_rl.agents.ppo import PPO, ActorCritic
from kelvin_rl.agents.ppo_lagrangian import PPOLagrangian

AGENTS = {"ppo": PPO, "ppo_lagrangian": PPOLagrangian}

__all__ = ["AGENTS", "PPO", "PPOConfig", "PPOLagrangian", "ActorCritic", "SyncEnvs", "compute_gae"]

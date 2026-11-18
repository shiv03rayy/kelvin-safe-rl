"""Plumbing shared by both PPO agents: env stacking, networks, GAE.

Nothing algorithmic that is worth arguing about lives here -- the clipped
objective and the Lagrangian update stay in the agent files, one readable file
each. What is here is the machinery those two need and should not each
reinvent, plus the two pieces of bookkeeping that are easy to get quietly
wrong:

* **Truncation is not termination.** ``BatteryPackCoolingEnv`` never
  terminates; it truncates when the load profile runs out. A profile ending is
  not the pack ceasing to exist, so the value of that final state is *not*
  zero and the advantage at that step must bootstrap from ``V(s_final)``.
  CleanRL's reference PPO treats the two identically, which biases the value
  target at every episode boundary. :func:`compute_gae` keeps them apart.
* **Observation normalisation is part of the policy.** Its running mean and
  variance are fitted on training data only and frozen for evaluation, so they
  travel in the checkpoint beside the weights. An agent evaluated under a
  different normaliser is a different agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- envs
class SyncEnvs:
    """A handful of independent envs stepped in lockstep, with autoreset.

    Deliberately not ``gymnasium.vector``: the env's ``info`` dict carries
    arrays and a possibly-``None`` profile index, which the vector wrappers
    either drop or coerce. Both PPO variants need ``info["cost"]`` intact on
    every step, so the loop that collects it is written out here where it can
    be read.

    On the step that ends an episode, ``obs`` is already the *next* episode's
    first observation, and the ended episode's true final observation comes
    back separately as ``final_obs`` -- which is what the value function has to
    bootstrap from.
    """

    def __init__(self, make_env, n_envs: int, seed: int):
        self.envs = [make_env() for _ in range(n_envs)]
        self.n_envs = n_envs
        # One reproducible stream per env, derived from the single seed the
        # experiment records, so a run is replayable from that one number.
        self.seeds = [seed * 10_000 + i for i in range(n_envs)]
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.obs_dim = int(np.prod(self.single_observation_space.shape))
        self.act_dim = int(np.prod(self.single_action_space.shape))
        self._ep_return = np.zeros(n_envs)
        self._ep_cost = np.zeros(n_envs)
        self._ep_violations = np.zeros(n_envs)
        self._ep_energy = np.zeros(n_envs)
        self._ep_peak = np.full(n_envs, -np.inf)
        self._ep_len = np.zeros(n_envs, dtype=int)

    def reset(self) -> np.ndarray:
        obs = [env.reset(seed=s)[0] for env, s in zip(self.envs, self.seeds)]
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        """Returns ``(obs, reward, cost, done, final_obs, completed)``."""
        n = self.n_envs
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        final_obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        reward = np.zeros(n, dtype=np.float32)
        cost = np.zeros(n, dtype=np.float32)
        done = np.zeros(n, dtype=bool)
        completed: list[dict] = []

        for i, env in enumerate(self.envs):
            o, r, terminated, truncated, info = env.step(actions[i])
            reward[i] = r
            cost[i] = info["cost"]
            self._ep_return[i] += r
            self._ep_cost[i] += info["cost"]
            self._ep_violations[i] += float(info["violation"])
            self._ep_energy[i] += info["pump_energy_J"]
            self._ep_peak[i] = max(self._ep_peak[i], info["T_peak_substep"])
            self._ep_len[i] += 1
            if terminated or truncated:
                done[i] = True
                final_obs[i] = o
                completed.append(
                    {
                        "env": i,
                        "return": float(self._ep_return[i]),
                        "cost": float(self._ep_cost[i]),
                        "pct_steps_violating": 100.0 * self._ep_violations[i] / self._ep_len[i],
                        "pump_energy_kJ": float(self._ep_energy[i]) / 1e3,
                        "peak_T_C": float(self._ep_peak[i]),
                        "length": int(self._ep_len[i]),
                        # Termination would mean the pack ceased to exist. The
                        # env only truncates, so a True here is a bug upstream
                        # and the GAE bootstrap below would be wrong.
                        "terminated": bool(terminated),
                    }
                )
                self._ep_return[i] = self._ep_cost[i] = self._ep_violations[i] = 0.0
                self._ep_energy[i] = 0.0
                self._ep_peak[i] = -np.inf
                self._ep_len[i] = 0
                o, _ = env.reset()
            obs[i] = o

        return obs, reward, cost, done, final_obs, completed


# ------------------------------------------------------------------ networks
def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    """Orthogonal initialisation, the PPO-paper default.

    The small ``std`` on a policy's last layer matters more than it looks here:
    it starts the policy near the centre of the flow range, so the first
    rollouts are mid-flow rather than a random walk between no cooling and full
    cooling -- the latter wastes most of a short training budget discovering
    that both extremes are bad.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
        d = h
    layers += [layer_init(nn.Linear(d, out_dim), std=out_std)]
    return nn.Sequential(*layers)


class RunningNorm:
    """Chan-style running mean/variance for observations, frozen at evaluation."""

    def __init__(self, dim: int, clip: float = 10.0):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip
        self.frozen = False

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.mean.size)
        n = x.shape[0]
        mean, var = x.mean(axis=0), x.var(axis=0)
        delta = mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m_a = self.var * self.count
        m_b = var * n
        self.var = (m_a + m_b + delta**2 * self.count * n / total) / total
        self.count = total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "clip": self.clip,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])
        self.clip = float(state["clip"])


class IdentityNorm:
    """The no-op normaliser, so the training loop needs no branch."""

    frozen = True

    def update(self, x: np.ndarray) -> None:  # noqa: D102
        return None

    def __call__(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.asarray(x, dtype=np.float32)

    def state_dict(self) -> dict:  # noqa: D102
        return {}

    def load_state_dict(self, state: dict) -> None:  # noqa: D102
        return None


# ----------------------------------------------------------------------- GAE
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generalised advantage estimation, with truncation handled honestly.

    Shapes are ``(T, n_envs)``, except ``last_values`` which is ``(n_envs,)``.

    .. math::
        \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
        A_t = \delta_t + \gamma\lambda\,(1 - d_t)\,A_{t+1}

    ``bootstrap_values[t]`` supplies :math:`V(s_{t+1})` for a step that *ended*
    an episode: the value of that episode's real final observation, not of the
    next episode's first one, and not zero. For a genuinely terminated episode
    the caller passes zero there instead. The ``(1 - d_t)`` factor still cuts
    the recursion at an episode boundary, because advantages must not flow
    across one; what must not be cut is the one-step value target itself.

    ``last_values`` covers the final row of the rollout, where the next
    observation exists but was not collected -- the usual PPO bootstrap off the
    end of the buffer.

    Returns ``(advantages, returns)`` with ``returns = advantages + values``,
    the value function's regression target.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)

    T, n = rewards.shape
    advantages = np.zeros((T, n), dtype=np.float64)
    last_gae = np.zeros(n, dtype=np.float64)
    for t in reversed(range(T)):
        next_in_buffer = last_values if t == T - 1 else values[t + 1]
        next_values = np.where(dones[t] > 0, bootstrap_values[t], next_in_buffer)
        delta = rewards[t] + gamma * next_values - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """1 - Var(residual)/Var(target). Below zero means worse than the mean."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    var = y_true.var()
    if var < 1e-12:
        return float("nan")
    return float(1.0 - (y_true - y_pred).var() / var)


def set_seed(seed: int) -> None:
    """Seed every generator the training loop touches, and say so in the log."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # no CUDA kernels here; CPU MLPs are already deterministic


@dataclass
class PPOConfig:
    """Hyperparameters. Every field is written into the run's metadata."""

    total_steps: int = 400_000
    n_envs: int = 8
    n_steps: int = 360  # one full episode per env per update
    n_epochs: int = 10
    n_minibatches: int = 32
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    hidden: tuple[int, ...] = (64, 64)
    log_std_init: float = -1.0
    norm_obs: bool = True
    norm_adv: bool = True
    # Fixed-penalty PPO only: the coefficient swept in E2.
    penalty: float = 1.0
    # PPO-Lagrangian only.
    cost_limit: float = 1.0
    lambda_lr: float = 0.05
    lambda_init: float = 0.0
    lambda_max: float = 100.0
    extra: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.n_minibatches

    @property
    def n_updates(self) -> int:
        return max(1, self.total_steps // self.batch_size)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["hidden"] = list(self.hidden)
        d.update(self.extra)
        return d

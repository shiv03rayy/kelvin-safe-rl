"""PPO with a fixed cost penalty, written out in one file.

This is the unconstrained baseline for E1 and the left-hand side of E2: it
folds the safety cost straight into the reward,

.. math:: \\tilde r_t = r_t - \\lambda_\\text{fixed}\\, c_t

and then runs ordinary PPO on :math:`\\tilde r`. Sweeping
:math:`\\lambda_\\text{fixed}` is the point: too small and the agent buys cheap
pumping at the price of breaking the temperature limit, too large and it
hides at full flow. Nothing in the algorithm knows that :math:`c_t` means
"degrees above the limit"; it is just a term in the scalar it maximises. That
is exactly the weakness :mod:`kelvin_rl.agents.ppo_lagrangian` addresses by
keeping the two signals apart.

The implementation follows the PPO paper and CleanRL's single-file style, with
three deliberate departures, each flagged where it happens:

1. Truncation bootstraps from the real final observation
   (:func:`~kelvin_rl.agents.common.compute_gae`), instead of being treated as
   termination.
2. The Gaussian policy is unbounded and the *environment* clips the action into
   ``[0, 1]``; the log-probability is computed for the unclipped sample it
   actually drew. See :meth:`ActorCritic.act`.
3. The value loss is a plain mean-squared error, not the clipped variant.
   Clipping a value loss is an unargued habit with no result behind it, and one
   fewer hyperparameter is one fewer thing to defend.

Equation-to-code map, for the reader who wants to check the maths:

===================================  ==========================================
Clipped surrogate, PPO eq. (7)       :meth:`PPO._policy_loss`
GAE, Schulman et al. 2016 eq. (16)   :func:`~kelvin_rl.agents.common.compute_gae`
Value regression                     :meth:`PPO._value_loss`
===================================  ==========================================
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from kelvin_rl.agents.common import (
    IdentityNorm,
    PPOConfig,
    RunningNorm,
    SyncEnvs,
    compute_gae,
    explained_variance,
    layer_init,
    mlp,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class ActorCritic(nn.Module):
    """A Gaussian policy and one or two value heads, no shared trunk.

    Separate trunks because the reward here (pumping energy, smooth and almost
    linear in the action) and the cost (degrees above a threshold, zero almost
    everywhere and then sharply not) have very little structure in common. A
    shared trunk would make the two value functions fight over it.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), log_std_init=-1.0, cost_critic=False):
        super().__init__()
        # Final-layer std 0.01 keeps the initial policy close to its bias, and
        # bias 0.5 puts that at mid-flow: a sane starting controller rather
        # than a coin flip between no cooling and full cooling.
        self.actor_mean = mlp(obs_dim, act_dim, hidden, out_std=0.01)
        nn.init.constant_(self.actor_mean[-1].bias, 0.5)
        # State-independent log-std, as in the PPO paper. The pack gives no
        # reason for the policy's confidence to depend on temperature, and a
        # state-dependent std is a common source of premature collapse.
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(log_std_init)))
        self.critic = mlp(obs_dim, 1, hidden, out_std=1.0)
        self.cost_critic = mlp(obs_dim, 1, hidden, out_std=1.0) if cost_critic else None

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def cost_value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.cost_critic is None:
            raise RuntimeError("this agent was built without a cost critic")
        return self.cost_critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action, and return the log-prob of what was sampled.

        The sample is unbounded. The env clips it into ``[0, 1]`` when it maps
        the action to a flow rate, and the *unclipped* sample is what gets
        stored and scored -- clipping first and then scoring the clipped value
        under a continuous density would assign finite density to an event of
        probability mass, which is simply the wrong likelihood. The cost of
        this choice is that probability mass piling up outside ``[0, 1]`` is
        invisible to the policy; ``mean_action_raw`` is logged so that it is at
        least visible to the reader.
        """
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.distribution(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)


class PPO:
    """Fixed-penalty PPO. Subclassed by PPO-Lagrangian, which reuses this loop."""

    algo_name = "ppo"
    uses_cost_critic = False

    def __init__(self, make_env, cfg: PPOConfig, seed: int, run_dir: Path | str | None = None):
        set_seed(seed)
        self.cfg = cfg
        self.seed = seed
        self.envs = SyncEnvs(make_env, cfg.n_envs, seed)
        self.obs_dim, self.act_dim = self.envs.obs_dim, self.envs.act_dim
        self.obs_norm = RunningNorm(self.obs_dim) if cfg.norm_obs else IdentityNorm()
        self.net = ActorCritic(
            self.obs_dim,
            self.act_dim,
            hidden=tuple(cfg.hidden),
            log_std_init=cfg.log_std_init,
            cost_critic=self.uses_cost_critic,
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.global_step = 0
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._rows: list[dict] = []
        self._ep_window: list[dict] = []

    # ------------------------------------------------------------- rollout
    def collect(self, obs: np.ndarray):
        """One rollout of ``n_steps`` per env. Returns the buffer and next obs."""
        cfg = self.cfg
        T, n = cfg.n_steps, cfg.n_envs
        buf = {
            "obs": np.zeros((T, n, self.obs_dim), dtype=np.float32),
            "actions": np.zeros((T, n, self.act_dim), dtype=np.float32),
            "logprobs": np.zeros((T, n), dtype=np.float32),
            "rewards": np.zeros((T, n), dtype=np.float32),
            "costs": np.zeros((T, n), dtype=np.float32),
            "dones": np.zeros((T, n), dtype=np.float32),
            "values": np.zeros((T, n), dtype=np.float32),
            "boot": np.zeros((T, n), dtype=np.float32),
            "cost_values": np.zeros((T, n), dtype=np.float32),
            "cost_boot": np.zeros((T, n), dtype=np.float32),
        }
        episodes: list[dict] = []

        for t in range(T):
            self.obs_norm.update(obs)
            norm_obs = self.obs_norm(obs)
            buf["obs"][t] = norm_obs
            with torch.no_grad():
                obs_t = torch.as_tensor(norm_obs)
                action, logprob, _ = self.net.act(obs_t)
                buf["values"][t] = self.net.value(obs_t).numpy()
                if self.uses_cost_critic:
                    buf["cost_values"][t] = self.net.cost_value(obs_t).numpy()
            buf["actions"][t] = action.numpy()
            buf["logprobs"][t] = logprob.numpy()

            obs, reward, cost, done, final_obs, completed = self.envs.step(action.numpy())
            self.global_step += n
            buf["rewards"][t] = reward
            buf["costs"][t] = cost
            buf["dones"][t] = done.astype(np.float32)

            if done.any():
                # Truncation, not termination: bootstrap from the value of the
                # episode's real final observation. The normaliser is *not*
                # updated on it -- it is scored, not trained on, and it has
                # already been counted once as the next episode's first obs.
                with torch.no_grad():
                    fin = torch.as_tensor(self.obs_norm(final_obs))
                    v_fin = self.net.value(fin).numpy()
                    buf["boot"][t] = np.where(done, v_fin, 0.0)
                    if self.uses_cost_critic:
                        vc_fin = self.net.cost_value(fin).numpy()
                        buf["cost_boot"][t] = np.where(done, vc_fin, 0.0)
                for ep in completed:
                    if ep["terminated"]:
                        raise RuntimeError(
                            "the env reported terminated=True; the GAE bootstrap here "
                            "assumes truncation only (see common.compute_gae)"
                        )
                episodes.extend(completed)

        with torch.no_grad():
            last_obs = torch.as_tensor(self.obs_norm(obs))
            buf["last_values"] = self.net.value(last_obs).numpy()
            buf["cost_last_values"] = (
                self.net.cost_value(last_obs).numpy() if self.uses_cost_critic else np.zeros(n, dtype=np.float32)
            )
        self._ep_window = episodes
        return buf, obs, episodes

    # ------------------------------------------------------------ advantages
    def advantages(self, buf: dict) -> tuple[np.ndarray, dict]:
        """Fold cost into reward with the fixed penalty, then run GAE once.

        This is the whole of the fixed-penalty method: one scalar, one critic,
        and a coefficient somebody has to pick by hand.
        """
        cfg = self.cfg
        shaped = buf["rewards"] - cfg.penalty * buf["costs"]
        adv, ret = compute_gae(
            shaped, buf["values"], buf["dones"], buf["boot"], buf["last_values"], cfg.gamma, cfg.gae_lambda
        )
        return adv, {"returns": ret, "shaped_reward_mean": float(shaped.mean())}

    # ----------------------------------------------------------------- losses
    def _policy_loss(self, logratio: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        r"""PPO's clipped surrogate, eq. (7).

        .. math::
            L = -\mathbb{E}\big[\min(\rho_t A_t,\;
                \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)\big],
            \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}

        The minimum, not the clip alone: clipping on its own would still let an
        improvement larger than :math:`1+\epsilon` through when the advantage
        is negative. Taking the pessimistic branch is what makes the bound a
        bound.
        """
        ratio = logratio.exp()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * adv
        return -torch.min(unclipped, clipped).mean()

    def _value_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((pred - target) ** 2).mean()

    # ----------------------------------------------------------------- update
    def update(self, buf: dict) -> dict:
        cfg = self.cfg
        adv, extras = self.advantages(buf)

        flat = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(-1, *np.shape(x)[2:]))
        b_obs = flat(buf["obs"])
        b_actions = flat(buf["actions"])
        b_logprobs = flat(buf["logprobs"])
        b_adv = flat(adv)
        b_returns = flat(extras["returns"])
        b_values = flat(buf["values"])
        b_cost_returns = flat(extras["cost_returns"]) if "cost_returns" in extras else None

        idx = np.arange(cfg.batch_size)
        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clipfrac": [], "vc_loss": []}
        stopped_epoch = cfg.n_epochs

        for epoch in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                new_logprob, entropy = self.net.evaluate(b_obs[mb], b_actions[mb])
                logratio = new_logprob - b_logprobs[mb]

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = self._policy_loss(logratio, mb_adv)
                v_loss = self._value_loss(self.net.value(b_obs[mb]), b_returns[mb])
                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

                vc_loss = torch.zeros(())
                if self.uses_cost_critic and b_cost_returns is not None:
                    vc_loss = self._value_loss(self.net.cost_value(b_obs[mb]), b_cost_returns[mb])
                    loss = loss + cfg.vf_coef * vc_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Schulman's k3 estimator: non-negative and lower variance
                    # than -mean(logratio), which can go negative and makes the
                    # early-stop test fire on noise.
                    approx_kl = ((logratio.exp() - 1) - logratio).mean().item()
                stats["pg_loss"].append(pg_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["vc_loss"].append(float(vc_loss.item()))
                stats["entropy"].append(entropy_loss.item())
                stats["approx_kl"].append(approx_kl)
                stats["clipfrac"].append(
                    ((logratio.exp() - 1.0).abs() > cfg.clip_coef).float().mean().item()
                )

            if cfg.target_kl is not None and np.mean(stats["approx_kl"][-cfg.n_minibatches :]) > cfg.target_kl:
                # Stop this update early rather than walk away from the policy
                # the advantages were estimated under.
                stopped_epoch = epoch + 1
                break

        out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
        out["epochs_run"] = stopped_epoch
        out["explained_var"] = explained_variance(b_values.numpy(), b_returns.numpy())
        out["adv_mean"] = float(np.mean(adv))
        out["shaped_reward_mean"] = extras.get("shaped_reward_mean", float("nan"))
        out["mean_action_raw"] = float(np.mean(buf["actions"]))
        out["log_std"] = float(self.net.actor_logstd.detach().mean().item())
        return out

    # ------------------------------------------------------------------ train
    def learn(self, progress_every: int = 1) -> list[dict]:
        cfg = self.cfg
        obs = self.envs.reset()
        t0 = time.time()
        for update in range(1, cfg.n_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1) / cfg.n_updates
                for group in self.optimizer.param_groups:
                    group["lr"] = frac * cfg.lr
            buf, obs, episodes = self.collect(obs)
            stats = self.update(buf)
            row = {
                "update": update,
                "global_step": self.global_step,
                "lr": self.optimizer.param_groups[0]["lr"],
                "sps": int(self.global_step / max(1e-9, time.time() - t0)),
                "n_episodes": len(episodes),
                **self._episode_stats(episodes),
                **stats,
                **self.extra_log(),
            }
            self._rows.append(row)
            if self.run_dir is not None:
                self._write_progress()
            if progress_every and update % progress_every == 0:
                print(
                    f"[{self.algo_name} seed={self.seed}] update {update}/{cfg.n_updates} "
                    f"step={self.global_step} return={row['ep_return']:.3f} "
                    f"cost={row['ep_cost']:.3f} viol%={row['ep_pct_violating']:.2f} "
                    f"kJ={row['ep_pump_energy_kJ']:.2f} peak={row['ep_peak_T_C']:.2f} "
                    f"{self.extra_log_str()}",
                    flush=True,
                )
        return self._rows

    @staticmethod
    def _episode_stats(episodes: list[dict]) -> dict:
        if not episodes:
            return {k: float("nan") for k in
                    ("ep_return", "ep_cost", "ep_pct_violating", "ep_pump_energy_kJ", "ep_peak_T_C", "ep_cost_max")}
        f = lambda key: float(np.mean([e[key] for e in episodes]))
        return {
            "ep_return": f("return"),
            "ep_cost": f("cost"),
            "ep_pct_violating": f("pct_steps_violating"),
            "ep_pump_energy_kJ": f("pump_energy_kJ"),
            "ep_peak_T_C": f("peak_T_C"),
            # The worst episode in the batch, because for a safety method the
            # mean over episodes is the least interesting number.
            "ep_cost_max": float(np.max([e["cost"] for e in episodes])),
        }

    def extra_log(self) -> dict:
        return {}

    def extra_log_str(self) -> str:
        return ""

    # ------------------------------------------------------------------- i/o
    def _write_progress(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "progress.csv"
        fields = list(self._rows[-1].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algo": self.algo_name,
                "seed": self.seed,
                "config": self.cfg.as_dict(),
                "model": self.net.state_dict(),
                # The normaliser travels with the weights: an agent evaluated
                # under a different one is a different agent.
                "obs_norm": self.obs_norm.state_dict(),
                "global_step": self.global_step,
                "state": self.algo_state(),
            },
            path,
        )

    def algo_state(self) -> dict:
        return {}

    def write_meta(self, extra: dict | None = None) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "algo": self.algo_name,
            "seed": self.seed,
            "config": self.cfg.as_dict(),
            "obs_dim": self.obs_dim,
            "git_commit": _git_commit(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            **(extra or {}),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _git_commit() -> str | None:
    """The commit a run was produced at, or None outside a clean checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None

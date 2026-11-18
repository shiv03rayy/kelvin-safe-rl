r"""PPO-Lagrangian: the same PPO core, with the constraint learned instead of priced.

The constrained problem is

.. math::
    \max_\theta\; J_r(\theta) \quad\text{subject to}\quad J_c(\theta) \le d

whose Lagrangian is :math:`J_r - \lambda\,(J_c - d)` with :math:`\lambda \ge 0`.
Solving it as a saddle point gives two alternating updates:

* **Policy** (descent on :math:`\theta`): ascend the combined advantage

  .. math:: A_t = \frac{A^r_t - \lambda\,A^c_t}{1 + \lambda}

  The :math:`1/(1+\lambda)` is not in the maths; it is there so the gradient
  does not grow without bound as :math:`\lambda` does. Without it the policy's
  effective learning rate rises with :math:`\lambda`, and a run that violates
  early then takes enormous steps exactly when it is least stable.

* **Multiplier** (ascent on :math:`\lambda`):

  .. math:: \lambda \leftarrow \big[\lambda + K_i\,(J_c - d)\big]_+

  Cost above the limit pushes :math:`\lambda` up until the policy gives way;
  cost below it lets :math:`\lambda` decay back towards zero, which is the
  whole point -- the trade-off is found rather than chosen.

Why this should beat a fixed penalty: the fixed penalty fixes the *price* of a
violation and lets the resulting safety be whatever it turns out to be, so the
coefficient has to be retuned for every change to the pack, the profiles or
the limit. The multiplier fixes the *constraint* and lets the price move, so
the same hyperparameters target the same safety level.

**PID-Lagrangian.** Stooke et al. (2020) point out that the update above is a
pure integral controller on the constraint error, which is why plain
PPO-Lagrangian oscillates: the integrator overshoots and the cost rings around
the limit. Adding proportional and derivative terms damps it. Both are
available here (``lambda_kp``, ``lambda_kd``) and default to zero, so the
default agent *is* the plain Lagrangian method and the PID version is an
ablation rather than a different implementation.
"""

from __future__ import annotations

import numpy as np

from kelvin_rl.agents.common import compute_gae
from kelvin_rl.agents.ppo import PPO


class PPOLagrangian(PPO):
    """PPO with a cost critic and a learned Lagrange multiplier."""

    algo_name = "ppo_lagrangian"
    uses_cost_critic = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg
        self.lam = float(cfg.lambda_init)
        self.lambda_kp = float(cfg.extra.get("lambda_kp", 0.0))
        self.lambda_kd = float(cfg.extra.get("lambda_kd", 0.0))
        self._prev_error = 0.0
        self._integral = float(cfg.lambda_init)
        self._last_cost = float("nan")
        self._last_error = float("nan")

    # ------------------------------------------------------------ advantages
    def advantages(self, buf: dict) -> tuple[np.ndarray, dict]:
        """Two GAEs -- one per signal -- combined under the current multiplier.

        The cost advantage uses the same :math:`\\gamma` and :math:`\\lambda_{GAE}`
        as the reward advantage. That is a choice, not a law: a shorter horizon
        for cost would make the agent myopic about heat that is still building,
        and heat in this pack builds over tens of control steps.
        """
        cfg = self.cfg
        adv_r, ret_r = compute_gae(
            buf["rewards"], buf["values"], buf["dones"], buf["boot"], buf["last_values"], cfg.gamma, cfg.gae_lambda
        )
        adv_c, ret_c = compute_gae(
            buf["costs"],
            buf["cost_values"],
            buf["dones"],
            buf["cost_boot"],
            buf["cost_last_values"],
            cfg.gamma,
            cfg.gae_lambda,
        )
        combined = (adv_r - self.lam * adv_c) / (1.0 + self.lam)
        return combined, {
            "returns": ret_r,
            "cost_returns": ret_c,
            "adv_r_mean": float(adv_r.mean()),
            "adv_c_mean": float(adv_c.mean()),
        }

    # ------------------------------------------------------------- multiplier
    def update_lambda(self, episode_costs: list[float]) -> None:
        r"""One multiplier step from the episodes that finished in this rollout.

        The error is measured on *episode* cost against the episode cost limit,
        not on per-step cost, because the constraint being enforced is about a
        whole drive cycle: a controller that runs hot for two steps and cold
        for three hundred is not safe, and a per-step average would call it so.

        Measured on the rollout just collected, so the multiplier reacts to the
        policy that produced it rather than to a policy two updates old.
        """
        if not episode_costs:
            return
        cost = float(np.mean(episode_costs))
        error = cost - self.cfg.cost_limit

        self._integral = max(0.0, self._integral + self.cfg.lambda_lr * error)
        derivative = max(0.0, error - self._prev_error)  # one-sided: damp rises, do not fight falls
        lam = self._integral + self.lambda_kp * error + self.lambda_kd * derivative

        self.lam = float(np.clip(lam, 0.0, self.cfg.lambda_max))
        self._prev_error = error
        self._last_cost = cost
        self._last_error = error

    def update(self, buf: dict) -> dict:
        # Order matters: the multiplier is updated *before* the policy step, so
        # the combined advantage the policy follows reflects the cost this
        # rollout actually incurred. Updating it afterwards would apply the
        # correction one rollout late.
        self.update_lambda([e["cost"] for e in self._ep_window])
        return super().update(buf)

    # -------------------------------------------------------------- logging
    def extra_log(self) -> dict:
        return {
            "lambda": self.lam,
            "lambda_integral": self._integral,
            "cost_limit": self.cfg.cost_limit,
            "constraint_error": self._last_error,
            "cost_for_lambda": self._last_cost,
        }

    def extra_log_str(self) -> str:
        return f"lambda={self.lam:.3f}"

    def algo_state(self) -> dict:
        return {
            "lambda": self.lam,
            "integral": self._integral,
            "prev_error": self._prev_error,
        }

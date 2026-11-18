"""One episode in, one row of metrics out.

The five headline metrics are the ones the README lists: pump energy per
episode (kJ), peak cell temperature, percentage of steps above ``T_max``,
degree-seconds above ``T_max`` and maximum temperature spread. Their
definitions match ``scripts/plot_constant_flow.py`` so the M1 constant-flow
sweep and the M2 baseline evaluation can be read side by side.

Two conventions worth spelling out:

* ``peak_T_C`` uses ``info["T_peak_substep"]``, the hottest any cell got at any
  1 s physics sub-step, not only at the 10 s control instants. A controller
  cannot hide a violation between two control decisions.
* ``degree_s_above`` sums the per-step cost (degrees above ``T_max``, summed
  over cells) and multiplies by ``dt_ctrl``, so it is cell-kelvin-seconds. It
  is a severity measure, where ``pct_steps_violating`` is a frequency measure.

Everything here reads the environment's ``info`` dict, which is logged data,
not privileged controller input.
"""

from __future__ import annotations

import numpy as np


def run_episode(env, controller, *, seed: int, profile_index: int, options: dict | None = None):
    """Roll out one full episode under a controller. Returns (infos, actions).

    The controller is reset first, then sees nothing but the observation the
    env hands back.
    """
    controller.reset()
    reset_options = {"profile_index": profile_index, **(options or {})}
    obs, _ = env.reset(seed=seed, options=reset_options)
    infos: list[dict] = []
    actions: list[float] = []
    done = False
    while not done:
        action = controller(obs)
        actions.append(float(np.asarray(action, dtype=float).reshape(-1)[0]))
        obs, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        done = terminated or truncated
    return infos, actions


def episode_metrics(env, infos: list[dict], actions: list[float] | None = None) -> dict:
    """The README metrics for one episode, plus a little actuator diagnostics."""
    metrics = {
        "pump_energy_kJ": sum(i["pump_energy_J"] for i in infos) / 1e3,
        "peak_T_C": max(i["T_peak_substep"] for i in infos),
        "pct_steps_violating": 100 * float(np.mean([i["violation"] for i in infos])),
        "degree_s_above": sum(i["cost"] for i in infos) * env.dt_ctrl,
        "max_spread_K": max(i["spread"] for i in infos),
        "pct_steps_spread_violating": 100 * float(np.mean([i["spread_violation"] for i in infos])),
        "pct_steps_rate_limited": 100 * float(np.mean([i["rate_limited"] for i in infos])),
        "mean_mdot_g_s": 1e3 * float(np.mean([i["mdot"] for i in infos])),
        "n_steps": len(infos),
    }
    if actions is not None:
        metrics["mean_action"] = float(np.mean(actions))
        metrics["n_action_reversals"] = int(np.sum(np.abs(np.diff(np.sign(np.diff(actions)))) > 0))
    return metrics

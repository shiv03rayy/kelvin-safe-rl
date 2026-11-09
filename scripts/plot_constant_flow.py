"""M1 sanity figure: how the pack behaves under constant coolant flow.

Sweeps constant actions over every held-out profile, writes the raw numbers to
``results/m1_constant_flow_sweep.csv``, and plots ``figures/m1_constant_flow.png``
from that CSV. Run with: ``python scripts/plot_constant_flow.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kelvin_rl.config import REPO_ROOT
from kelvin_rl.envs import BatteryPackCoolingEnv

ACTIONS = np.round(np.linspace(0.0, 1.0, 11), 2)


def rollout(env, action, profile_index, seed=0):
    env.reset(seed=seed, options={"profile_index": profile_index, "action_init": action})
    T, done, infos = [env.T.copy()], False, []
    while not done:
        _, _, term, trunc, info = env.step(np.array([action], dtype=np.float32))
        T.append(info["T_cells"])
        infos.append(info)
        done = term or trunc
    return np.array(T), infos


def sweep(env) -> pd.DataFrame:
    rows = []
    for idx in range(env.profiles.count("test")):
        prof = env.profiles.get("test", idx)
        for a in ACTIONS:
            _, infos = rollout(env, float(a), idx)
            rows.append(
                {
                    "profile": idx,
                    "fast_charge": prof.has_fast_charge,
                    "action": a,
                    "mdot_kg_s": env.action_to_mdot(a),
                    "peak_T_C": max(i["T_peak_substep"] for i in infos),
                    "pct_steps_violating": 100 * np.mean([i["violation"] for i in infos]),
                    "degree_s_above": sum(i["cost"] for i in infos) * env.dt_ctrl,
                    "max_spread_K": max(i["spread"] for i in infos),
                    "pump_energy_kJ": sum(i["pump_energy_J"] for i in infos) / 1e3,
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example-action", type=float, default=0.3)
    args = ap.parse_args()

    env = BatteryPackCoolingEnv(split="test")
    results_dir, fig_dir = REPO_ROOT / "results", REPO_ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    df = sweep(env)
    csv_path = results_dir / "m1_constant_flow_sweep.csv"
    df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)  # the figure is drawn from the logged file, not memory

    example = next(i for i in range(env.profiles.count("test")) if env.profiles.get("test", i).has_fast_charge)
    current = env.profiles.get("test", example).current
    T, _ = rollout(env, args.example_action, example)
    t_ctrl = np.arange(len(T)) * env.dt_ctrl / 60

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    ax = axes[0]
    ax.plot(np.arange(len(current)) * env.dt_phys / 60, current, lw=0.6, color="0.3")
    ax.set(ylabel="Cell current (A)", xlabel="Time (min)",
           title=f"Held-out profile {example}: current (negative = fast charge)")

    ax = axes[1]
    cmap = plt.get_cmap("coolwarm")
    for i in range(env.n):
        ax.plot(t_ctrl, T[:, i], color=cmap(i / (env.n - 1)), lw=1.2,
                label=f"cell {i + 1}" if i in (0, env.n - 1) else None)
    ax.axhline(env.T_max, color="k", ls="--", lw=1, label=f"T_max = {env.T_max:.0f} °C")
    ax.set(ylabel="Cell temperature (°C)", xlabel="Time (min)",
           title=f"Constant flow, action {args.example_action} "
                 f"(ṁ = {env.action_to_mdot(args.example_action) * 1e3:.1f} g/s); coolant enters at cell 1")
    ax.legend(loc="upper left")

    ax = axes[2]
    g = df.groupby("action")
    ax.plot(g["peak_T_C"].max().index, g["peak_T_C"].max(), "o-", color="C3", label="worst profile")
    ax.plot(g["peak_T_C"].mean().index, g["peak_T_C"].mean(), "o--", color="C3", alpha=0.6, label="mean over profiles")
    ax.axhline(env.T_max, color="k", ls="--", lw=1)
    ax.set(xlabel="Constant action (0 = min flow, 1 = max flow)", ylabel="Peak cell temperature (°C)",
           title=f"Sweep over all {df.profile.nunique()} held-out profiles")
    ax2 = ax.twinx()
    ax2.plot(g["pump_energy_kJ"].mean().index, g["pump_energy_kJ"].mean(), "s-", color="C0", label="pump energy")
    ax2.set_ylabel("Pump energy per episode (kJ)", color="C0")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper center")

    out = fig_dir / "m1_constant_flow.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {csv_path.relative_to(REPO_ROOT)} and {out.relative_to(REPO_ROOT)}")

    summary = g.agg(
        mdot_g_s=("mdot_kg_s", lambda s: 1e3 * s.iloc[0]),
        peak_mean=("peak_T_C", "mean"),
        peak_worst=("peak_T_C", "max"),
        profiles_violating=("pct_steps_violating", lambda s: int((s > 0).sum())),
        spread_worst=("max_spread_K", "max"),
        pump_kJ=("pump_energy_kJ", "mean"),
    )
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

"""M2 evaluation: every classical baseline on the HELD-OUT test profiles.

Runs constant maximum flow, the thermostat and the PID over every test profile
for several seeds, in each observation mode, and writes one row per
(controller, obs_mode, profile, seed) to ``results/m2_baselines_test.csv``. The
summary printed at the end is computed from that file after reading it back, so
the table and the CSV cannot drift apart.

Controller parameters come from ``kelvin_rl.control.baselines.TUNED``, which was
selected on training profiles by ``scripts/tune_baselines.py``. Nothing is
re-tuned here: this script only measures.

Metrics are the ones the README lists - pump energy per episode (kJ), peak cell
temperature, percentage of steps above T_max, degree-seconds above T_max,
maximum temperature spread - defined in :mod:`kelvin_rl.eval.metrics`.

Run with::

    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --seeds 0 1 2 3 4 --obs-modes full partial
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from kelvin_rl.config import REPO_ROOT
from kelvin_rl.control import baselines
from kelvin_rl.envs import BatteryPackCoolingEnv
from kelvin_rl.eval.metrics import episode_metrics, run_episode

SPLIT = "test"
CSV_NAME = "m2_baselines_test.csv"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)  # "at least 5 seeds", per the README


def describe(name: str, env) -> str:
    """The parameters a controller is about to run with, for the log."""
    if name == "constant_max":
        return "action = 1.0"
    return repr(baselines.tuned_params(name, env.obs_mode))


def evaluate(obs_modes, seeds, controllers) -> pd.DataFrame:
    rows = []
    t0 = time.perf_counter()
    for obs_mode in obs_modes:
        env = BatteryPackCoolingEnv(obs_mode=obs_mode, split=SPLIT)
        for name in controllers:
            controller = baselines.make_controller(name, env)
            for profile in range(env.profiles.count(SPLIT)):
                for seed in seeds:
                    infos, actions = run_episode(
                        env, controller, seed=seed, profile_index=profile
                    )
                    rows.append(
                        {
                            "controller": name,
                            "obs_mode": obs_mode,
                            "profile": profile,
                            "seed": seed,
                            "fast_charge": env.profiles.get(SPLIT, profile).has_fast_charge,
                            **episode_metrics(env, infos, actions),
                        }
                    )
            print(
                f"  {obs_mode:>15s} / {name:<12s} {describe(name, env)}"
                f"  [{time.perf_counter() - t0:6.1f} s]",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, T_max: float) -> pd.DataFrame:
    """Aggregate the per-episode log over profiles and seeds."""
    return df.groupby(["controller", "obs_mode"]).agg(
        episodes=("profile", "count"),
        pump_kJ_mean=("pump_energy_kJ", "mean"),
        pump_kJ_worst=("pump_energy_kJ", "max"),
        peak_T_mean=("peak_T_C", "mean"),
        peak_T_worst=("peak_T_C", "max"),
        pct_steps_above_mean=("pct_steps_violating", "mean"),
        degree_s_above_mean=("degree_s_above", "mean"),
        degree_s_above_worst=("degree_s_above", "max"),
        spread_worst=("max_spread_K", "max"),
        episodes_violating=("peak_T_C", lambda s: int((s > T_max).sum())),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--obs-modes", nargs="+", default=["full", "partial"])
    ap.add_argument("--controllers", nargs="+", default=list(baselines.BASELINES))
    args = ap.parse_args()

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / CSV_NAME

    env = BatteryPackCoolingEnv(split=SPLIT)
    print(
        f"evaluating {args.controllers} on {env.profiles.count(SPLIT)} held-out "
        f"{SPLIT} profiles x {len(args.seeds)} seeds {args.seeds}",
        flush=True,
    )
    df = evaluate(args.obs_modes, args.seeds, args.controllers)
    df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)  # the summary is computed from the logged file
    print(f"\nwrote {csv_path.relative_to(REPO_ROOT)} ({len(df)} episodes)")

    summary = summarise(df, env.T_max)
    print(f"\nT_max = {env.T_max:.0f} C, dT_max = {env.dT_max:.0f} K, "
          f"pump energy ceiling = {env.nominal_params.P_pump_max * env.dt_ctrl * env.n_ctrl_steps / 1e3:.0f} kJ")
    print(summary.round(3).to_string())

    worst = df.loc[df.groupby(["controller", "obs_mode"])["peak_T_C"].idxmax()]
    print("\nworst episode per controller (highest peak cell temperature):")
    print(
        worst[
            ["controller", "obs_mode", "profile", "seed", "fast_charge", "peak_T_C",
             "degree_s_above", "pump_energy_kJ"]
        ]
        .round(3)
        .to_string(index=False)
    )

    unsafe = summary[summary.episodes_violating > 0]
    if len(unsafe):
        print(
            "\nNOT SAFE on the held-out split: "
            + ", ".join(
                f"{c}/{m} in {int(n)} of {int(e)} episodes"
                for (c, m), n, e in zip(unsafe.index, unsafe.episodes_violating, unsafe.episodes)
            )
        )
    else:
        print("\nevery controller stayed below T_max in every held-out episode")

    # Energy relative to the ceiling that constant maximum flow sets.
    constant = df[df.controller == "constant_max"]["pump_energy_kJ"]
    if len(constant):
        rel = 100 * df.groupby(["controller", "obs_mode"])["pump_energy_kJ"].mean() / constant.mean()
        print("\nmean pump energy as a percentage of the constant-max ceiling:")
        print(rel.round(2).to_string())
        print(f"(ceiling = {constant.mean():.3f} kJ, mean over constant_max episodes)")


if __name__ == "__main__":
    main()

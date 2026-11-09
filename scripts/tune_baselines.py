"""M2 tuning sweep: pick the thermostat and PID parameters on TRAINING profiles.

Rolls every grid point in :mod:`kelvin_rl.control.tuning` over the fixed set of
training profiles, writes one row per (controller, obs_mode, grid point,
profile) to ``results/m2_tuning_sweep.csv``, then reads that file back and
applies the selection rules to it. The printed block is the content of ``TUNED``
in ``kelvin_rl.control.baselines``; nothing is chosen by eye, and the held-out
test split is never touched here.

``thermostat`` and ``thermostat_margin`` are the same controller judged by two
rules, so they share one set of simulated episodes - only the controllers in
``SWEEP_CONTROLLERS`` are rolled out. The shortlists printed at the end show
how many candidates each rule had to choose from, including whether the
margin rule had any viable candidate at all.

Run with::

    python scripts/tune_baselines.py              # full sweep, rewrites the CSV
    python scripts/tune_baselines.py --check      # re-select from the logged CSV only

``--check`` compares the selection against the defaults currently in
``baselines.TUNED`` and exits non-zero if they disagree, which is the same
assertion ``tests/test_baselines.py`` makes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from kelvin_rl.config import REPO_ROOT
from kelvin_rl.control import baselines
from kelvin_rl.control.tuning import (
    GRIDS,
    SWEEP_CONTROLLERS,
    TUNED_OBS_MODES,
    TUNING_PROFILES,
    TUNING_SPLIT,
    aggregate,
    build_params,
    candidates,
    rank,
    select_best,
    selection_rules,
)
from kelvin_rl.envs import BatteryPackCoolingEnv
from kelvin_rl.eval.metrics import episode_metrics, run_episode

CSV_NAME = "m2_tuning_sweep.csv"
PARAM_COLUMNS = sorted({key for grid in GRIDS.values() for key in grid})


_ENVS: dict[str, BatteryPackCoolingEnv] = {}


def _env(obs_mode: str) -> BatteryPackCoolingEnv:
    """One env per observation mode, cached (also per worker process)."""
    if obs_mode not in _ENVS:
        _ENVS[obs_mode] = BatteryPackCoolingEnv(obs_mode=obs_mode, split=TUNING_SPLIT)
    return _ENVS[obs_mode]


def run_candidate(task) -> list[dict]:
    """All tuning episodes for one grid point. The unit of work, also in a worker."""
    obs_mode, name, values, profiles = task
    env = _env(obs_mode)
    controller = baselines.make_controller(name, env, build_params(name, values))
    rows = []
    for profile in profiles:
        # Episode seed = profile index: fixed initial temperatures and a fixed
        # sensor-noise stream, so every candidate meets exactly the same episodes.
        infos, actions = run_episode(env, controller, seed=profile, profile_index=profile)
        rows.append(
            {
                "controller": name,
                "obs_mode": obs_mode,
                **{c: values.get(c, float("nan")) for c in PARAM_COLUMNS},
                "split": TUNING_SPLIT,
                "profile": profile,
                "seed": profile,
                "fast_charge": env.profiles.get(TUNING_SPLIT, profile).has_fast_charge,
                **episode_metrics(env, infos, actions),
            }
        )
    return rows


def sweep(obs_modes, profiles, jobs: int = 1) -> pd.DataFrame:
    """Every grid point of every baseline, over every tuning profile.

    Episodes are independent and individually seeded, so running them in
    parallel changes nothing but the wall clock; the frame is sorted before it
    is written, which keeps the CSV byte-identical whatever ``jobs`` was.
    """
    tasks = [
        (obs_mode, name, values, tuple(profiles))
        for obs_mode in obs_modes
        for name in SWEEP_CONTROLLERS  # a controller that differs only in its
        for values in candidates(name)  # selection rule needs no extra episodes
    ]
    t0 = time.perf_counter()
    rows: list[dict] = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for done, batch in enumerate(pool.map(run_candidate, tasks, chunksize=4), start=1):
                rows.extend(batch)
                if done % 50 == 0 or done == len(tasks):
                    print(
                        f"  {done:4d}/{len(tasks)} candidates, {time.perf_counter() - t0:6.1f} s",
                        flush=True,
                    )
    else:
        for done, task in enumerate(tasks, start=1):
            rows.extend(run_candidate(task))
            if done % 50 == 0 or done == len(tasks):
                print(
                    f"  {done:4d}/{len(tasks)} candidates, {time.perf_counter() - t0:6.1f} s",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    return df.sort_values(["controller", "obs_mode", *PARAM_COLUMNS, "profile"]).reset_index(
        drop=True
    )


def format_selection(selected) -> str:
    lines = ["TUNED = {"]
    for (name, obs_mode), params in sorted(selected.items()):
        lines.append(f'    ("{name}", "{obs_mode}"): {params!r},')
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not sweep; re-select from the logged CSV and compare with baselines.TUNED",
    )
    ap.add_argument(
        "--n-profiles",
        type=int,
        default=len(TUNING_PROFILES),
        help="use only the first N tuning profiles (smoke tests; the committed sweep uses all)",
    )
    ap.add_argument("--obs-modes", nargs="+", default=list(TUNED_OBS_MODES))
    ap.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="worker processes; episodes are independent, so this only changes the wall clock",
    )
    args = ap.parse_args()

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / CSV_NAME

    if args.check:
        logged = pd.read_csv(csv_path)
        selected = select_best(logged)
        print(f"selection re-derived from {csv_path.relative_to(REPO_ROOT)}:")
        print(format_selection(selected))
        print(report_candidates(logged))
        mismatched = {k: v for k, v in selected.items() if baselines.TUNED.get(k) != v}
        if mismatched:
            print("\nMISMATCH against baselines.TUNED:")
            for key, params in sorted(mismatched.items()):
                print(f"  {key}: logged {params!r} vs code {baselines.TUNED.get(key)!r}")
            return 1
        print("\nbaselines.TUNED matches the logged sweep.")
        return 0

    profiles = TUNING_PROFILES[: args.n_profiles]
    print(
        f"tuning on {TUNING_SPLIT} profiles {list(profiles)} "
        f"(test split untouched), obs modes {args.obs_modes}, jobs {args.jobs}",
        flush=True,
    )
    df = sweep(args.obs_modes, profiles, jobs=args.jobs)
    df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)  # the selection is made from the logged file, not memory

    selected = select_best(df)
    print(f"\nwrote {csv_path.relative_to(REPO_ROOT)} ({len(df)} episodes)")
    print("\nselected by the rules in kelvin_rl.control.tuning.selection_rules:")
    print(format_selection(selected))
    print(report_candidates(df))
    return 0


def report_candidates(df: pd.DataFrame) -> str:
    """Per-rule shortlists, so the choice can be seen rather than taken on trust."""
    lines = ["\nshortlists, aggregated over the tuning profiles:"]
    for rule in selection_rules():
        if not GRIDS[rule.controller]:
            continue
        for obs_mode in sorted(set(df.loc[df["controller"] == rule.sweep_controller, "obs_mode"])):
            agg = aggregate(df, rule, obs_mode)
            safe = agg[agg["degree_s_above"] == 0.0]
            limit = rule.max_worst_peak_C
            constraint = (
                "no constraint on the worst peak"
                if limit is None
                else f"worst tuning peak required at or below {limit:.2f} C: "
                f"{int((safe['worst_peak_T_C'] <= limit).sum())} of {len(safe)} qualify"
            )
            lines.append(
                f"  {rule.controller}/{obs_mode}: {len(agg)} candidates, {len(safe)} with zero "
                f"degree-seconds above T_max on all tuning profiles; {constraint}"
            )
            try:
                shortlist = rank(agg, rule).head(5)
            except ValueError as exc:  # the rule rules everything out: say so
                lines.append(f"    NO VIABLE CANDIDATE: {exc}")
                continue
            lines.append(shortlist.round(3).to_string(index=False))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

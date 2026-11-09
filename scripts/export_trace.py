"""Export one episode as a ``kelvin.trace/1`` file for the Kelvin viewer.

Runs a single episode of :class:`BatteryPackCoolingEnv` under one controller and
writes every per-step quantity the environment logs to
``results/traces/<name>.json``, then rebuilds ``results/traces/manifest.json``
from the trace files on disk so lab mode's selector is generated rather than
hardcoded.

Every number in a trace is either copied straight out of the environment's
``info`` dict or derived from it by one explicit step: a unit change, a sum over
the episode, or the inverse of the observation scaling documented in
``battery_pack.py``. Nothing is hand-typed, and nothing is interpolated or
resampled. Blocks whose data does not exist (``series.estimate`` without a
Kalman filter, ``series.lambda`` without a Lagrange multiplier) are **absent,
not null-filled**, so the viewer's layer has nothing to draw and hides itself.

Traces are written uncompressed. Gzip is the viewer's shipping format, but
compressing here would make the files opaque to ``jq``, to diffing and to the
tests; compression belongs to the deploy step that copies ``results/traces``
into the static build.

Run with, for example::

    python scripts/export_trace.py --controller constant --action 0.3 --obs-mode partial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from kelvin_rl.envs import OBS_MODES, BatteryPackCoolingEnv

SCHEMA = "kelvin.trace/1"
MANIFEST_SCHEMA = "kelvin.trace.manifest/1"

# Anchored on this file rather than on ``kelvin_rl.config.REPO_ROOT``: in a git
# worktree the installed package may live in another checkout, and a trace must
# record the commit and config of the tree that produced it.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "env_default.yaml"
DEFAULT_TRACE_DIR = REPO_ROOT / "results" / "traces"

# trace series key -> env info key.
SCALAR_FIELDS = {
    "T_peak_substep_C": "T_peak_substep",
    "mdot": "mdot",
    "mdot_cmd": "mdot_cmd",
    "rate_limited": "rate_limited",
    "pump_energy_J": "pump_energy_J",
    "cost": "cost",
    "violation": "violation",
    "spread_K": "spread",
}
VECTOR_FIELDS = {"T_cells_C": "T_cells", "T_coolant_C": "T_coolant"}
ENERGY_FIELDS = {
    "gen": "energy_gen_J",
    "cool": "energy_cool_J",
    "amb": "energy_amb_J",
    "cond": "energy_cond_J",
    "stored_change": "stored_change_J",
}
# Logged by the Kalman filter (M4); see ``estimate_block``.
ESTIMATE_FIELDS = {"mean_C": "estimate_mean_C", "var_K2": "estimate_var_K2"}


# --------------------------------------------------------------- controllers
@runtime_checkable
class Controller(Protocol):
    """A controller is a named callable from observation to action.

    ``name`` identifies the run in ``meta.run.controller``, in the trace
    filename and in the manifest, so it carries any parameterisation with it.
    Two optional attributes are honoured if present and ignored if absent:

    * ``initial_action`` -- the flow setting before the first control step.
    * ``lambda_value()`` -- the current Lagrange multiplier. A controller that
      defines it gets a ``series.lambda``; one that does not has no such key.
    """

    name: str

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ConstantAction:
    """Hold one action for the whole episode: the M1 baseline.

    The episode also starts from this action, as ``scripts/plot_constant_flow.py``
    does, so the actuator rate limit does not blur the first few steps.
    """

    action: float

    @property
    def name(self) -> str:
        return f"constant_a{self.action:.2f}"

    @property
    def initial_action(self) -> float:
        return self.action

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.action], dtype=np.float32)


# M2 baselines (thermostat, PID) and M3 policies drop in here; they only have to
# satisfy the Controller protocol above.
CONTROLLERS = {"constant": lambda args: ConstantAction(args.action)}


def make_controller(args: argparse.Namespace) -> Controller:
    if args.controller not in CONTROLLERS:
        raise ValueError(f"unknown controller '{args.controller}'; have {sorted(CONTROLLERS)}")
    return CONTROLLERS[args.controller](args)


# ------------------------------------------------------------------- rollout
@dataclass
class Rollout:
    """What one episode produced. Nothing here is aggregated yet."""

    infos: list[dict] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    observations: list[np.ndarray] = field(default_factory=list)
    lambdas: list[float] | None = None
    current: np.ndarray = field(default_factory=lambda: np.empty(0))


def rollout(
    env: BatteryPackCoolingEnv, controller: Controller, *, seed: int, profile_index: int
) -> Rollout:
    """Run one episode, keeping the post-step info, reward and observation."""
    options: dict = {"profile_index": profile_index}
    initial_action = getattr(controller, "initial_action", None)
    if initial_action is not None:
        options["action_init"] = float(initial_action)

    obs, _ = env.reset(seed=seed, options=options)
    rec = Rollout(
        lambdas=[] if hasattr(controller, "lambda_value") else None, current=env.current.copy()
    )
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(controller(obs))
        rec.infos.append(info)
        rec.rewards.append(float(reward))
        rec.observations.append(np.asarray(obs, dtype=float))
        if rec.lambdas is not None:
            rec.lambdas.append(float(controller.lambda_value()))
        done = terminated or truncated
    return rec


# ------------------------------------------------------------ trace assembly
def _scalars(infos: list[dict], key: str) -> list:
    values = [i[key] for i in infos]
    if values and isinstance(values[0], (bool, np.bool_)):
        return [bool(v) for v in values]
    return [float(v) for v in values]


def _vectors(infos: list[dict], key: str) -> list[list[float]]:
    return [np.asarray(i[key], dtype=float).tolist() for i in infos]


def sensor_readings(
    env: BatteryPackCoolingEnv, obs: np.ndarray
) -> tuple[list[float], float, float]:
    """Undo the observation scaling to recover what the sensors reported, in degC.

    The environment never puts its sensor readings in ``info``; they exist only
    inside the observation, as ``(T - T_ref) / T_scale``. Inverting that is the
    only way to log what the controller actually saw, noise included. Under
    ``partial_history`` the newest frame is the last base block of the stack.
    Under ``full`` the environment adds no sensor noise, so these are the true
    temperatures at the sensed cells.
    """
    n_temps = env.n if env.obs_mode == "full" else len(env.sensor_idx)
    base = np.asarray(obs, dtype=float)[-(n_temps + 5) :]
    temps = base[:n_temps] * env.T_scale + env.T_ref
    T_in, T_out = base[n_temps : n_temps + 2] * env.T_scale + env.T_ref
    cells = temps if env.obs_mode != "full" else temps[env.sensor_idx]
    return [float(v) for v in cells], float(T_in), float(T_out)


def estimate_block(obs_mode: str, infos: list[dict]) -> dict | None:
    """``series.estimate``, or ``None`` when the run has no estimator.

    Only ``obs_mode='estimated'`` has a Kalman filter behind it, and that mode
    arrives with M4. Any other mode returns ``None`` and the caller omits the
    key entirely -- it is never written as null. When the mode *is* estimated, a
    missing field is an error rather than a zero: inventing a variance would
    draw an uncertainty shell that no filter ever reported.
    """
    if obs_mode != "estimated":
        return None
    missing = sorted(k for k in ESTIMATE_FIELDS.values() if any(k not in i for i in infos))
    if missing:
        raise KeyError(f"obs_mode='estimated' but the env logged no {missing}; cannot fill them in")
    return {name: _vectors(infos, key) for name, key in ESTIMATE_FIELDS.items()}


def git_commit(root: Path = REPO_ROOT) -> str:
    """Short HEAD SHA of the tree holding this script. Mandatory, so it raises."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def config_sha256(path: Path) -> str:
    """sha256 of the resolved config file's bytes, as the viewer displays it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_trace(
    env: BatteryPackCoolingEnv,
    controller: Controller,
    rec: Rollout,
    *,
    seed: int,
    split: str,
    config_path: Path,
) -> dict:
    """Assemble one ``kelvin.trace/1`` document from a finished rollout."""
    infos, n_steps = rec.infos, len(rec.infos)
    if n_steps == 0:
        raise ValueError("the rollout produced no steps")
    p = env.nominal_params  # T_in, T_amb and P_pump_max are not perturbed

    series: dict = {
        # Simulated time at the end of each control step: every series below
        # records the state after that step, not before it.
        "t_s": [(k + 1) * env.dt_ctrl for k in range(n_steps)],
        # The load profile sample the environment read at the start of the step,
        # taken at the profile's own dt_phys resolution and not averaged.
        "I_A": [float(rec.current[k * env.substeps]) for k in range(n_steps)],
    }
    series.update({name: _vectors(infos, key) for name, key in VECTOR_FIELDS.items()})
    series.update({name: _scalars(infos, key) for name, key in SCALAR_FIELDS.items()})
    series["energy_J"] = {name: _scalars(infos, key) for name, key in ENERGY_FIELDS.items()}

    readings = [sensor_readings(env, obs) for obs in rec.observations]
    series["sensors"] = {
        "cells_C": [r[0] for r in readings],
        "T_in_C": [r[1] for r in readings],
        "T_out_C": [r[2] for r in readings],
    }
    estimate = estimate_block(env.obs_mode, infos)
    if estimate is not None:
        series["estimate"] = estimate
    if rec.lambdas is not None:
        series["lambda"] = [float(v) for v in rec.lambdas]

    # Summaries are read back off the series above, so a trace cannot disagree
    # with itself. The definitions match scripts/plot_constant_flow.py.
    summary = {
        "pump_energy_kJ": sum(series["pump_energy_J"]) / 1e3,
        "peak_T_C": max(series["T_peak_substep_C"]),
        "pct_steps_violating": 100.0 * sum(series["violation"]) / n_steps,
        "degree_s_above": sum(series["cost"]) * env.dt_ctrl,
        "max_spread_K": max(series["spread_K"]),
        "return": sum(rec.rewards),  # undiscounted sum of the env's rewards
        "cost_total": sum(series["cost"]),
    }

    return {
        "schema": SCHEMA,
        "meta": {
            "git_commit": git_commit(),
            "config_sha256": config_sha256(config_path),
            "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "env": {
                "n_cells": env.n,
                "T_max_C": env.T_max,
                "dT_max_K": env.dT_max,
                "mdot_min": env.mdot_min,
                "mdot_max": env.mdot_max,
                "delta_mdot_max": env.delta_mdot_max,
                "T_in_C": p.T_in,
                "T_amb_C": p.T_amb,
                "P_pump_max_W": p.P_pump_max,
                "dt_ctrl_s": env.dt_ctrl,
                "dt_phys_s": env.dt_phys,
                "sensor_cells": [int(i) for i in env.sensor_idx],
            },
            "run": {
                "controller": controller.name,
                "obs_mode": env.obs_mode,
                "seed": int(seed),
                "split": split,
                "profile_index": int(infos[-1]["profile_index"]),
                "mismatch_scale": float(env.cfg["mismatch"]["scale"]),
            },
        },
        "episode": {"n_steps": n_steps, "summary": summary},
        "series": series,
    }


def trace_name(trace: dict) -> str:
    """Default file stem, in the style of the design document's ``source_csv``."""
    run = trace["meta"]["run"]
    return (
        f"{run['controller']}_{run['obs_mode']}_seed{run['seed']}"
        f"_{run['split']}{run['profile_index']:02d}"
    )


# ------------------------------------------------------------------ manifest
def manifest_entry(trace: dict, file_name: str) -> dict:
    meta, run = trace["meta"], trace["meta"]["run"]
    return {
        "file": file_name,
        "controller": run["controller"],
        "obs_mode": run["obs_mode"],
        "seed": run["seed"],
        "split": run["split"],
        "profile_index": run["profile_index"],
        "mismatch_scale": run["mismatch_scale"],
        "n_steps": trace["episode"]["n_steps"],
        "git_commit": meta["git_commit"],
        "config_sha256": meta["config_sha256"],
        "exported_utc": meta["exported_utc"],
    }


def rebuild_manifest(directory: Path) -> dict:
    """Index every ``kelvin.trace/1`` file in ``directory`` into ``manifest.json``.

    The manifest is rebuilt from the files on disk rather than appended to, so
    adding, overwriting or deleting a trace all leave it correct and it can
    never claim a trace that is not there. Files of another schema are skipped.
    """
    entries = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "manifest.json":
            continue
        with open(path, encoding="utf-8") as f:
            trace = json.load(f)
        if trace.get("schema") != SCHEMA:
            continue
        entries.append(manifest_entry(trace, path.name))
    entries.sort(
        key=lambda e: (e["controller"], e["obs_mode"], e["split"], e["profile_index"], e["seed"])
    )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "traces": entries,
    }
    write_json(Path(directory) / "manifest.json", manifest)
    return manifest


def write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=1)
        f.write("\n")
    return path


def export(args: argparse.Namespace) -> tuple[Path, dict]:
    """Run the episode and write the trace, then the manifest beside it."""
    controller = make_controller(args)
    env = BatteryPackCoolingEnv(config=args.config, obs_mode=args.obs_mode, split=args.split)
    rec = rollout(env, controller, seed=args.seed, profile_index=args.profile_index)
    trace = build_trace(
        env, controller, rec, seed=args.seed, split=args.split, config_path=args.config
    )

    out = Path(args.out) if args.out else DEFAULT_TRACE_DIR / f"{trace_name(trace)}.json"
    write_json(out, trace)
    rebuild_manifest(out.parent)
    return out, trace


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--controller", default="constant", choices=sorted(CONTROLLERS))
    ap.add_argument(
        "--action", type=float, default=0.3, help="constant controller: the held action in [0, 1]"
    )
    ap.add_argument("--obs-mode", default="partial", choices=list(OBS_MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile-index", type=int, default=7)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--out", type=Path, default=None, help=f"default: {DEFAULT_TRACE_DIR}/<name>.json"
    )
    return ap


def main(argv: list[str] | None = None):
    out, trace = export(build_parser().parse_args(argv))
    s = trace["episode"]["summary"]
    print(f"wrote {out} ({trace['episode']['n_steps']} steps, commit {trace['meta']['git_commit']})")
    print(f"      peak {s['peak_T_C']:.2f} C, {s['pct_steps_violating']:.1f}% of steps violating, "
          f"{s['degree_s_above']:.1f} K.s above T_max, pump {s['pump_energy_kJ']:.2f} kJ, "
          f"return {s['return']:.2f}")
    print(f"      manifest {out.parent / 'manifest.json'}")


if __name__ == "__main__":
    main()

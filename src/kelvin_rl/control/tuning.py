"""The grids and the selection rules used to tune the classical baselines.

Kept next to the controllers, and separate from the sweep script, so that the
script that *produces* ``results/m2_tuning_sweep.csv`` and the test that checks
the defaults in :mod:`kelvin_rl.control.baselines` came out of that file apply
exactly the same rules.

The prose version of the procedure is in the :mod:`kelvin_rl.control.baselines`
docstring. The things worth stating here:

* **Training profiles only.** ``TUNING_SPLIT`` is the train split and
  ``TUNING_PROFILES`` are fixed indices into it. The held-out test split never
  enters the tuning loop.
* **Rules applied to logged data.** :func:`select_best` reads the sweep CSV and
  sorts candidates lexicographically by total degree-seconds above ``T_max``,
  then by mean pump energy, then by the parameter values so the outcome is
  deterministic. Nothing is chosen by eye.

Two of the three rules in :func:`selection_rules` are that bare lexicographic
sort. The third, for ``thermostat_margin``, adds a feasibility constraint on the
worst peak temperature over the tuning profiles. It was added *after* the
plain thermostat was seen to violate ``T_max`` on held-out profiles, so it is
not pre-registered; :mod:`kelvin_rl.control.baselines` says so in full. The
constraint is deliberately not a free parameter: the limit comes from the
configured sensor noise (:func:`margin_limit_C`), so there is nothing left to
tune by hand, and if no logged candidate satisfies it the selection raises
instead of quietly relaxing it.

``thermostat_margin`` is the same controller on the same grid as ``thermostat``,
so both rules read the same logged episodes (see :data:`SWEEP_CONTROLLER`) and
only a controller in :data:`SWEEP_CONTROLLERS` is ever simulated.

The grids are deliberately coarse and wide rather than centred on a hand-picked
starting point: the gain grids span a decade and a half in logarithmic steps,
and the margins span 1-9 K below ``T_max``, which is wider than the pack's
cell-to-cell spread. That way the sweep, not the author, decides the order of
magnitude.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kelvin_rl.config import load_config
from kelvin_rl.control.baselines import PIDParams, ThermostatParams

TUNING_SPLIT = "train"
TUNING_PROFILES = tuple(range(10))
TUNED_OBS_MODES = ("full", "partial")

THERMOSTAT_GRID = {
    "on_margin_K": (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0),
    "band_K": (0.5, 1.0, 2.0, 3.0),
}

GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    # Constant maximum flow has nothing to tune: the action is 1.0 by definition.
    "constant_max": {},
    "thermostat": THERMOSTAT_GRID,
    "thermostat_margin": THERMOSTAT_GRID,  # same controller, same grid, stricter rule
    "pid": {
        "setpoint_margin_K": (1.0, 3.0, 5.0, 7.0, 9.0),
        "Kp": (0.05, 0.1, 0.2, 0.5),
        "Ki": (0.0, 0.002, 0.01),
        "Kd": (0.0, 1.0, 4.0),
    },
}

PARAM_CLASSES = {
    "thermostat": ThermostatParams,
    "thermostat_margin": ThermostatParams,
    "pid": PIDParams,
}

# Controllers whose rows the sweep actually simulates. A controller that differs
# from one of these only in its selection rule reads those same logged episodes.
SWEEP_CONTROLLER = {"thermostat_margin": "thermostat"}
SWEEP_CONTROLLERS = ("constant_max", "thermostat", "pid")

# The lexicographic selection rule, as (column, ascending) pairs.
SELECTION_KEYS = (("degree_s_above", True), ("pump_energy_kJ", True))

# How much margin the thermostat_margin rule demands, in sensor standard
# deviations. Two sigma of the cell sensors, so the switch-on point is far
# enough below T_max that a noisy reading alone cannot sit on the limit.
MARGIN_SIGMAS = 2.0


def margin_limit_C(cfg: dict | None = None) -> float:
    """The worst tuning peak the ``thermostat_margin`` rule tolerates, in degC.

    Derived from the configured sensor noise rather than written down, so it
    follows ``configs/env_default.yaml`` instead of pinning a number here.
    """
    cfg = cfg if cfg is not None else load_config()
    return float(cfg["safety"]["T_max"]) - MARGIN_SIGMAS * float(cfg["sensors"]["sigma_cell"])


@dataclass(frozen=True)
class SelectionRule:
    """How one controller's parameters are picked out of the sweep log.

    ``max_worst_peak_C`` is an extra feasibility constraint on the hottest
    temperature reached on any tuning profile; ``None`` means the bare
    lexicographic sort.
    """

    controller: str
    max_worst_peak_C: float | None = None

    @property
    def sweep_controller(self) -> str:
        return SWEEP_CONTROLLER.get(self.controller, self.controller)


def selection_rules(cfg: dict | None = None) -> tuple[SelectionRule, ...]:
    return (
        SelectionRule("thermostat"),
        SelectionRule("thermostat_margin", max_worst_peak_C=margin_limit_C(cfg)),
        SelectionRule("pid"),
    )


def candidates(name: str) -> list[dict[str, float]]:
    """Every point of the grid for one controller, as keyword dicts."""
    grid = GRIDS[name]
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def build_params(name: str, values: dict[str, float]):
    """Turn one grid point into the controller's parameter dataclass."""
    if name not in PARAM_CLASSES:
        return None
    return PARAM_CLASSES[name](**{k: float(v) for k, v in values.items()})


def aggregate(df: pd.DataFrame, rule: SelectionRule, obs_mode: str) -> pd.DataFrame:
    """One row per grid point: its total cost, mean energy and worst peak."""
    keys = list(GRIDS[rule.controller])
    sub = df[(df["controller"] == rule.sweep_controller) & (df["obs_mode"] == obs_mode)]
    return sub.groupby(keys, as_index=False).agg(
        degree_s_above=("degree_s_above", "sum"),
        pump_energy_kJ=("pump_energy_kJ", "mean"),
        worst_peak_T_C=("peak_T_C", "max"),
    )


def rank(agg: pd.DataFrame, rule: SelectionRule) -> pd.DataFrame:
    """Feasible candidates in selection order, best first.

    Raises if the rule's constraint rules every candidate out: a sweep that
    cannot satisfy its own rule is a result to report, not a reason to widen
    the grid.
    """
    keys = list(GRIDS[rule.controller])
    feasible = agg
    if rule.max_worst_peak_C is not None:
        feasible = agg[agg["worst_peak_T_C"] <= rule.max_worst_peak_C]
        if not len(feasible):
            raise ValueError(
                f"no candidate in the {rule.controller} grid keeps the worst tuning peak "
                f"at or below {rule.max_worst_peak_C:.2f} C "
                f"(best was {agg['worst_peak_T_C'].min():.2f} C). "
                "Report that, rather than widening the grid until something wins."
            )
    columns = [c for c, _ in SELECTION_KEYS] + keys
    ascending = [a for _, a in SELECTION_KEYS] + [True] * len(keys)
    return feasible.sort_values(columns, ascending=ascending)


def select_best(df: pd.DataFrame, cfg: dict | None = None) -> dict[tuple[str, str], object]:
    """Pick one parameter set per (controller, obs_mode) from logged episodes.

    ``df`` is the per-episode sweep log: one row per (simulated controller,
    obs_mode, grid point, profile). Each rule in :func:`selection_rules`
    aggregates its controller's episodes over the tuning profiles - degree-
    seconds summed, pump energy averaged, peak maximised - and takes the first
    feasible candidate in :func:`rank` order. Controllers with an empty grid
    have nothing to select and get no entry.
    """
    selected: dict[tuple[str, str], object] = {}
    for rule in selection_rules(cfg):
        keys = list(GRIDS[rule.controller])
        if not keys:
            continue
        for obs_mode in sorted(set(df.loc[df["controller"] == rule.sweep_controller, "obs_mode"])):
            best = rank(aggregate(df, rule, obs_mode), rule).iloc[0]
            params = build_params(rule.controller, {k: best[k] for k in keys})
            selected[(rule.controller, obs_mode)] = params
    return selected


def select_best_from_csv(path: str | Path, cfg: dict | None = None) -> dict[tuple[str, str], object]:
    """:func:`select_best` applied to a sweep CSV on disk."""
    return select_best(pd.read_csv(path), cfg)

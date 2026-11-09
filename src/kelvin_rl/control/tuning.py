"""The grids and the selection rule used to tune the classical baselines.

Kept next to the controllers, and separate from the sweep script, so that the
script that *produces* ``results/m2_tuning_sweep.csv`` and the test that checks
the defaults in :mod:`kelvin_rl.control.baselines` came out of that file apply
exactly the same rule.

The prose version of the procedure is in the :mod:`kelvin_rl.control.baselines`
docstring. The two things worth stating here:

* **Training profiles only.** ``TUNING_SPLIT`` is the train split and
  ``TUNING_PROFILES`` are fixed indices into it. The held-out test split never
  enters the tuning loop.
* **One rule, applied to logged data.** :func:`select_best` reads the sweep CSV
  and sorts candidates lexicographically by total degree-seconds above
  ``T_max``, then by mean pump energy, then by the parameter values so the
  outcome is deterministic. Nothing is chosen by eye.

The grids are deliberately coarse and wide rather than centred on a hand-picked
starting point: the gain grids span a decade and a half in logarithmic steps,
and the margins span 1-9 K below ``T_max``, which is wider than the pack's
cell-to-cell spread. That way the sweep, not the author, decides the order of
magnitude.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from kelvin_rl.control.baselines import PIDParams, ThermostatParams

TUNING_SPLIT = "train"
TUNING_PROFILES = tuple(range(10))
TUNED_OBS_MODES = ("full", "partial")

GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    # Constant maximum flow has nothing to tune: the action is 1.0 by definition.
    "constant_max": {},
    "thermostat": {
        "on_margin_K": (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0),
        "band_K": (0.5, 1.0, 2.0, 3.0),
    },
    "pid": {
        "setpoint_margin_K": (1.0, 3.0, 5.0, 7.0, 9.0),
        "Kp": (0.05, 0.1, 0.2, 0.5),
        "Ki": (0.0, 0.002, 0.01),
        "Kd": (0.0, 1.0, 4.0),
    },
}

PARAM_CLASSES = {"thermostat": ThermostatParams, "pid": PIDParams}

# The lexicographic selection rule, as (column, ascending) pairs.
SELECTION_KEYS = (("degree_s_above", True), ("pump_energy_kJ", True))


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


def select_best(df: pd.DataFrame) -> dict[tuple[str, str], object]:
    """Pick one parameter set per (controller, obs_mode) from logged episodes.

    ``df`` is the per-episode sweep log: one row per (controller, obs_mode,
    grid point, profile). Candidates are aggregated over the tuning profiles -
    degree-seconds summed, pump energy averaged - and then sorted by
    :data:`SELECTION_KEYS`, with the parameter columns as the final tie-break.
    Controllers with an empty grid are skipped, having nothing to select.
    """
    selected: dict[tuple[str, str], object] = {}
    for (name, obs_mode), sub in df.groupby(["controller", "obs_mode"], sort=True):
        keys = list(GRIDS.get(name, {}))
        if not keys:
            continue
        agg = sub.groupby(keys, as_index=False).agg(
            degree_s_above=("degree_s_above", "sum"),
            pump_energy_kJ=("pump_energy_kJ", "mean"),
        )
        columns = [c for c, _ in SELECTION_KEYS] + keys
        ascending = [a for _, a in SELECTION_KEYS] + [True] * len(keys)
        best = agg.sort_values(columns, ascending=ascending).iloc[0]
        selected[(name, obs_mode)] = build_params(name, {k: best[k] for k in keys})
    return selected


def select_best_from_csv(path: str | Path) -> dict[tuple[str, str], object]:
    """:func:`select_best` applied to a sweep CSV on disk."""
    return select_best(pd.read_csv(path))

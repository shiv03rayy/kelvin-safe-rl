"""Lumped-parameter thermal model of a battery pack on one coolant channel.

Everything here is a pure function of a :class:`PackParams` and arrays, so the
environment, the Kalman filter and the tests all use exactly the same physics.

Cell i (0-based, coolant flows from cell 0 to cell N-1)::

    C_i dT_i/dt = Q_gen,i - G(mdot) (T_i - T_c,i)
                  - k_cond (2 T_i - T_{i-1} - T_{i+1})       (one neighbour at the ends)
                  - h_amb A_amb (T_i - T_amb)

Coolant segment i (quasi-steady: the coolant has no heat capacity of its own)::

    mdot cp (T_c,i - T_c,i-1) = G(mdot) (T_i - T_c,i),     T_c,-1 = T_in

with G(mdot) = h_eff(mdot) A and h_eff = h0 (mdot / mdot_ref)^0.8.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PackParams:
    """Physical parameters of one pack. Arrays are per cell, shape (n_cells,)."""

    n_cells: int
    C_cell: np.ndarray  # J/K
    R: np.ndarray  # ohm
    k_cond: float  # W/K
    h_amb: float  # W/(m^2 K)
    A_amb: float  # m^2
    T_amb: float  # degC
    cp: float  # J/(kg K)
    h0: float  # W/(m^2 K)
    A: float  # m^2
    mdot_ref: float  # kg/s
    flow_exponent: float
    T_in: float  # degC
    P_pump_max: float  # W
    mdot_max: float  # kg/s

    @classmethod
    def from_config(cls, cfg: dict) -> "PackParams":
        """Nominal parameters: what the Kalman filter and model-based baselines believe."""
        p, c = cfg["pack"], cfg["coolant"]
        n = int(p["n_cells"])
        return cls(
            n_cells=n,
            C_cell=np.full(n, float(p["C_cell"])),
            R=np.full(n, float(p["R_nominal"])),
            k_cond=float(p["k_cond"]),
            h_amb=float(p["h_amb"]),
            A_amb=float(p["A_amb"]),
            T_amb=float(p["T_amb"]),
            cp=float(c["cp"]),
            h0=float(c["h0"]),
            A=float(c["A"]),
            mdot_ref=float(c["mdot_ref"]),
            flow_exponent=float(c["flow_exponent"]),
            T_in=float(c["T_in"]),
            P_pump_max=float(c["P_pump_max"]),
            mdot_max=float(c["mdot_max"]),
        )

    def perturbed(self, mismatch: dict) -> "PackParams":
        """The 'true' plant: nominal parameters with the configured mismatch applied.

        Sampled from its own seed so the plant is fixed across training seeds.
        With ``scale = 0`` this returns the nominal model exactly.
        """
        s = float(mismatch["scale"])
        rng = np.random.default_rng(int(mismatch["seed"]))
        h0_factor = 1.0 + rng.uniform(-1, 1) * mismatch["h0_rel"] * s
        C_factor = 1.0 + rng.uniform(-1, 1) * mismatch["C_cell_rel"] * s
        R_factor = 1.0 + rng.standard_normal(self.n_cells) * mismatch["R_spread_rel"] * s
        return replace(
            self,
            h0=self.h0 * h0_factor,
            C_cell=self.C_cell * C_factor,
            R=self.R * np.clip(R_factor, 0.5, 1.5),
        )


def h_eff(params: PackParams, mdot: float) -> float:
    """Dittus-Boelter-style scaling, h ~ mdot^0.8. An approximation, not a correlation fit."""
    return params.h0 * (mdot / params.mdot_ref) ** params.flow_exponent


def pump_power(params: PackParams, mdot: float) -> float:
    """Pump affinity laws: shaft power scales with the cube of flow."""
    return params.P_pump_max * (mdot / params.mdot_max) ** 3


def coolant_temperatures(params: PackParams, T: np.ndarray, mdot: float) -> np.ndarray:
    """Coolant temperature leaving each segment, marching from inlet to outlet.

    Solving the segment balance for T_c,i gives a weighted average of the
    upstream coolant and the cell, so it is always bounded and needs no sub-steps::

        T_c,i = (mdot cp T_c,i-1 + G T_i) / (mdot cp + G)
    """
    G = h_eff(params, mdot) * params.A
    w = mdot * params.cp
    T_c = np.empty_like(T, dtype=float)
    upstream = params.T_in
    for i in range(params.n_cells):
        upstream = (w * upstream + G * T[i]) / (w + G)
        T_c[i] = upstream
    return T_c


def heat_flows(
    params: PackParams, T: np.ndarray, mdot: float, I: float
) -> dict[str, np.ndarray]:
    """Per-cell heat flows in watts. Positive means heat leaving the cell (except q_gen)."""
    T_c = coolant_temperatures(params, T, mdot)
    G = h_eff(params, mdot) * params.A
    left = np.concatenate(([T[0]], T[:-1]))  # edge cells: neighbour term vanishes
    right = np.concatenate((T[1:], [T[-1]]))
    return {
        "q_gen": I**2 * params.R,
        "q_cool": G * (T - T_c),
        "q_cond": params.k_cond * (2 * T - left - right),
        "q_amb": params.h_amb * params.A_amb * (T - params.T_amb),
        "T_c": T_c,
    }


def derivatives(params: PackParams, T: np.ndarray, mdot: float, I: float) -> np.ndarray:
    """dT/dt for every cell, in K/s."""
    q = heat_flows(params, T, mdot, I)
    return (q["q_gen"] - q["q_cool"] - q["q_cond"] - q["q_amb"]) / params.C_cell


def euler_step(
    params: PackParams, T: np.ndarray, mdot: float, I: float, dt: float
) -> tuple[np.ndarray, dict[str, float]]:
    """One explicit Euler step. Also returns the energy (J) in each flow over the step.

    The energies are computed from the same flows used in the update, so the
    discrete energy balance holds to floating-point precision.
    """
    q = heat_flows(params, T, mdot, I)
    dT = (q["q_gen"] - q["q_cool"] - q["q_cond"] - q["q_amb"]) / params.C_cell
    energy = {
        "gen": float(q["q_gen"].sum() * dt),
        "cool": float(q["q_cool"].sum() * dt),
        "amb": float(q["q_amb"].sum() * dt),
        "cond": float(q["q_cond"].sum() * dt),  # should be ~0: conduction is internal
    }
    return T + dt * dT, energy


def system_matrix(params: PackParams, mdot: float) -> np.ndarray:
    """Continuous-time matrix A in dT/dt = A T + (inputs), for a fixed flow.

    For fixed mdot the model is linear in T, so column j is simply the response
    to a unit temperature on cell j with all inputs (current, T_in, T_amb) at zero.
    The Kalman filter (M4) will use this same matrix.
    """
    zero = replace(params, T_in=0.0, T_amb=0.0)
    n = params.n_cells
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = derivatives(zero, e, mdot, I=0.0)
    return A


def euler_spectral_radius(params: PackParams, mdot: float, dt: float) -> float:
    """Spectral radius of the Euler update matrix I + dt A. Must be < 1 for stability."""
    M = np.eye(params.n_cells) + dt * system_matrix(params, mdot)
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def check_euler_stability(
    params: PackParams, mdot_min: float, mdot_max: float, dt: float, n_grid: int = 11
) -> float:
    """Raise if explicit Euler is unstable or oscillatory anywhere in the flow range.

    We require every eigenvalue of I + dt A to lie in (0, 1): stable and
    non-oscillating, so a cooling cell never overshoots below its coolant.
    Returns the worst-case spectral radius.
    """
    worst = 0.0
    for mdot in np.linspace(mdot_min, mdot_max, n_grid):
        eig = np.linalg.eigvals(np.eye(params.n_cells) + dt * system_matrix(params, mdot))
        if np.any(np.abs(eig) >= 1.0) or np.any(eig.real <= 0.0):
            raise ValueError(
                f"Explicit Euler with dt={dt}s is unstable or oscillatory at mdot={mdot:.4f} kg/s "
                f"(eigenvalues of I + dt A: {np.sort(eig.real)})"
            )
        worst = max(worst, float(np.max(np.abs(eig))))
    return worst

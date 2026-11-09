"""Physics sanity checks (handoff section 8). These must pass before any training."""

from dataclasses import replace

import numpy as np
import pytest

from kelvin_rl.envs import physics

from conftest import run_constant


@pytest.fixture
def params(cfg):
    return physics.PackParams.from_config(cfg)


def test_energy_balance_over_episode(make_env):
    """Heat generated = change in stored heat + heat to coolant + heat to ambient."""
    env = make_env()
    infos = run_constant(env, 0.3, seed=0, options={"profile_index": 0})
    gen = sum(i["energy_gen_J"] for i in infos)
    stored = sum(i["stored_change_J"] for i in infos)
    cool = sum(i["energy_cool_J"] for i in infos)
    amb = sum(i["energy_amb_J"] for i in infos)
    assert gen > 0
    assert gen == pytest.approx(stored + cool + amb, rel=1e-9, abs=1e-6)
    assert abs(sum(i["energy_cond_J"] for i in infos)) < 1e-6 * gen  # conduction is internal


def test_coolant_absorbs_exactly_what_cells_reject(params):
    """The coolant energy balance: mdot cp (T_out - T_in) = total heat to coolant."""
    T = np.linspace(30.0, 40.0, params.n_cells)
    mdot = 0.01
    q = physics.heat_flows(params, T, mdot, I=0.0)
    assert mdot * params.cp * (q["T_c"][-1] - params.T_in) == pytest.approx(q["q_cool"].sum())


def test_no_heat_converges_to_inlet_temperature(params):
    """With zero current and T_in = T_amb, every cell relaxes to that temperature."""
    p = replace(params, T_in=20.0, T_amb=20.0)
    T = np.full(p.n_cells, 40.0)
    for _ in range(20_000):  # 20 000 s, many thermal time constants
        T, _ = physics.euler_step(p, T, mdot=0.01, I=0.0, dt=1.0)
    np.testing.assert_allclose(T, 20.0, atol=1e-3)


def test_no_heat_stays_between_inlet_and_ambient(params):
    p = replace(params, T_in=20.0, T_amb=30.0)
    T = np.full(p.n_cells, 50.0)
    for _ in range(20_000):
        T, _ = physics.euler_step(p, T, mdot=0.01, I=0.0, dt=1.0)
    assert np.all(T > 20.0) and np.all(T < 30.0)


def test_downstream_cells_run_hotter_under_uniform_heating(params):
    T = np.full(params.n_cells, 25.0)
    for _ in range(5_000):
        T, _ = physics.euler_step(params, T, mdot=0.01, I=100.0, dt=1.0)
    assert np.all(np.diff(T) > 0), T


def test_higher_flow_lowers_peak_and_raises_pump_energy(make_env):
    env = make_env()
    peaks, energies = [], []
    for a in (0.1, 0.4, 0.7, 1.0):
        infos = run_constant(env, a, seed=0, options={"profile_index": 1, "action_init": a})
        peaks.append(max(i["T_peak_substep"] for i in infos))
        energies.append(sum(i["pump_energy_J"] for i in infos))
    assert np.all(np.diff(peaks) < 0), peaks
    assert np.all(np.diff(energies) > 0), energies


def test_heat_transfer_and_pump_scaling(params):
    assert physics.h_eff(params, params.mdot_ref) == pytest.approx(params.h0)
    assert physics.h_eff(params, 2 * params.mdot_ref) == pytest.approx(params.h0 * 2**0.8)
    assert physics.pump_power(params, params.mdot_max) == pytest.approx(params.P_pump_max)
    assert physics.pump_power(params, params.mdot_max / 2) == pytest.approx(params.P_pump_max / 8)


def test_euler_substep_is_stable_for_default_parameters(cfg, params):
    c = cfg["coolant"]
    rho = physics.check_euler_stability(params, c["mdot_min"], c["mdot_max"], dt=cfg["time"]["dt_phys"])
    assert rho < 1.0


def test_stability_check_rejects_a_huge_step(cfg, params):
    c = cfg["coolant"]
    with pytest.raises(ValueError):
        physics.check_euler_stability(params, c["mdot_min"], c["mdot_max"], dt=10_000.0)


def test_system_matrix_matches_derivatives(params):
    """dT/dt = A T exactly when all inputs are zero, so A is safe to reuse in the Kalman filter."""
    p = replace(params, T_in=0.0, T_amb=0.0)
    T = np.random.default_rng(0).normal(size=p.n_cells)
    np.testing.assert_allclose(physics.system_matrix(p, 0.01) @ T, physics.derivatives(p, T, 0.01, 0.0))


def test_zero_mismatch_gives_nominal_plant(cfg, params):
    true = params.perturbed({**cfg["mismatch"], "scale": 0.0})
    assert true.h0 == params.h0
    np.testing.assert_array_equal(true.R, params.R)
    np.testing.assert_array_equal(true.C_cell, params.C_cell)


def test_mismatch_is_fixed_by_its_seed(cfg, params):
    a = params.perturbed(cfg["mismatch"])
    b = params.perturbed(cfg["mismatch"])
    np.testing.assert_array_equal(a.R, b.R)
    assert a.h0 == b.h0 != params.h0

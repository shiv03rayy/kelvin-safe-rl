"""Classical baselines: action space, hysteresis, PID direction, observation modes.

These are behavioural tests, not performance tests: nothing here asserts that a
baseline is good, only that it does what its docstring says and stays inside
the interface the env offers. The one number-bearing test
(:func:`test_tuned_defaults_match_logged_sweep`) checks provenance - that the
tuned defaults really are what the logged training sweep selected.
"""

import numpy as np
import pytest

from kelvin_rl.config import REPO_ROOT
from kelvin_rl.control import baselines
from kelvin_rl.control.baselines import (
    BASELINES,
    ControllerSpec,
    PIDController,
    PIDParams,
    ThermostatController,
    ThermostatParams,
    make_controller,
)
from kelvin_rl.control.tuning import margin_limit_C, select_best_from_csv
from kelvin_rl.eval.metrics import episode_metrics, run_episode

MODES = ["full", "partial", "partial_history"]
TUNING_CSV = REPO_ROOT / "results" / "m2_tuning_sweep.csv"
EVALUATION_CSV = REPO_ROOT / "results" / "m2_baselines_test.csv"


def synthetic_obs(env, temps, *, T_in=30.0, T_out=35.0, current=0.0, prev_action=0.5, t=0.0):
    """One observation frame built the way the env builds them, for unit tests.

    Lets a controller be driven by a chosen temperature sequence without a
    simulation in the loop.
    """
    spec = ControllerSpec.from_env(env)
    temps = np.broadcast_to(np.asarray(temps, dtype=float), (spec.n_temps,))
    scaled = (np.concatenate([temps, [T_in, T_out]]) - env.T_ref) / env.T_scale
    return np.concatenate([scaled, [current / env.I_scale, prev_action, t]]).astype(np.float32)


# --------------------------------------------------------------- the interface
@pytest.mark.parametrize("name", BASELINES)
@pytest.mark.parametrize("mode", MODES)
def test_actions_stay_inside_the_action_space(make_env, name, mode):
    env = make_env(obs_mode=mode)
    controller = make_controller(name, env)
    infos, actions = run_episode(env, controller, seed=0, profile_index=0)
    assert len(infos) == env.n_ctrl_steps
    for action in actions:
        boxed = np.array([action], dtype=np.float32)
        assert env.action_space.contains(boxed), action


@pytest.mark.parametrize("name", BASELINES)
@pytest.mark.parametrize("mode", ["full", "partial"])
def test_reset_clears_state_so_episodes_are_reproducible(make_env, name, mode):
    """Two runs of the same episode must agree, or internal state leaked across resets."""
    env = make_env(obs_mode=mode)
    controller = make_controller(name, env)
    _, first = run_episode(env, controller, seed=3, profile_index=1)
    _, second = run_episode(env, controller, seed=3, profile_index=1)
    np.testing.assert_allclose(first, second)


def test_unknown_baseline_is_rejected(make_env):
    with pytest.raises(ValueError):
        make_controller("bang_bang_but_spelled_wrong", make_env())


# ------------------------------------------------------- reading the observation
def test_full_mode_decoding_recovers_the_cell_temperatures(make_env):
    """The controller works in degrees Celsius, so the inverse scaling must be right."""
    env = make_env(obs_mode="full")
    obs, info = env.reset(seed=0)
    spec = ControllerSpec.from_env(env)
    np.testing.assert_allclose(spec.measured_temperatures_C(obs), info["T_cells"], atol=1e-3)
    assert spec.hottest_measured_C(obs) == pytest.approx(info["T_cells"].max(), abs=1e-3)


def test_partial_mode_only_sees_the_sensed_cells(make_env):
    """K noisy readings, and not the hottest (downstream) cell: that is the point."""
    env = make_env(obs_mode="partial")
    obs, info = env.reset(seed=0)
    spec = ControllerSpec.from_env(env)
    assert spec.n_temps == len(env.sensor_idx) < env.n
    sensed = info["T_cells"][env.sensor_idx]
    errors = spec.measured_temperatures_C(obs) - sensed
    assert np.all(np.abs(errors) < 6 * env.sigma_cell)  # noisy, but about the sensed cells
    assert env.n - 1 not in env.sensor_idx


def test_history_mode_decodes_the_newest_frame(make_env):
    env = make_env(obs_mode="partial_history")
    spec = ControllerSpec.from_env(env)
    obs, _ = env.reset(seed=0)
    assert len(obs) == spec.frame_dim * env.history_len
    np.testing.assert_allclose(spec.latest_frame(obs), obs[: spec.frame_dim])  # padded at reset
    obs, *_ = env.step(np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(spec.latest_frame(obs), obs[-spec.frame_dim :])
    assert not np.allclose(spec.latest_frame(obs), obs[: spec.frame_dim])


# ------------------------------------------------------------- constant max flow
def test_constant_max_commands_maximum_flow(make_env):
    env = make_env()
    controller = make_controller("constant_max", env)
    infos, actions = run_episode(env, controller, seed=0, profile_index=0)
    assert set(actions) == {1.0}
    ramp = int(np.ceil(1.0 / env.cfg["coolant"]["rate_limit_frac"]))  # env rate-limits the ramp
    assert all(i["mdot"] == pytest.approx(env.mdot_max) for i in infos[ramp:])
    assert infos[-1]["pump_energy_J"] == pytest.approx(env.params.P_pump_max * env.dt_ctrl)


def test_constant_max_sets_the_energy_ceiling(make_env):
    """No other baseline can burn more pump energy than constant maximum flow."""
    env = make_env()
    energies = {}
    for name in BASELINES:
        infos, actions = run_episode(env, make_controller(name, env), seed=0, profile_index=0)
        energies[name] = episode_metrics(env, infos, actions)["pump_energy_kJ"]
    assert max(energies, key=energies.get) == "constant_max", energies


# -------------------------------------------------------------------- thermostat
def test_thermostat_hysteresis_prevents_chattering(make_env):
    """A measurement jittering inside the band must not flip the relay.

    Driven with a temperature that oscillates across the switch-on point by
    less than the band, the thermostat should switch on once and stay on. The
    same sequence with a zero-width band chatters, which is what the band is
    there to prevent.
    """
    env = make_env()
    spec = ControllerSpec.from_env(env)
    params = ThermostatParams(on_margin_K=3.0, band_K=1.0)
    T_on = env.T_max - params.on_margin_K
    # Starts below the band (so the relay switches off), crosses the switch-on
    # point, then jitters by +-0.4 K around it: inside the 1 K band, so no
    # switch-off is allowed.
    temps = [T_on - 2.0] * 3 + list(T_on + 0.4 * np.array([1, -1] * 20))

    def actions_for(band):
        controller = ThermostatController(spec, ThermostatParams(params.on_margin_K, band))
        return [float(controller(synthetic_obs(env, T))[0]) for T in temps]

    with_band = actions_for(params.band_K)
    without_band = actions_for(0.0)
    switches = int(np.count_nonzero(np.diff(with_band)))
    assert switches == 1, with_band
    assert int(np.count_nonzero(np.diff(without_band))) > 10, "zero band should chatter"
    assert with_band[0] == params.action_low and with_band[-1] == params.action_high


def test_thermostat_is_bang_bang_and_starts_safe(make_env):
    env = make_env()
    spec = ControllerSpec.from_env(env)
    params = ThermostatParams(on_margin_K=3.0, band_K=1.0)
    controller = ThermostatController(spec, params)
    assert controller.on is True  # the relay starts on the safe side of the band
    inside_band = controller.T_on - 0.5 * params.band_K
    assert float(controller(synthetic_obs(env, inside_band))[0]) == 1.0  # holds, does not drop
    hot = float(controller(synthetic_obs(env, env.T_max + 5.0))[0])
    cold = float(controller(synthetic_obs(env, 10.0))[0])
    assert (hot, cold) == (1.0, 0.0)  # only the two extreme actions ever appear


@pytest.mark.parametrize("mode", ["full", "partial"])
def test_thermostat_reacts_to_the_hottest_measured_cell(make_env, mode):
    """Only the maximum over the measured cells may matter, not their order."""
    env = make_env(obs_mode=mode)
    spec = ControllerSpec.from_env(env)
    controller = ThermostatController(spec, ThermostatParams(on_margin_K=3.0, band_K=1.0))
    cool = np.full(spec.n_temps, 20.0)
    one_hot = cool.copy()
    one_hot[-1] = env.T_max  # a single hot cell is enough to call for flow
    assert float(controller(synthetic_obs(env, cool))[0]) == 0.0
    assert float(controller(synthetic_obs(env, one_hot))[0]) == 1.0


# --------------------------------------------------------------------------- PID
@pytest.mark.parametrize("mode", ["full", "partial"])
def test_pid_increases_flow_as_temperature_rises(make_env, mode):
    env = make_env(obs_mode=mode)
    spec = ControllerSpec.from_env(env)
    # P and D only here: the integral term is exercised in its own tests, and a
    # ramp makes the P + D response monotone in the temperature by construction.
    controller = PIDController(spec, PIDParams(setpoint_margin_K=5.0, Kp=0.1, Ki=0.0, Kd=1.0))
    temps = np.linspace(controller.setpoint - 10.0, controller.setpoint + 10.0, 40)
    actions = [float(controller(synthetic_obs(env, T))[0]) for T in temps]
    assert np.all(np.diff(actions) >= -1e-9), actions  # monotone in the temperature
    assert actions[0] == 0.0 and actions[-1] == 1.0
    assert actions[int(np.argmin(np.abs(temps - controller.setpoint)))] > 0.0


def test_pid_integral_increases_flow_while_the_pack_stays_too_hot(make_env):
    """A persistent positive error must keep pushing the flow up, not settle short."""
    env = make_env()
    spec = ControllerSpec.from_env(env)
    controller = PIDController(spec, PIDParams(setpoint_margin_K=5.0, Kp=0.05, Ki=0.01, Kd=0.0))
    held = controller.setpoint + 1.0
    actions = [float(controller(synthetic_obs(env, held))[0]) for _ in range(6)]
    assert np.all(np.diff(actions) > 0), actions
    assert controller.integral > 0.0


def test_pid_is_zero_below_the_setpoint_and_saturates_above(make_env):
    env = make_env()
    spec = ControllerSpec.from_env(env)
    controller = PIDController(spec, PIDParams(setpoint_margin_K=5.0, Kp=0.2, Ki=0.0, Kd=0.0))
    assert float(controller(synthetic_obs(env, controller.setpoint - 1.0))[0]) == 0.0
    assert float(controller(synthetic_obs(env, controller.setpoint + 100.0))[0]) == 1.0


def test_pid_integral_does_not_wind_up_while_saturated(make_env):
    """After a long cold stretch at zero flow, a step above the setpoint must act at once."""
    env = make_env()
    spec = ControllerSpec.from_env(env)
    controller = PIDController(spec, PIDParams(setpoint_margin_K=5.0, Kp=0.1, Ki=0.01, Kd=0.0))
    for _ in range(200):  # 2000 s pinned at minimum flow, large negative error
        controller(synthetic_obs(env, controller.setpoint - 15.0))
    assert controller.integral <= 0.0
    immediate = float(controller(synthetic_obs(env, controller.setpoint + 5.0))[0])
    assert immediate == pytest.approx(0.1 * 5.0, abs=1e-6)


def test_pid_derivative_anticipates_a_rising_temperature(make_env):
    """With Kd > 0 a rising temperature gets more flow than the same temperature held."""
    env = make_env()
    spec = ControllerSpec.from_env(env)
    params = PIDParams(setpoint_margin_K=5.0, Kp=0.05, Ki=0.0, Kd=2.0)
    T = env.T_max - 5.0

    rising = PIDController(spec, params)
    rising.reset()
    rising(synthetic_obs(env, T - 2.0))
    rising_action = float(rising(synthetic_obs(env, T))[0])

    steady = PIDController(spec, params)
    steady.reset()
    steady(synthetic_obs(env, T))
    steady_action = float(steady(synthetic_obs(env, T))[0])
    assert rising_action > steady_action


# ---------------------------------------------------------------- tuning provenance
@pytest.mark.skipif(not TUNING_CSV.exists(), reason="run scripts/tune_baselines.py first")
def test_tuned_defaults_match_logged_sweep():
    """The defaults in baselines.TUNED must be what the logged sweep selected."""
    selected = select_best_from_csv(TUNING_CSV)
    assert selected, "the sweep log has no tunable controller in it"
    for key, params in sorted(selected.items()):
        assert baselines.TUNED[key] == params, f"{key}: code has {baselines.TUNED[key]!r}"


@pytest.mark.skipif(not TUNING_CSV.exists(), reason="run scripts/tune_baselines.py first")
def test_tuning_used_training_profiles_only():
    """A sweep that peeked at the held-out split would invalidate every M2 number."""
    import pandas as pd

    from kelvin_rl.control.tuning import TUNING_PROFILES, TUNING_SPLIT

    df = pd.read_csv(TUNING_CSV)
    assert TUNING_SPLIT == "train"
    assert set(df["split"]) == {"train"}
    assert set(df["profile"]) <= set(TUNING_PROFILES)


@pytest.mark.skipif(not EVALUATION_CSV.exists(), reason="run scripts/evaluate_baselines.py first")
def test_evaluation_used_held_out_profiles_only(make_env):
    """The evaluation log must carry its own evidence that it is held out."""
    import pandas as pd

    env = make_env(split="test")
    df = pd.read_csv(EVALUATION_CSV)
    assert set(df["split"]) == {"test"}
    assert set(df["profile"]) == set(range(env.profiles.count("test")))
    assert df["seed"].nunique() >= 5  # the README asks for at least five seeds
    assert set(df["controller"]) == set(BASELINES)


@pytest.mark.skipif(not TUNING_CSV.exists(), reason="run scripts/tune_baselines.py first")
@pytest.mark.parametrize("mode", ["full", "partial"])
def test_margin_rule_is_satisfied_by_what_it_selected(mode):
    """thermostat_margin's defining constraint, checked against the logged sweep.

    The limit is derived from the configured sensor noise, so this also pins the
    derivation rather than a number.
    """
    import pandas as pd

    from kelvin_rl.config import load_config
    from kelvin_rl.control.tuning import MARGIN_SIGMAS

    cfg = load_config()
    limit = margin_limit_C(cfg)
    assert limit == cfg["safety"]["T_max"] - MARGIN_SIGMAS * cfg["sensors"]["sigma_cell"]

    params = baselines.TUNED[("thermostat_margin", mode)]
    df = pd.read_csv(TUNING_CSV)
    rows = df[
        (df["controller"] == "thermostat")
        & (df["obs_mode"] == mode)
        & (df["on_margin_K"] == params.on_margin_K)
        & (df["band_K"] == params.band_K)
    ]
    assert len(rows) == len(set(rows["profile"])) > 0
    assert rows["peak_T_C"].max() <= limit
    assert rows["degree_s_above"].sum() == 0.0


def test_both_thermostat_variants_exist_and_are_the_same_controller(make_env):
    """One controller, two selection rules: the failure and the fix stay comparable."""
    env = make_env()
    plain = make_controller("thermostat", env)
    margin = make_controller("thermostat_margin", env)
    assert type(plain) is type(margin) is ThermostatController
    assert plain.params != margin.params

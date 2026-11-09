"""Environment behaviour: Gymnasium API, observation modes, actuator limits, safety of max flow."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from kelvin_rl.envs import BatteryPackCoolingEnv

from conftest import run_constant


@pytest.mark.parametrize("mode", ["full", "partial", "partial_history"])
def test_gymnasium_env_checker(mode):
    check_env(BatteryPackCoolingEnv(obs_mode=mode), skip_render_check=True)


def test_estimated_mode_not_available_yet():
    with pytest.raises(NotImplementedError):
        BatteryPackCoolingEnv(obs_mode="estimated")


def test_observation_sizes(cfg):
    n, k, h = cfg["pack"]["n_cells"], len(cfg["sensors"]["cell_indices"]), cfg["sensors"]["history_len"]
    assert BatteryPackCoolingEnv(obs_mode="full").observation_space.shape == (n + 5,)
    assert BatteryPackCoolingEnv(obs_mode="partial").observation_space.shape == (k + 5,)
    assert BatteryPackCoolingEnv(obs_mode="partial_history").observation_space.shape == ((k + 5) * h,)


def test_partial_observation_is_noisy_but_unbiased():
    env = BatteryPackCoolingEnv(obs_mode="partial")
    errors = []
    for seed in range(300):
        obs, info = env.reset(seed=seed)
        true = (info["T_cells"][env.sensor_idx] - env.T_ref) / env.T_scale
        errors.append(obs[: len(env.sensor_idx)] - true)
    errors = np.array(errors) * env.T_scale
    assert abs(errors.mean()) < 0.1
    assert errors.std() == pytest.approx(env.sigma_cell, rel=0.15)


def test_rate_limit_is_never_exceeded():
    env = BatteryPackCoolingEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    prev = env.mdot
    any_limited = False
    for _ in range(200):
        _, _, _, truncated, info = env.step(rng.choice([0.0, 1.0], size=1).astype(np.float32))
        assert abs(info["mdot"] - prev) <= env.delta_mdot_max + 1e-12
        any_limited |= info["rate_limited"]
        prev = info["mdot"]
        if truncated:
            break
    assert any_limited, "bang-bang actions should have triggered the rate limit"


def test_actions_outside_box_are_clipped():
    env = BatteryPackCoolingEnv()
    env.reset(seed=0, options={"action_init": 1.0})
    _, _, _, _, info = env.step(np.array([5.0], dtype=np.float32))
    assert info["mdot"] == pytest.approx(env.mdot_max)


def test_reward_is_normalised_pump_energy():
    env = BatteryPackCoolingEnv()
    env.reset(seed=0, options={"action_init": 1.0})
    _, reward, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert reward == pytest.approx(-1.0)
    assert info["pump_energy_J"] == pytest.approx(env.params.P_pump_max * env.dt_ctrl)


def test_episode_length_and_truncation():
    env = BatteryPackCoolingEnv()
    infos = run_constant(env, 0.5, seed=0)
    assert len(infos) == env.n_ctrl_steps == 360


@pytest.mark.parametrize("scale", [0.0, 1.0])
@pytest.mark.parametrize("split", ["train", "test"])
def test_constant_max_flow_never_violates(split, scale):
    """If max flow cannot keep the pack safe, the parameters are unrealistic."""
    env = BatteryPackCoolingEnv(split=split, overrides={"mismatch.scale": scale})
    n = env.profiles.count(split) if split == "test" else 40
    for idx in range(n):
        infos = run_constant(env, 1.0, seed=idx, options={"profile_index": idx, "action_init": 1.0})
        peak = max(i["T_peak_substep"] for i in infos)
        assert peak < env.T_max, f"{split} profile {idx}: peak {peak:.1f} C at max flow"


def test_minimum_flow_does_violate_somewhere():
    """Otherwise the control problem is trivial: min flow would be optimal."""
    env = BatteryPackCoolingEnv(split="train")
    violated = 0
    for idx in range(20):
        infos = run_constant(env, 0.0, seed=idx, options={"profile_index": idx, "action_init": 0.0})
        violated += any(i["violation"] for i in infos)
    assert violated > 0

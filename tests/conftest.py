import numpy as np
import pytest

from kelvin_rl.config import load_config
from kelvin_rl.envs import BatteryPackCoolingEnv


@pytest.fixture
def cfg():
    return load_config()


def run_constant(env, action, **reset_kwargs):
    """Roll out one episode at a constant action. Returns the list of step infos."""
    env.reset(**reset_kwargs)
    infos = []
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        infos.append(info)
        done = terminated or truncated
    return infos


@pytest.fixture
def make_env():
    def _make(**kwargs):
        return BatteryPackCoolingEnv(**kwargs)

    return _make

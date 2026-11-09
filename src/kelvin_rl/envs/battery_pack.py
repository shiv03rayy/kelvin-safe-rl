"""``BatteryPackCoolingEnv``: choose coolant flow to keep every cell below T_max.

Reward and cost are returned separately so that constrained RL can treat them
separately: ``reward`` is normalised pumping energy (always <= 0) and
``info["cost"]`` is the sum of degrees above T_max.

Design choices worth knowing:

* **The true plant differs from the nominal model.** The env runs on
  :meth:`PackParams.perturbed`, while ``env.nominal_params`` is what an
  estimator or model-based controller is allowed to know.
* **Observation modes differ only in temperature information.** Every mode
  includes coolant inlet/outlet temperature, current, previous action and
  normalised time, so comparing modes isolates the effect of observability.
* **The actuator is rate-limited by the env**, not the agent. Any clipping is
  reported in ``info``.
* **Episodes never terminate on a violation.** They are truncated when the
  load profile ends. Violations are priced through the cost signal instead.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np

from kelvin_rl.config import DEFAULT_ENV_CONFIG, load_config
from kelvin_rl.envs import physics
from kelvin_rl.envs.load_profiles import LoadProfiles

OBS_MODES = ("full", "partial", "partial_history", "estimated")


class BatteryPackCoolingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict | str | Path | None = None,
        obs_mode: str = "full",
        split: str = "train",
        overrides: dict | None = None,
    ):
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got '{obs_mode}'")
        if obs_mode == "estimated":
            raise NotImplementedError("The 'estimated' mode arrives with the Kalman filter (M4).")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        if isinstance(config, dict):
            cfg = config
        else:
            cfg = load_config(config or DEFAULT_ENV_CONFIG, overrides)
        self.cfg = cfg
        self.obs_mode = obs_mode
        self.split = split

        self.nominal_params = physics.PackParams.from_config(cfg)
        self.params = self.nominal_params.perturbed(cfg["mismatch"])
        self.n = self.nominal_params.n_cells

        c, t = cfg["coolant"], cfg["time"]
        self.mdot_min, self.mdot_max = float(c["mdot_min"]), float(c["mdot_max"])
        self.delta_mdot_max = float(c["rate_limit_frac"]) * (self.mdot_max - self.mdot_min)
        self.dt_phys, self.dt_ctrl = float(t["dt_phys"]), float(t["dt_ctrl"])
        self.substeps = int(round(self.dt_ctrl / self.dt_phys))
        if not np.isclose(self.substeps * self.dt_phys, self.dt_ctrl):
            raise ValueError("dt_ctrl must be a whole multiple of dt_phys")
        # Fail loudly at construction rather than produce a silently wrong simulation.
        self.euler_spectral_radius = physics.check_euler_stability(
            self.params, self.mdot_min, self.mdot_max, self.dt_phys
        )

        self.T_max = float(cfg["safety"]["T_max"])
        self.dT_max = float(cfg["safety"]["dT_max"])
        self.E_norm = self.nominal_params.P_pump_max * self.dt_ctrl  # so reward is in [-1, 0]

        s, o = cfg["sensors"], cfg["observation"]
        self.sensor_idx = np.asarray(s["cell_indices"], dtype=int)
        if self.sensor_idx.min() < 0 or self.sensor_idx.max() >= self.n:
            raise ValueError("sensor cell_indices out of range")
        self.sigma_cell, self.sigma_coolant = float(s["sigma_cell"]), float(s["sigma_coolant"])
        self.history_len = int(s["history_len"])
        self.T_ref, self.T_scale, self.I_scale = (
            float(o["T_ref"]),
            float(o["T_scale"]),
            float(o["I_scale"]),
        )

        self.profiles = LoadProfiles(cfg)
        self.n_ctrl_steps = self.profiles.n_samples // self.substeps

        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        base_dim = (self.n if obs_mode == "full" else len(self.sensor_idx)) + 5
        obs_dim = base_dim * (self.history_len if obs_mode == "partial_history" else 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def action_to_mdot(self, a: float) -> float:
        return self.mdot_min + float(np.clip(a, 0.0, 1.0)) * (self.mdot_max - self.mdot_min)

    def mdot_to_action(self, mdot: float) -> float:
        return (mdot - self.mdot_min) / (self.mdot_max - self.mdot_min)

    def _scale_T(self, T):
        return (np.asarray(T, dtype=float) - self.T_ref) / self.T_scale

    def _base_obs(self) -> np.ndarray:
        T_c = physics.coolant_temperatures(self.params, self.T, self.mdot)
        T_in, T_out = self.params.T_in, T_c[-1]
        if self.obs_mode == "full":
            temps = self.T
        else:
            rng = self.np_random
            temps = self.T[self.sensor_idx] + self.sigma_cell * rng.standard_normal(len(self.sensor_idx))
            T_in = T_in + self.sigma_coolant * rng.standard_normal()
            T_out = T_out + self.sigma_coolant * rng.standard_normal()
        I = self.current[min(self.k, len(self.current) - 1)]
        return np.concatenate(
            [
                self._scale_T(temps),
                self._scale_T([T_in, T_out]),
                [I / self.I_scale, self.mdot_to_action(self.mdot), self.step_idx / self.n_ctrl_steps],
            ]
        ).astype(np.float32)

    def _obs(self) -> np.ndarray:
        base = self._base_obs()
        if self.obs_mode != "partial_history":
            return base
        self.history.append(base)
        while len(self.history) < self.history_len:  # pad at the start of an episode
            self.history.appendleft(base)
        return np.concatenate(self.history).astype(np.float32)

    # --------------------------------------------------------------- Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "profile" in options:  # a raw current array, used by tests and analysis
            self.current = np.asarray(options["profile"], dtype=float)
            self.profile_index = None
        else:
            idx = options.get("profile_index")
            if idx is None:
                idx = int(self.np_random.integers(self.profiles.count(self.split)))
            self.current = self.profiles.get(self.split, idx).current
            self.profile_index = idx
        self.n_ctrl_steps = len(self.current) // self.substeps

        init = self.cfg["init"]
        T0 = options.get("T_init")
        if T0 is None:
            T0 = init["T_init"] + self.np_random.uniform(0.0, init["T_init_jitter"], self.n)
        self.T = np.broadcast_to(np.asarray(T0, dtype=float), (self.n,)).copy()
        self.mdot = self.action_to_mdot(options.get("action_init", init["action_init"]))
        self.k = 0
        self.step_idx = 0
        self.history = deque(maxlen=self.history_len)
        return self._obs(), self._info_reset()

    def step(self, action):
        a = float(np.asarray(action, dtype=float).reshape(-1)[0])
        mdot_cmd = self.action_to_mdot(a)
        mdot = float(np.clip(mdot_cmd, self.mdot - self.delta_mdot_max, self.mdot + self.delta_mdot_max))
        self.mdot = mdot

        energy = {"gen": 0.0, "cool": 0.0, "amb": 0.0, "cond": 0.0}
        T_start = self.T.copy()
        T_peak = float(self.T.max())
        for _ in range(self.substeps):
            I = self.current[self.k]
            self.T, e = physics.euler_step(self.params, self.T, mdot, I, self.dt_phys)
            for key in energy:
                energy[key] += e[key]
            T_peak = max(T_peak, float(self.T.max()))
            self.k += 1
        self.step_idx += 1

        pump_J = physics.pump_power(self.params, mdot) * self.dt_ctrl
        reward = -pump_J / self.E_norm
        excess = np.maximum(0.0, self.T - self.T_max)
        spread = float(self.T.max() - self.T.min())
        truncated = self.step_idx >= self.n_ctrl_steps
        info = {
            "cost": float(excess.sum()),
            "violation": bool(excess.any()),
            "T_cells": self.T.copy(),
            "T_coolant": physics.coolant_temperatures(self.params, self.T, mdot),
            "T_peak_substep": T_peak,
            "spread": spread,
            "spread_violation": spread > self.dT_max,
            "mdot": mdot,
            "mdot_cmd": mdot_cmd,
            "rate_limited": not np.isclose(mdot, mdot_cmd),
            "pump_energy_J": pump_J,
            "energy_gen_J": energy["gen"],
            "energy_cool_J": energy["cool"],
            "energy_amb_J": energy["amb"],
            "energy_cond_J": energy["cond"],
            "stored_change_J": float(np.sum(self.params.C_cell * (self.T - T_start))),
            "profile_index": self.profile_index,
        }
        return self._obs(), float(reward), False, truncated, info

    def _info_reset(self) -> dict:
        return {
            "T_cells": self.T.copy(),
            "mdot": self.mdot,
            "profile_index": self.profile_index,
        }

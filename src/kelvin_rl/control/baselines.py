"""Classical baseline controllers: constant maximum flow, thermostat(s), PID.

These are the reference points the RL agents have to beat (M3). Each one is a
small stateful callable with the same interface, so the evaluation script can
treat them interchangeably::

    controller = make_controller("pid", env)
    controller.reset()
    action = controller(obs)          # float32 array in the env's Box([0, 1])

What a controller may look at
-----------------------------
Only the observation vector, plus the *static* description of that vector held
in :class:`ControllerSpec`: how many temperature channels it carries, the
``(T - T_ref) / T_scale`` scaling the env applies, the safety limit being aimed
at, and the control interval. Nothing reaches into ``env.T``, the true plant
parameters or the load profile. Under the ``partial`` modes the controller
therefore sees only the (noisy) sensed cells, which is the point: the hottest
cell of the pack is downstream of both sensors and is never measured directly.

Two environment details shape the designs:

* the env rate-limits the actuator itself (``rate_limit_frac`` of the flow
  range per control step) and reports any clipping in ``info["rate_limited"]``,
  so a controller may command a step change and let the env ramp it;
* a control decision is taken every ``dt_ctrl`` (10 s) while the physics runs
  at ``dt_phys`` (1 s), so the PID's integral and derivative use ``dt_ctrl``.

Tuning procedure
----------------
Reproduce with ``python scripts/tune_baselines.py`` (grids and selection rule
live in :mod:`kelvin_rl.control.tuning`):

1. Free parameters are swept over the fixed grids in that module. Constant
   maximum flow has none; the thermostat has a switch-on margin below ``T_max``
   and a hysteresis band; the PID has a setpoint margin below ``T_max`` and the
   three gains.
2. Each candidate is rolled out on the
   :data:`~kelvin_rl.control.tuning.TUNING_PROFILES` **training** profiles
   (episode seed = profile index). The held-out test split is never read by the
   sweep.
3. Every episode's metrics are logged to ``results/m2_tuning_sweep.csv``.
4. The winner is chosen from that file by a lexicographic rule
   (:func:`~kelvin_rl.control.tuning.select_best`): least total degree-seconds
   above ``T_max`` summed over the tuning profiles, then least mean pump
   energy, then the parameter values themselves for determinism. Safety first,
   energy second; if no candidate is safe on every tuning profile the rule
   still returns the least-unsafe one rather than silently hiding the failure.
   That is the whole rule for every controller except ``thermostat_margin``,
   which adds one constraint - see the section below, and read that section
   before quoting any of this as pre-registered.
5. The winner per (controller, obs_mode) is the default in :data:`TUNED`.
   ``tests/test_baselines.py::test_tuned_defaults_match_logged_sweep`` re-runs
   step 4 against the committed CSV, so a default that did not come out of the
   logged sweep fails the test suite.

Each observation mode is tuned separately, because "the hottest measured
temperature" means something different when it is one of eight true cell
temperatures and when it is the warmer of two noisy upstream sensors.
``partial_history`` reuses the ``partial`` tuning: the sensors are the same and
these controllers only ever use the newest frame.

The two thermostats, and where the second rule came from
--------------------------------------------------------
``thermostat`` and ``thermostat_margin`` are the *same* controller on the same
grid, selected by two different rules from the same logged sweep.

``thermostat`` uses the rule above, which was fixed before any evaluation. On
the held-out profiles that thermostat exceeded ``T_max`` in some episodes, which
``results/m2_baselines_test.csv`` records. The cause is visible in the tuning
log: because the rule asks only for zero violations *on the tuning profiles* and
otherwise minimises energy, it selects the most aggressive candidate that just
survives training, leaving no margin for a profile it has not seen.

``thermostat_margin`` was added **after** that was observed, so - unlike every
other number in this module - its selection rule is **not pre-registered**, and
it should be read as a post-hoc design fix rather than as a prediction that was
confirmed. What it adds is one feasibility constraint, documented in
:mod:`kelvin_rl.control.tuning`: the worst peak over the tuning profiles must
also sit at or below ``T_max - 2 * sigma_cell``, computed from the configured
sensor noise instead of being written down as a number. It still never looks at
the test split, and it is still the same lexicographic rule inside the feasible
set, so nothing is hand-picked; but it was motivated by a held-out failure, and
that is a weaker claim than the rest of this module makes. Both variants are
kept and both are evaluated, so the failure stays on the record next to the fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINES = ("constant_max", "thermostat", "thermostat_margin", "pid")

# Modes that share a tuning: partial_history sees the same sensors as partial.
TUNING_OBS_MODE = {"full": "full", "partial": "partial", "partial_history": "partial"}


@dataclass(frozen=True)
class ControllerSpec:
    """Static, non-privileged facts a baseline needs about its environment.

    ``n_temps`` temperature channels come first in each observation frame,
    followed by coolant inlet and outlet temperature, current, previous action
    and normalised time. Under ``partial_history`` the observation is several
    such frames concatenated with the newest last, so decoding always works on
    the final frame.
    """

    n_temps: int
    T_ref: float
    T_scale: float
    T_max: float
    dt_ctrl: float

    @classmethod
    def from_env(cls, env) -> "ControllerSpec":
        n_temps = env.n if env.obs_mode == "full" else len(env.sensor_idx)
        return cls(
            n_temps=int(n_temps),
            T_ref=float(env.T_ref),
            T_scale=float(env.T_scale),
            T_max=float(env.T_max),
            dt_ctrl=float(env.dt_ctrl),
        )

    @property
    def frame_dim(self) -> int:
        return self.n_temps + 5  # temps, T_in, T_out, current, prev action, time

    def latest_frame(self, obs) -> np.ndarray:
        return np.asarray(obs, dtype=float).reshape(-1)[-self.frame_dim :]

    def measured_temperatures_C(self, obs) -> np.ndarray:
        """Undo the env's observation scaling, to get degrees Celsius back."""
        return self.latest_frame(obs)[: self.n_temps] * self.T_scale + self.T_ref

    def hottest_measured_C(self, obs) -> float:
        return float(np.max(self.measured_temperatures_C(obs)))

    def previous_action(self, obs) -> float:
        return float(self.latest_frame(obs)[self.n_temps + 3])


def _action(value: float) -> np.ndarray:
    return np.array([np.clip(value, 0.0, 1.0)], dtype=np.float32)


class BaselineController:
    """Common interface: ``reset()`` once per episode, then ``act(obs)`` per step."""

    name = "baseline"

    def __init__(self, spec: ControllerSpec):
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        """Clear any internal state. Called before every episode."""

    def act(self, obs) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, obs) -> np.ndarray:
        return self.act(obs)


class ConstantFlowController(BaselineController):
    """Hold one flow setting for the whole episode.

    With ``action = 1.0`` this is the "constant maximum flow" baseline: it
    ignores the observation entirely, is as safe as this actuator can be, and
    pays the highest pump energy the actuator can draw, which fixes the energy
    ceiling the other controllers are measured against.
    """

    name = "constant_max"

    def __init__(self, spec: ControllerSpec, action: float = 1.0):
        self.action = float(action)
        super().__init__(spec)

    def act(self, obs) -> np.ndarray:
        return _action(self.action)


@dataclass(frozen=True)
class ThermostatParams:
    """Switch-on point and hysteresis band, both measured down from ``T_max``."""

    on_margin_K: float  # switch to full flow at T_max - on_margin_K
    band_K: float  # switch back to minimum flow band_K below that point
    action_high: float = 1.0
    action_low: float = 0.0


class ThermostatController(BaselineController):
    """Bang-bang control on the hottest measured temperature, with hysteresis.

    Full flow above ``T_on = T_max - on_margin_K``, minimum flow below
    ``T_off = T_on - band_K``, and *hold the current state* in between. The band
    is what stops the output chattering: a measurement hovering at the
    threshold, or sensor noise straddling it, cannot flip the relay back and
    forth, because flipping requires crossing the whole band. The episode
    starts in the "on" state, which is the safe side of the relay.
    """

    name = "thermostat"

    def __init__(self, spec: ControllerSpec, params: ThermostatParams):
        self.params = params
        self.T_on = spec.T_max - params.on_margin_K
        self.T_off = self.T_on - params.band_K
        super().__init__(spec)

    def reset(self) -> None:
        self.on = True

    def act(self, obs) -> np.ndarray:
        T = self.spec.hottest_measured_C(obs)
        if T >= self.T_on:
            self.on = True
        elif T <= self.T_off:
            self.on = False
        return _action(self.params.action_high if self.on else self.params.action_low)


@dataclass(frozen=True)
class PIDParams:
    """Setpoint margin below ``T_max`` and the three gains.

    With the error in kelvin and time in seconds, and an output in normalised
    action units, ``Kp`` is 1/K, ``Ki`` is 1/(K s) and ``Kd`` is s/K.
    """

    setpoint_margin_K: float
    Kp: float
    Ki: float
    Kd: float


class PIDController(BaselineController):
    """PID on the hottest measured temperature, driving it to the setpoint.

    The error is ``e = T_hot - (T_max - setpoint_margin_K)``, positive when the
    pack is too hot, so all three gains are non-negative and a positive error
    asks for more flow. The setpoint sits below ``T_max`` on purpose: the
    controller only ever sees the cells it measures, and under the partial
    modes the hottest cell is not one of them, so the margin buys back the
    unmeasured gradient along the channel.

    Two practical details:

    * **Derivative on the measurement**, over one control interval. There is no
      setpoint change to differentiate, so this costs nothing and avoids a
      derivative kick.
    * **Conditional-integration anti-windup.** The integral only accumulates
      when the unsaturated output is inside ``[0, 1]``, or when the error points
      back out of saturation. Saturation matters here: on light load the
      actuator sits pinned at minimum flow with a large negative error, and an
      integral winding down over those stretches would delay the response to
      the next current burst.
    """

    name = "pid"

    def __init__(self, spec: ControllerSpec, params: PIDParams):
        self.params = params
        self.setpoint = spec.T_max - params.setpoint_margin_K
        super().__init__(spec)

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_T: float | None = None

    def act(self, obs) -> np.ndarray:
        p, dt = self.params, self.spec.dt_ctrl
        T = self.spec.hottest_measured_C(obs)
        error = T - self.setpoint
        derivative = 0.0 if self.prev_T is None else (T - self.prev_T) / dt
        self.prev_T = T

        raw = p.Kp * error + p.Ki * self.integral + p.Kd * derivative
        clipped = float(np.clip(raw, 0.0, 1.0))
        unsaturated = raw == clipped
        pushing_back = (raw > 1.0 and error < 0.0) or (raw < 0.0 and error > 0.0)
        if unsaturated or pushing_back:
            self.integral += error * dt
        return _action(clipped)


# Tuned defaults. Every number below was selected by scripts/tune_baselines.py
# from results/m2_tuning_sweep.csv under the rules documented in this module's
# docstring - including, for thermostat_margin, the post-hoc one; none of them
# is hand-picked and none should be edited by hand.
# tests/test_baselines.py::test_tuned_defaults_match_logged_sweep re-derives
# them from the committed CSV and fails on any drift.
TUNED: dict[tuple[str, str], ThermostatParams | PIDParams] = {
    ("pid", "full"): PIDParams(setpoint_margin_K=3.0, Kp=0.2, Ki=0.0, Kd=4.0),
    ("pid", "partial"): PIDParams(setpoint_margin_K=5.0, Kp=0.2, Ki=0.0, Kd=0.0),
    ("thermostat", "full"): ThermostatParams(on_margin_K=1.0, band_K=0.5),
    ("thermostat", "partial"): ThermostatParams(on_margin_K=2.0, band_K=0.5),
    ("thermostat_margin", "full"): ThermostatParams(on_margin_K=2.0, band_K=0.5),
    ("thermostat_margin", "partial"): ThermostatParams(on_margin_K=4.0, band_K=0.5),
}

_CLASSES = {
    "thermostat": ThermostatController,
    "thermostat_margin": ThermostatController,  # same controller, stricter selection rule
    "pid": PIDController,
}


def tuned_params(name: str, obs_mode: str) -> ThermostatParams | PIDParams:
    """The tuned parameters for one controller under one observation mode."""
    if obs_mode not in TUNING_OBS_MODE:
        raise ValueError(f"unknown obs_mode '{obs_mode}'")
    return TUNED[(name, TUNING_OBS_MODE[obs_mode])]


def make_controller(name: str, env, params=None) -> BaselineController:
    """Build one baseline for ``env``, using the tuned defaults unless overridden."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline '{name}', expected one of {BASELINES}")
    spec = ControllerSpec.from_env(env)
    if name == "constant_max":
        return ConstantFlowController(spec)
    if params is None:
        params = tuned_params(name, env.obs_mode)
    return _CLASSES[name](spec, params)

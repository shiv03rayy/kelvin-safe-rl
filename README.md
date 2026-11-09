# Kelvin

**Safe reinforcement learning with Kalman state estimation for battery thermal management**

> **Status: early development.** The simulated environment and its physics tests exist; the controllers, the Kalman filter and the RL agents do not yet. This README describes the design and the planned experiments. It has no results section yet: results will only appear here once a script has generated them from logged data in `results/`. See the [roadmap](#roadmap) for progress.

Kelvin is an open-source research prototype for **safety-constrained control under partial observability**. A reinforcement learning agent controls the coolant flow through a simulated battery pack. It has to keep every cell below a temperature limit while using as little pumping energy as possible. Only a few cells have sensors, and those sensors are noisy, so a Kalman filter built from the pack's physics model estimates the hidden cell temperatures, and the controller acts on those estimates.

The project compares a standard RL agent (PPO with a fixed cost penalty) against a constrained one (PPO-Lagrangian) and against classical controllers, with every configuration run over multiple random seeds.

---

## Contents

- [Why this problem](#why-this-problem)
- [Environment](#environment-batterypackcoolingenv)
- [State estimation](#state-estimation)
- [Controllers](#controllers)
- [Experiments](#experiments)
- [Design principles](#design-principles)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [References](#references)

---

## Why this problem

Thermal management combines three things that make control hard:

1. **Hard safety constraints.** A single cell above its temperature limit counts as a failure, however efficient the controller is on average.
2. **Partial observability.** Real packs rarely have a sensor on every cell, and the hottest cell is often one you cannot see directly.
3. **A real cost of acting.** Pump power grows roughly with the cube of flow rate, so "always run at full flow" is safe but expensive.

A fixed penalty on constraint violations forces you to hand-tune a trade-off between safety and energy. Constrained RL (PPO-Lagrangian) learns that trade-off instead, by adjusting a Lagrange multiplier until the average cost meets a target. Kelvin tests whether that holds up under fair conditions: held-out load profiles, several seeds, model mismatch between the estimator and the true system, and classical baselines for comparison.

---

## Environment: `BatteryPackCoolingEnv`

A [Gymnasium](https://gymnasium.farama.org/) environment built on a **lumped-parameter thermal network**, written in vectorised NumPy so it trains quickly on a laptop CPU.

### Physics

- **N cells** in a row (default `N = 8`) along a single coolant channel. Coolant enters at cell 1 and leaves after cell N, so downstream cells see warmer coolant. This creates a temperature gradient along the pack.
- **Cell temperature dynamics**, one equation per cell (edge cells have one neighbour):

```math
C_\text{cell}\,\frac{dT_i}{dt} = Q_{\text{gen},i}(t) - h_\text{eff}(\dot m)\,A\,(T_i - T_{c,i}) - k_\text{cond}\,(2T_i - T_{i-1} - T_{i+1}) - h_\text{amb}\,(T_i - T_\text{amb})
```

- **Heat generation** from Joule heating, $Q_{\text{gen},i} = I(t)^2 R_i$, with a small random cell-to-cell spread in $R_i$ to mimic manufacturing variation.
- **Coolant** follows a quasi-steady energy balance per segment, so it warms as it moves along the channel:

```math
\dot m\, c_p\,(T_{c,i} - T_{c,i-1}) = h_\text{eff}(\dot m)\,A\,(T_i - T_{c,i})
```

- **Heat-transfer scaling** $h_\text{eff}(\dot m) = h_0\,(\dot m / \dot m_\text{ref})^{0.8}$, a Dittus–Boelter-style approximation.
- **Pump power** $P_\text{pump} \propto \dot m^3$, following the pump affinity laws.
- **Integration** uses explicit Euler with physics sub-steps (for example 1 s), with a control decision every 10 s. A test checks that the sub-step is numerically stable for the chosen parameters.

All physical parameters live in `configs/env_default.yaml`. Each one is labelled with its source, or marked *illustrative* where no source applies.

### Load profiles

The current $I(t)$ comes from synthetic drive-cycle-style profiles that mix steady cruising, acceleration bursts and fast-charge segments. Some profiles are **held out for evaluation only**, and all reported results use those held-out profiles.

### Actions

- `Box([0, 1])`, mapped linearly to a flow rate $\dot m \in [\dot m_\text{min}, \dot m_\text{max}]$.
- **Actuator rate limit:** $|\dot m_t - \dot m_{t-1}| \le \Delta\dot m_\text{max}$. The environment enforces it by clipping the action, and logs every clip.

### Reward and cost

Reward and cost are kept separate so that constrained RL methods can treat them separately:

| Signal | Definition |
|---|---|
| Reward | $r_t = -P_\text{pump}\,\Delta t_\text{ctrl} / E_\text{norm}$ (normalised pumping energy) |
| Cost | $c_t = \sum_i \max(0,\ T_i - T_\text{max})$ (degrees above the limit), returned in `info`, plus a 0/1 violation flag for reporting |

The primary constraint is $\max_i T_i \le T_\text{max}$ (default 45 °C). The temperature spread across cells, $\max T - \min T \le \Delta T_\text{max}$ (default 5 °C), is logged and can optionally be used as a second constraint. Both limits are configurable. Their default values will be checked against published literature and cited in the config.

### Observation modes

| Mode | What the agent sees |
|---|---|
| `full` | All N cell temperatures, coolant outlet temperature, current, previous action, normalised time. An oracle upper bound. |
| `partial` | Noisy readings from K sensed cells (default K = 2), coolant inlet and outlet temperatures, current, previous action. |
| `partial_history` | `partial` plus a short frame stack, a common and cheap fix for partial observability. |
| `estimated` | Kalman filter mean and variance for all N cells, plus the `partial` signals. |

### Model mismatch

The Kalman filter and any model-based baseline use **nominal** parameters, while the environment runs on **perturbed** true parameters (for example $h_0$ ±20 %, $C_\text{cell}$ ±10 %, plus the resistance spread). A config value controls the size of the mismatch, so the estimator is tested for robustness instead of being given the true model.

---

## State estimation

For a known flow rate and current, the cell temperatures follow **linear time-varying** dynamics. A standard Kalman filter with time-varying matrices $A(\dot m)$ and $B$, discretised from the same physics, therefore applies, and no extended Kalman filter is needed for the core version.

- **Inputs:** previous action, known current, K noisy sensor readings.
- **Outputs:** estimated mean and variance for every cell.
- **Noise covariances** $Q$ and $R$ are tuned by a documented procedure on training profiles, never on the test set.
- **Evaluation:** RMSE per cell, maximum absolute error on the hottest cell (the one that matters for safety), and calibration, meaning whether the 95 % intervals contain the true temperature about 95 % of the time.

---

## Controllers

### Classical baselines

| Controller | Role |
|---|---|
| Constant maximum flow | Always safe, maximally wasteful. Sets the energy ceiling. |
| Thermostat | Bang-bang control with hysteresis on the hottest measured temperature. |
| PID | Acts on the hottest measured (or estimated) temperature, with a documented tuning method. |

### RL agents

The core agents are written by hand as single readable files in the style of [CleanRL](https://github.com/vwxyzjn/cleanrl), with comments that link each part of the code to the maths. Neither agent is hidden behind a library call.

| Agent | Description |
|---|---|
| **PPO with fixed penalty** (`ppo.py`) | Optimises $r_t - \lambda_\text{fixed}\, c_t$. A sweep over $\lambda_\text{fixed}$ shows the usual problem: too small breaks the limit, too large wastes energy. |
| **PPO-Lagrangian** (`ppo_lagrangian.py`) | The same PPO core with an extra cost value head and a learned multiplier $\lambda$, updated by gradient ascent on (average episode cost − cost limit). The trajectory of $\lambda$ is logged throughout training. |

As a sanity check, Stable-Baselines3 PPO is run once on the `full` observation mode to confirm the hand-written PPO performs in the same range.

---

## Experiments

Every configuration runs with **at least 5 seeds**, and all metrics are reported on **held-out** load profiles.

| ID | Question | Configurations |
|---|---|---|
| E1 | Can RL beat classical control on energy while staying safe, given full observation? | Constant, thermostat, PID, PPO-penalty (several λ), PPO-Lagrangian |
| E2 | Does constrained RL handle the safety–energy trade-off better than a hand-picked penalty? | PPO-penalty λ sweep vs PPO-Lagrangian, shown as an energy-vs-violation Pareto plot |
| E3 | What does partial observability cost, and does the Kalman filter recover it? | PPO-Lagrangian × {`full`, `partial`, `partial_history`, `estimated`} |
| E4 | How accurate is the estimator, and how robust is it to model mismatch? | Kalman filter RMSE and calibration at three mismatch levels |

**Metrics:** pump energy per episode (kJ); peak cell temperature; percentage of steps above $T_\text{max}$; degree-seconds above $T_\text{max}$; maximum temperature spread; learning curves of return and cost (mean and spread across seeds); and the λ trajectory for PPO-Lagrangian.

### Planned extension: seed-variance study

Rerun E1–E3 with **10 or more seeds**, and report interquartile mean (IQM) with bootstrap confidence intervals and probability of improvement, following Agarwal et al. (2021) and the [`rliable`](https://github.com/google-research/rliable) library. The main question is whether the ranking of methods changes depending on which seeds you look at: with 3 seeds instead of 10, how often would you reach a different conclusion about PPO-Lagrangian vs PPO-penalty? Per-seed worst-case violations are reported as well, because for a safety method the worst seed matters more than the average.

---

## Design principles

1. **No hand-typed results.** Every number in this README, every figure and every table is produced by a script from logged data in `results/`. Failures are reported, not hidden.
2. **Readable core algorithms.** PPO, PPO-Lagrangian and the Kalman filter are implemented directly, with docstrings that explain *why* each piece is there.
3. **Honest physics.** This is a simplified lumped-parameter model, not a validated battery model. It makes no claim of real-world accuracy.
4. **Reproducibility.** Seeds are fixed and logged, dependencies are pinned, and one command regenerates each figure.
5. **Multiple seeds from the start.** No single-seed results and no hand-picked best runs.
6. **CPU-friendly.** Everything should train on a laptop CPU in reasonable time.

---

## Repository layout

The planned structure (most of these files do not exist yet):

```
kelvin-safe-rl/
├── README.md
├── pyproject.toml             # pinned dependencies
├── configs/
│   ├── env_default.yaml       # physics parameters, each with a source or "illustrative" label
│   ├── obs_modes.yaml
│   └── experiments/           # e1.yaml … e4.yaml, seeds.yaml
├── src/kelvin_rl/
│   ├── envs/
│   │   ├── physics.py         # pure functions: derivatives, coolant balance, pump power
│   │   ├── battery_pack.py    # Gymnasium environment
│   │   └── load_profiles.py   # synthetic profiles and train/test split
│   ├── estimation/kalman.py
│   ├── control/baselines.py
│   ├── agents/
│   │   ├── ppo.py
│   │   └── ppo_lagrangian.py
│   └── eval/
│       ├── metrics.py
│       └── stats.py           # bootstrap CIs, IQM
├── scripts/
│   ├── train.py               # --agent --obs-mode --seed --config
│   ├── evaluate.py            # writes results/*.csv
│   ├── run_experiment.py      # loops over the seeds and configs in an experiment file
│   └── make_figures.py        # generates every figure in this README
├── tests/
├── results/                   # raw per-seed CSV logs, committed
└── figures/
```

The import name is `kelvin_rl`, because `kelvin` is already taken on PyPI by an unrelated project.

---

## Getting started

> These commands describe the intended workflow. They will work once the matching milestones are complete.

```bash
git clone https://github.com/shiv03rayy/kelvin-safe-rl.git
cd kelvin-safe-rl
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                                                   # physics and API sanity tests
python scripts/train.py --agent ppo_lagrangian --obs-mode full --seed 0 --config configs/env_default.yaml
python scripts/run_experiment.py configs/experiments/e1.yaml
python scripts/make_figures.py
```

---

## Roadmap

- [x] **M0:** Repository scaffold, pinned dependencies, config system, CI running `pytest`
- [ ] **M1 (built, parameters under review):** Physics, environment and load profiles, with sanity tests for energy balance, equilibrium, monotonicity, the spatial gradient, the Gymnasium API and the rate limit. Sanity figure: `python scripts/plot_constant_flow.py`
- [ ] **M2:** Classical baselines evaluated on held-out profiles
- [ ] **M3:** PPO-penalty and PPO-Lagrangian on `full` observations, at least 5 seeds (E1, E2)
- [ ] **M4:** Partial observability and the Kalman filter (E3, E4)
- [ ] **M5:** Results and limitations written up from generated data
- [ ] **Extension:** 10+ seed study with IQM, bootstrap CIs, probability of improvement and rank stability
- [ ] **Stretch:** Joint state and parameter estimation (augmented-state EKF) or a hybrid physics + neural-network estimator, with an ablation
- [ ] **Stretch:** Domain randomisation of physics parameters during training

---

## Limitations

These apply by design and will stay in this README once results exist:

- The thermal model is **lumped-parameter and simplified**. It captures qualitative behaviour (spatial gradients, flow-dependent heat transfer, cubic pump power) but is not validated against a real pack.
- Several parameters are **illustrative** rather than measured. The config file labels each one.
- The heat-transfer scaling is an approximation, and the coolant model assumes quasi-steady flow.
- Load profiles are synthetic. Results on synthetic profiles do not guarantee performance on real drive cycles.

---

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR.
- Ray, A., Achiam, J. & Amodei, D. (2019). *Benchmarking Safe Exploration in Deep Reinforcement Learning.* OpenAI.
- Stooke, A., Achiam, J. & Abbeel, P. (2020). *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods.* ICML.
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. & Bellemare, M. G. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS.
- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems.* Journal of Basic Engineering, 82(1), 35–45.

---

## License

[MIT](LICENSE)

## Author

Shivam Ray

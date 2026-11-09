# Kelvin Viewer — design document

**Status: design only. No application code exists yet.** This document is the spec to argue with
before anything is built. It defines what the viewer shows, where its numbers come from, how it is
prevented from inventing them, and in what order it gets built.

The viewer is an interactive 3D visualisation of the Kelvin project: a battery pack being cooled, a
controller deciding how hard to run the pump, and an estimator guessing at the temperatures nobody
measured.

---

## Contents

- [Why 3D is justified here](#why-3d-is-justified-here)
- [Three entry points](#three-entry-points)
- [Story mode: the beats](#story-mode-the-beats)
- [The layer system](#the-layer-system)
- [Encoding table](#encoding-table)
- [Data contract](#data-contract)
- [The sandbox and its parity gate](#the-sandbox-and-its-parity-gate)
- [Visual language](#visual-language)
- [Motion and choreography](#motion-and-choreography)
- [Architecture](#architecture)
- [Build order](#build-order)
- [Risks](#risks)
- [Non-goals](#non-goals)
- [Open questions](#open-questions)

---

## Why 3D is justified here

Most machine-learning projects gain nothing from 3D, because their state is an abstract vector and a
3D view is decoration. Kelvin is different in three specific ways, and the viewer exists to exploit
exactly those three.

1. **The state is spatial.** Eight cells sit in a row along one coolant channel. Coolant enters at
   cell 1 at `T_in = 30 °C` and warms as it travels, so downstream cells are cooled less well. The
   temperature gradient along the pack is a physical arrangement, and geometry carries it directly.
2. **Hidden state is the research question.** Sensors sit on cells 1 and 4 (`sensors.cell_indices:
   [0, 3]`). The hottest cell is normally cell 8, which is unsensed — the config says so explicitly.
   "What is true" versus "what the controller can see" becomes a visual contrast between two
   overlaid renderings of one object, instead of a pair of line charts.
3. **Uncertainty has an extent.** A Kalman filter reports a mean and a variance per cell. A ±2σ
   interval around each cell is a shell with a thickness, and a filter converging is those shells
   contracting onto the truth. That is the clearest available explanation of what a Kalman filter
   does, and it is only available in a spatial medium.

If a proposed feature does not lean on one of those three, it probably belongs in a static figure
produced by `scripts/make_figures.py` instead.

---

## Three entry points

One scene, one data model, three ways in. They share every component; they differ in what is on
screen and who is driving.

| Mode | For | Default state |
|---|---|---|
| **Story** | A reader arriving from the README who gives the project ninety seconds | Scroll-driven. Camera and layers choreographed. Controls hidden. |
| **Sandbox** | Anyone who wants to feel the trade-off | Free play. You hold the flow control and the pack fights you. Then you hand over to the agent. |
| **Lab** | You, inspecting a run that behaved oddly | Everything exposed. Load any trace, scrub, read per-cell numbers, compare up to four runs, pick seeds. |

Story mode is the front door and ends by offering the sandbox. Lab mode is a toggle, not a separate
page — the same pack stays on screen while the chrome around it changes.

---

## Story mode: the beats

Seven beats. Each is one camera position, one layer change, and one sentence of text. The pack is
never torn down and rebuilt; it persists and is revealed.

### 1. Heat

The pack alone, cooling disabled. A fast-charge profile drives current through it, Joule heating
climbs, and the cells walk up the colour ramp until they cross the 45 °C plane, which flashes.

*Establishes:* there is a failure state, and it arrives on its own.

### 2. The cost of cooling

The coolant channel switches on beneath the cells. Particles accelerate along it; cells cool from
cell 1 downstream in a visible wave, and the particles themselves warm as they travel, so you see
why cell 8 ends up hottest. The pump's energy meter grows as the cube of flow — push the flow to
maximum and it leaps to 40 W.

*Establishes:* cooling works, and it is not free. `P_pump = 40 W × (ṁ / 0.05)³`.

### 3. Blindness

The reveal, and the beat that makes the project make sense. Six cells drop to ghosted wireframe.
Cells 1 and 4 stay solid, each with a sensor beacon whose readout jitters by ±0.5 K. The camera
drops to the controller's eye level, where only those two readings and the coolant inlet and outlet
exist. Cell 8 — still the hottest, still climbing — is a ghost.

*Establishes:* the cell that decides safety is the one nobody can see.

### 4. Estimation

Translucent shells inflate around all eight cells at ±2σ from the Kalman filter, then contract as
the filter locks on. Truth stays faintly visible inside, so the estimate can be seen chasing it.
Because the filter runs on nominal parameters while the plant runs perturbed ones (`h0` ±20 %,
`C_cell` ±10 %), the shells do not collapse to nothing, and must not be drawn as though they do.

*Establishes:* physics plus two noisy sensors recovers the hidden cells, with quantified doubt.

### 5. Control

Up to four packs appear in a row in the same space, same held-out profile, synchronised clock:
constant maximum flow, thermostat, PID, PPO-Lagrangian. Constant-flow glows cold while its energy
meter runs away; the fixed-penalty PPO rides the limit. Four packs in one scene beats four line
charts, because the eye compares spatial patterns faster than it compares traces.

*Establishes:* E1 and E2, as an object rather than a table.

### 6. The trade-off space

The camera pulls back and each pack collapses into a point in an energy-versus-violations plane you
can fly through, one point per seed per configuration. Click a point and it expands back into a
running pack.

*Establishes:* the Pareto result, and — more importantly — the link between an aggregate point and
the episode that produced it. Most dashboards never connect those two.

### 7. Your turn

The sandbox. You hold the flow slider against a profile you have not seen. The actuator rate limit
means you cannot react instantly: your commanded flow is drawn as a ghost ahead of the actual flow.
When you fail, or when you have spent enough energy, the agent takes over on the same profile —
seeing two cells out of eight.

*Establishes:* the thesis, as something that happened to you rather than something you were told.

**Degradation rule.** Beats 4–6 need data from milestones M2–M4, which do not exist yet. A beat
whose trace files are absent is **hidden, not faked**. The viewer never ships placeholder physics
dressed as a result. This is the README's design principle #1 applied to the viewer.

---

## The layer system

Beats are presets over an orthogonal set of layers. In lab mode every layer is an independent
toggle, which is what makes the thing useful for debugging rather than only for showing off.

| Layer | Draws | Source field |
|---|---|---|
| **Truth** | Solid cells shaded by true temperature | `T_cells_C` |
| **Coolant** | Channel, particle flow, per-segment coolant temperature | `T_coolant_C`, `mdot` |
| **Sensors** | Beacons on sensed cells, noisy readouts, ghosting of the rest | `sensors.*` |
| **Estimate** | ±2σ shells and estimated means on all cells | `estimate.mean_C`, `estimate.var_K2` |
| **Limits** | The `T_max` plane, spread indicator, violation flashes | `T_max_C`, `spread_K`, `violation` |
| **Actuation** | Actual flow, commanded-flow ghost, rate-limit clamp events | `mdot`, `mdot_cmd`, `rate_limited` |
| **Heat ledger** | Where the heat goes: generated, to coolant, to ambient, conducted, stored | `energy_J.*` |
| **Multiplier** | PPO-Lagrangian's λ as a tipping balance between safe and cheap | `lambda` |

The heat ledger deserves a note. The environment already logs a closed energy balance
(`energy_gen_J`, `energy_cool_J`, `energy_amb_J`, `energy_cond_J`, `stored_change_J`) and there is
already a test asserting that it closes. That makes a flow diagram of heat through the pack both
free to build and verified, which is a rare combination.

---

## Encoding table

The discipline that keeps this from becoming an animated screensaver: **every visual channel maps to
one logged quantity.** If a channel is not in this table, it is not allowed to vary.

| Channel | Encodes | Domain |
|---|---|---|
| Cell fill colour | True cell temperature | Fixed `[30, 45] °C`, with an overflow colour above |
| Cell emissive intensity | Degrees above `T_max` | `[0, 5] K`, zero below the limit |
| Cell outline pulse | Violation flag for this cell | Binary |
| Cell opacity | Whether the controller can observe this cell | Binary: solid or ghosted |
| Shell thickness | Kalman ±2σ | `[0, 5] K` |
| Particle speed | Coolant mass flow | `[0.002, 0.05] kg/s` |
| Particle colour | Coolant segment temperature | Same ramp as cells |
| Pump meter height | Instantaneous pump power | `[0, 40] W` |
| Ghost-flow offset | Commanded minus actual flow | `± 0.0096 kg/s`, the rate limit |
| Current ribbon width | Load current | `[0, max profile current]` |

Two rules about the colour domain. It is **fixed**, not fitted per run: a domain rescaled to each
run's own min and max would make a safe run and a dangerous one look identical, which is the single
easiest way to lie with a visualisation. And the limit at 45 °C gets **three** redundant cues —
geometry plane, numeric readout, pulse — because colour alone fails for a colourblind viewer and on
a bad projector.

---

## Data contract

### Current state of the data

`results/m1_constant_flow_sweep.csv` is one row per run: aggregate metrics only. The viewer needs
per-timestep state, which nothing currently writes. The first deliverable is therefore not
application code but **`scripts/export_trace.py`** — run one episode under one controller, write one
trace file.

### Sizing, which settles the format question

An episode is 3600 s with a 10 s control interval, so 360 control steps. A full trace is roughly
360 × 65 numbers, about 23 000 values: around 300 KB of raw JSON, near 60 KB gzipped. Twenty
held-out profiles across four controllers is a few megabytes.

**So plain gzipped JSON is sufficient.** No binary format, no streaming, no backend. Traces ship as
static assets and the whole app deploys to GitHub Pages.

Playback note: 360 steps at one step per frame is six seconds of animation for a simulated hour.
That compression ratio suits story mode; lab mode wants 0.25×–4× and frame-accurate scrubbing.

### Schema

`kelvin.trace/1`. Field names follow the environment's `info` dict, so the exporter stays close to a
transcription rather than a translation.

```json
{
  "schema": "kelvin.trace/1",
  "meta": {
    "git_commit": "fd4fedc",
    "config_sha256": "...",
    "exported_utc": "2026-11-09T00:00:00Z",
    "env": {
      "n_cells": 8, "T_max_C": 45.0, "dT_max_K": 5.0,
      "mdot_min": 0.002, "mdot_max": 0.05, "delta_mdot_max": 0.0096,
      "T_in_C": 30.0, "T_amb_C": 30.0, "P_pump_max_W": 40.0,
      "dt_ctrl_s": 10.0, "dt_phys_s": 1.0, "sensor_cells": [0, 3]
    },
    "run": {
      "controller": "ppo_lagrangian", "obs_mode": "estimated",
      "seed": 0, "split": "test", "profile_index": 7, "mismatch_scale": 1.0
    },
    "source_csv": "results/e3/ppo_lagrangian_estimated_seed0_test07.csv"
  },
  "episode": {
    "n_steps": 360,
    "summary": {
      "pump_energy_kJ": 0.0, "peak_T_C": 0.0, "pct_steps_violating": 0.0,
      "degree_s_above": 0.0, "max_spread_K": 0.0, "return": 0.0, "cost_total": 0.0
    }
  },
  "series": {
    "t_s": [],
    "I_A": [],
    "T_cells_C": [[]],
    "T_coolant_C": [[]],
    "T_peak_substep_C": [],
    "mdot": [],
    "mdot_cmd": [],
    "rate_limited": [],
    "pump_energy_J": [],
    "cost": [],
    "violation": [],
    "spread_K": [],
    "energy_J": { "gen": [], "cool": [], "amb": [], "cond": [], "stored_change": [] },
    "sensors": { "cells_C": [[]], "T_in_C": [], "T_out_C": [] },
    "estimate": { "mean_C": [[]], "var_K2": [[]] },
    "lambda": []
  }
}
```

Rules:

- `series.estimate` is present only for `obs_mode: "estimated"`, and `series.lambda` only for
  PPO-Lagrangian. **Absent, never null-filled** — the layer then has nothing to draw and hides
  itself.
- `meta.git_commit` and `meta.config_sha256` are mandatory, and the viewer displays them. Any number
  on screen can be traced back to the commit and config that produced it.
- A `traces/manifest.json` indexes available traces by controller, observation mode, seed and
  profile, so lab mode's selector is generated rather than hardcoded.
- The schema version bumps on any breaking change, and the loader refuses an unknown major version
  rather than rendering a half-understood file.

---

## The sandbox and its parity gate

The sandbox needs physics in the browser, which means a second implementation of the model. That is
a real hazard: an unvalidated JavaScript port that drifts from the NumPy one turns the viewer into
exactly the hand-typed result the project forbids.

Mitigation, in three parts.

1. **Separation.** Replayed traces and sandbox simulation are visually distinct and labelled. Any
   screen fed by the JS port carries a `simulated in-browser` marker. No figure or claim ever comes
   from the sandbox.
2. **A golden-vector parity test.** A script writes `viz/tests/fixtures/golden_physics.json` from
   the NumPy model: a fixed profile, fixed initial temperatures, a fixed flow schedule, and the
   resulting per-step cell temperatures. The committed fixture is asserted against by the JS test
   suite in CI. Both sides perform the same explicit-Euler arithmetic in double precision, so
   agreement to about 1e-6 K over 3600 substeps is the expectation; the tolerance is set there and
   failing it fails the build. Changing `physics.py` without regenerating the fixture breaks CI,
   which is the point.
3. **Publishing the number.** The app's about panel states the measured maximum deviation from the
   reference implementation, so the honesty is visible rather than merely documented.

The policy runs client-side from exported weights. The networks are small enough that a hand-rolled
matrix multiply is adequate, which avoids a runtime inference dependency.

---

## Visual language

- **Temperature ramp.** Perceptually uniform and sequential, cool neutral through to hot, with a
  distinct overflow colour above 45 °C. Not a rainbow ramp: rainbow invents boundaries where the
  data has none and ranks badly for colour vision deficiency.
- **Theme.** Dark primary, because emissive heat reads better against it. Light theme supported by
  token swap, with the ramp re-checked for contrast rather than reused unchanged.
- **Typography.** One geometric sans. Tabular numerals everywhere a number updates per frame, so
  the digits do not jitter.
- **Geometry.** Schematic, not CAD. The cells are blocks and the channel is a tube. The viewer
  states that it depicts a lumped-parameter model and not a real pack — the same limitation the
  README carries, in the same words.
- **Restraint.** Bloom only on genuinely over-limit cells. If everything glows, nothing reads.

---

## Motion and choreography

Motion here is explanatory, so it gets the same discipline as colour.

- **Camera.** Eased moves, 600–900 ms, cubic in-out, no overshoot and no bounce. The camera moves
  between beats and is still within a beat. It never drifts under its own momentum.
- **Reveals.** Layer changes stagger 40 ms per cell from the inlet end, so the eye follows the
  coolant direction and learns the pack's orientation without being told.
- **The simulation clock is separate from the UI clock.** One authoritative frame index advances the
  data; transitions animate the presentation. Never interpolate between physics steps and present
  the result as state — interpolate the rendering, display the step.
- **Accessibility.** Under `prefers-reduced-motion`, transitions cut rather than animate and the
  particle field becomes static arrows. Data playback is preserved, because that is content rather
  than decoration.
- **The loop.** One beat is cut to a 20-second silent loop for the README hero. Candidate: beat 3,
  blindness, which is the most self-explanatory image in the project.

---

## Architecture

```
viz/
├── index.html
├── src/
│   ├── scene/            Pack, CoolantChannel, SensorBeacons, UncertaintyShells,
│   │                     LimitPlane, PumpUnit, HeatLedger, ComparisonRail, ParetoSpace
│   ├── story/            beat definitions: camera, layers, copy
│   ├── sim/              JS physics port + policy runner (sandbox only)
│   ├── data/             trace loader, schema validation, manifest
│   ├── state/            one store: { trace, frame, layers, mode, camera }
│   └── ui/               scrubber, layer toggles, readouts, about panel
├── public/traces/        exported traces + manifest.json
└── tests/fixtures/       golden_physics.json
```

- **React Three Fiber and drei**, because every frame is "render the state at index `t`", and
  declarative state-to-scene binding is precisely that shape. Zustand for the store, Vite to build.
- **Derived rendering, single clock.** Exactly one `useFrame` advances the frame index, and every
  component is a pure function of `(trace, frame, layers)`. No component holds animation state of
  its own. This is what makes scrubbing, pausing and reverse playback work for free instead of being
  three separate features.
- **Performance is a non-issue for the pack**: eight instanced boxes. The only real cost is the
  coolant particle field, which is one instanced buffer driven by a shader with a capped count.
- **Deployment.** Static build to GitHub Pages from `viz/`, so the README can link to a live URL.

---

## Build order

Tied to the research roadmap, and deliberately never on its critical path. The viewer only ever
consumes traces, so no milestone waits for it.

| Stage | Needs | Delivers |
|---|---|---|
| **V0** | M1, done | `scripts/export_trace.py`, the schema, the loader, Pack + Coolant + Limits layers, lab-mode scrubber, beats 1–3 |
| **V1** | M2 baselines | Comparison rail, heat ledger, actuation layer with the commanded-flow ghost, beat 5 for classical controllers |
| **V2** | M3 PPO runs | PPO and PPO-Lagrangian in the rail, Pareto flythrough, multiplier layer, beat 6 |
| **V3** | M4 Kalman | Estimate layer, ±2σ shells, beat 4 — the estimation story completes |
| **V4** | JS port and parity fixture | Sandbox, you-versus-agent handoff, beat 7 |

V0 is buildable today. Beat 3 works immediately: ghosting unsensed cells needs only the sensor
configuration, which already exists, and `obs_mode: "partial"` already emits noisy readings.

---

## Risks

| Risk | Mitigation |
|---|---|
| JS port diverges from NumPy and the app shows fiction | Golden-vector fixture asserted in CI, deviation published in-app, sandbox labelled and never cited |
| Beautiful but uninformative | The encoding table is binding: a visual channel not listed there may not vary |
| Colour domain fitted per run flatters dangerous runs | Fixed `[30, 45] °C` domain with an explicit overflow band |
| Viewer competes with research for time | Viewer consumes traces only, and stages are gated behind milestones that must land anyway |
| Over-claiming physical fidelity | The README's limitations are reproduced in the app, not summarised away |
| Missing data silently faked | Absent trace blocks hide their layer; no placeholder physics in a replay view |

---

## Non-goals

- Not a replacement for `scripts/make_figures.py`. Paper figures stay static, reproducible and
  generated from CSVs.
- Not a training dashboard. TensorBoard or an equivalent handles curves during training.
- Not a validated battery simulator, and the app says so.
- No backend, no accounts, no telemetry. Static files on a CDN.

---

## Open questions

1. **Does the pack render as eight prismatic cells, or as a cutaway with a visible cold plate?** The
   cutaway explains the coolant path better; plain blocks are more honest about how crude a
   lumped-parameter model is.
2. **Is beat 6, the Pareto flythrough, worth the build?** It is the most technically involved beat
   and the one most easily replaced by a good static scatter plot with a hover.
3. **Does the sandbox expose model mismatch as a slider?** Letting a visitor crank `mismatch.scale`
   and watch the Kalman shells lose calibration is a strong demonstration, and also the easiest
   place for a visitor to reach a conclusion the research has not made.
4. **Same repo, or separate?** The same repo keeps the trace schema and the exporter in one place,
   at the cost of a Node toolchain in a Python project's CI.
5. **How long should story mode be?** Seven beats is roughly two minutes. Three beats would be
   thirty seconds but would lose the estimation story, which is the most distinctive part.

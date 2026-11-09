# Kelvin Viewer — V0

Stage V0 of [the viewer design](../docs/viz-design.md): the pack, the coolant channel, the limit
plane, the sensor layer, and a scrubber over a logged episode.

## What this is

One self-contained `index.html` using Three.js from a CDN. No build step and no package manager:
the page is the whole application. That is a deliberate V0 choice, not the end state — it gets a
working scene in front of people before committing to a toolchain, and it still deploys as a static
file.

The design document specifies React Three Fiber and Vite from V1 onward, where the comparison rail,
the Pareto flythrough and the estimator layer arrive. Nothing here blocks that; the data contract is
the only thing the two share, and it does not change.

## Running it

The page reads traces over HTTP, so opening the file directly with `file://` will not work — the
fetch is blocked. Serve the repository root:

```bash
python scripts/export_trace.py --controller constant --action 0.1 --obs-mode partial --split test --profile-index 0 --seed 0
python -m http.server 8000
# then open http://localhost:8000/viz/
```

It looks for traces in `traces/` first (the layout used when the page is published on its own) and
falls back to `../results/traces/` (the layout inside this repository), so the same file works in
both places.

## What it shows

| Element | Encodes |
|---|---|
| Cell height | Cell temperature, over a fixed 30–45 °C domain |
| Cell colour | The same temperature, on a sequential ramp with rising lightness |
| Cell emissive glow | Degrees above `T_max`, zero below it |
| Cell ghosting | Whether the controller can observe that cell |
| Particle speed | Coolant mass flow |
| Particle colour | Coolant segment temperature |
| Plane at 45 °C | `T_max`, as geometry the cells can rise through |

Height is an addition to the encoding table in the design document: it makes the limit a plane that
cells visibly break through, rather than a number you have to read. The colour domain is fixed at
30–45 °C and never fitted to the run on screen, because per-run scaling would make a safe episode
and a dangerous one look identical.

## Honest limits

- Every number on screen comes from a logged trace. Nothing is simulated in the browser; the
  sandbox and its parity gate are stage V4.
- The three view presets move the camera and toggle layers. They do not change the physics: the
  episode on screen is whichever logged run is selected.
- The file is written as page content without a `<!doctype>` wrapper, so that it can be published
  directly as an artifact. Served standalone it renders in quirks mode; every rule here sets
  `box-sizing` explicitly so that is cosmetically harmless, and the V1 build step will emit a full
  document.
- Only one trace is committed, per the repository's fixture policy. Generate the others with
  `scripts/export_trace.py`; the page lists whatever the manifest contains.

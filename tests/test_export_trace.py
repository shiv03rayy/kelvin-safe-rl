"""The trace exporter: schema conformance, and that every number comes from the env.

The expected key sets below are transcribed from the ``kelvin.trace/1`` block of
``docs/viz-design.md``. They are written out literally rather than derived from
the exporter, so this file is the schema and the exporter has to match it.
"""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from kelvin_rl.envs import BatteryPackCoolingEnv

from conftest import run_constant

# scripts/ is not a package; the exporter is a script, like plot_constant_flow.py.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))

import export_trace  # noqa: E402

ACTION = 0.3
PROFILE = 7
SEED = 0

TRACE_KEYS = {"schema", "meta", "episode", "series"}
META_KEYS = {"git_commit", "config_sha256", "exported_utc", "env", "run"}
ENV_KEYS = {
    "n_cells", "T_max_C", "dT_max_K", "mdot_min", "mdot_max", "delta_mdot_max",
    "T_in_C", "T_amb_C", "P_pump_max_W", "dt_ctrl_s", "dt_phys_s", "sensor_cells",
}
RUN_KEYS = {"controller", "obs_mode", "seed", "split", "profile_index", "mismatch_scale"}
SUMMARY_KEYS = {
    "pump_energy_kJ", "peak_T_C", "pct_steps_violating",
    "degree_s_above", "max_spread_K", "return", "cost_total",
}
REQUIRED_SERIES = {
    "t_s", "I_A", "T_cells_C", "T_coolant_C", "T_peak_substep_C", "mdot", "mdot_cmd",
    "rate_limited", "pump_energy_J", "cost", "violation", "spread_K", "energy_J", "sensors",
}
OPTIONAL_SERIES = {"estimate", "lambda"}
ENERGY_KEYS = {"gen", "cool", "amb", "cond", "stored_change"}
SENSOR_KEYS = {"cells_C", "T_in_C", "T_out_C"}


def export_to(tmp_path, *extra, name="trace.json"):
    """Export one trace into a temp directory and return (path, parsed trace)."""
    out = tmp_path / name
    argv = [
        "--controller", "constant", "--action", str(ACTION), "--obs-mode", "partial",
        "--split", "test", "--profile-index", str(PROFILE), "--seed", str(SEED),
        "--out", str(out), *extra,
    ]
    args = export_trace.build_parser().parse_args(argv)
    path, trace = export_trace.export(args)
    assert path == out
    with open(path, encoding="utf-8") as f:
        return path, json.load(f)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One exported trace, read back from disk, shared by the schema tests."""
    return export_to(tmp_path_factory.mktemp("traces"))


# --------------------------------------------------------------------- schema
def test_schema_version(exported):
    _, trace = exported
    assert trace["schema"] == "kelvin.trace/1"


def test_top_level_keys_and_nesting(exported):
    _, trace = exported
    assert set(trace) == TRACE_KEYS
    assert set(trace["meta"]) == META_KEYS
    assert set(trace["meta"]["env"]) == ENV_KEYS
    assert set(trace["meta"]["run"]) == RUN_KEYS
    assert set(trace["episode"]) == {"n_steps", "summary"}
    assert set(trace["episode"]["summary"]) == SUMMARY_KEYS
    assert set(trace["series"]) - OPTIONAL_SERIES == REQUIRED_SERIES
    assert set(trace["series"]["energy_J"]) == ENERGY_KEYS
    assert set(trace["series"]["sensors"]) == SENSOR_KEYS


def test_meta_env_matches_the_environment(exported):
    _, trace = exported
    env = BatteryPackCoolingEnv(split="test")
    e = trace["meta"]["env"]
    assert e["n_cells"] == env.n
    assert e["T_max_C"] == env.T_max
    assert e["dT_max_K"] == env.dT_max
    assert (e["mdot_min"], e["mdot_max"]) == (env.mdot_min, env.mdot_max)
    assert e["delta_mdot_max"] == pytest.approx(env.delta_mdot_max)
    assert e["T_in_C"] == env.nominal_params.T_in
    assert e["T_amb_C"] == env.nominal_params.T_amb
    assert e["P_pump_max_W"] == env.nominal_params.P_pump_max
    assert (e["dt_ctrl_s"], e["dt_phys_s"]) == (env.dt_ctrl, env.dt_phys)
    assert e["sensor_cells"] == list(env.sensor_idx)


def test_meta_run_records_the_request(exported):
    _, trace = exported
    assert trace["meta"]["run"] == {
        "controller": f"constant_a{ACTION:.2f}",
        "obs_mode": "partial",
        "seed": SEED,
        "split": "test",
        "profile_index": PROFILE,
        "mismatch_scale": 1.0,
    }


def test_commit_and_config_hash_are_present_and_true(exported):
    """Both are mandatory: they are what lets a number on screen be traced back."""
    _, trace = exported
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=export_trace.REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert trace["meta"]["git_commit"] == head != ""
    expected = hashlib.sha256(export_trace.DEFAULT_CONFIG.read_bytes()).hexdigest()
    assert trace["meta"]["config_sha256"] == expected
    assert trace["meta"]["exported_utc"].endswith("Z")


# --------------------------------------------------------------------- shapes
def test_every_series_has_one_entry_per_step(exported):
    _, trace = exported
    n = trace["episode"]["n_steps"]
    assert n == 360
    series = trace["series"]
    for key in REQUIRED_SERIES - {"energy_J", "sensors"}:
        assert len(series[key]) == n, key
    for key, values in series["energy_J"].items():
        assert len(values) == n, f"energy_J.{key}"
    for key, values in series["sensors"].items():
        assert len(values) == n, f"sensors.{key}"


def test_per_cell_arrays_are_pack_width(exported):
    _, trace = exported
    n_cells = trace["meta"]["env"]["n_cells"]
    n_sensors = len(trace["meta"]["env"]["sensor_cells"])
    series = trace["series"]
    for key in ("T_cells_C", "T_coolant_C"):
        assert {len(row) for row in series[key]} == {n_cells}, key
    assert {len(row) for row in series["sensors"]["cells_C"]} == {n_sensors}


def test_time_axis_is_the_control_grid(exported):
    _, trace = exported
    dt = trace["meta"]["env"]["dt_ctrl_s"]
    n = trace["episode"]["n_steps"]
    assert trace["series"]["t_s"] == [(k + 1) * dt for k in range(n)]
    assert trace["series"]["t_s"][-1] == pytest.approx(3600.0)


def test_flags_are_json_booleans(exported):
    _, trace = exported
    for key in ("violation", "rate_limited"):
        assert all(isinstance(v, bool) for v in trace["series"][key]), key


# ------------------------------------------------------- optional blocks
def test_optional_blocks_are_absent_not_null(exported):
    """A constant controller in a non-estimated mode has neither block."""
    path, trace = exported
    assert "estimate" not in trace["series"]
    assert "lambda" not in trace["series"]
    assert "null" not in path.read_text(encoding="utf-8")


def test_estimate_block_is_omitted_until_a_filter_logs_it():
    infos = [{"estimate_mean_C": [30.0, 31.0], "estimate_var_K2": [0.1, 0.2]}]
    for mode in ("full", "partial", "partial_history"):
        assert export_trace.estimate_block(mode, infos) is None
    block = export_trace.estimate_block("estimated", infos)
    assert block == {"mean_C": [[30.0, 31.0]], "var_K2": [[0.1, 0.2]]}


def test_estimated_mode_without_estimator_fields_raises_rather_than_filling():
    with pytest.raises(KeyError):
        export_trace.estimate_block("estimated", [{"T_cells": [30.0]}])


class StubLagrangian:
    """Stand-in for the M3 PPO-Lagrangian controller: the lambda hook only."""

    name = "stub_lagrangian"

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return np.array([self.action], dtype=np.float32)

    def lambda_value(self):
        return float(self.calls)


def test_lambda_series_appears_only_for_a_controller_that_has_one(tmp_path):
    env = BatteryPackCoolingEnv(split="test")
    controller = StubLagrangian(ACTION)
    rec = export_trace.rollout(env, controller, seed=SEED, profile_index=PROFILE)
    trace = export_trace.build_trace(
        env, controller, rec, seed=SEED, split="test",
        config_path=export_trace.DEFAULT_CONFIG,
    )
    n = trace["episode"]["n_steps"]
    assert trace["series"]["lambda"] == [float(k + 1) for k in range(n)]
    assert "estimate" not in trace["series"]

    plain = export_trace.ConstantAction(ACTION)
    rec_plain = export_trace.rollout(env, plain, seed=SEED, profile_index=PROFILE)
    assert rec_plain.lambdas is None


# ------------------------------------------------------- numbers come from env
def test_series_transcribe_the_env_info_dict(exported):
    """The same episode run through the plain env must produce the same numbers."""
    _, trace = exported
    env = BatteryPackCoolingEnv(obs_mode="partial", split="test")
    infos = run_constant(
        env, ACTION, seed=SEED, options={"profile_index": PROFILE, "action_init": ACTION}
    )
    series = trace["series"]
    assert len(infos) == trace["episode"]["n_steps"]
    np.testing.assert_allclose(series["T_cells_C"], [i["T_cells"] for i in infos])
    np.testing.assert_allclose(series["T_coolant_C"], [i["T_coolant"] for i in infos])
    np.testing.assert_allclose(series["T_peak_substep_C"], [i["T_peak_substep"] for i in infos])
    np.testing.assert_allclose(series["spread_K"], [i["spread"] for i in infos])
    np.testing.assert_allclose(series["mdot"], [i["mdot"] for i in infos])
    np.testing.assert_allclose(series["mdot_cmd"], [i["mdot_cmd"] for i in infos])
    np.testing.assert_allclose(series["pump_energy_J"], [i["pump_energy_J"] for i in infos])
    np.testing.assert_allclose(series["cost"], [i["cost"] for i in infos])
    assert series["violation"] == [bool(i["violation"]) for i in infos]
    assert series["rate_limited"] == [bool(i["rate_limited"]) for i in infos]
    for name, key in export_trace.ENERGY_FIELDS.items():
        np.testing.assert_allclose(series["energy_J"][name], [i[key] for i in infos])


def test_current_series_is_the_profile_at_step_boundaries(exported):
    _, trace = exported
    env = BatteryPackCoolingEnv(split="test")
    current = env.profiles.get("test", PROFILE).current
    expected = [current[k * env.substeps] for k in range(trace["episode"]["n_steps"])]
    np.testing.assert_allclose(trace["series"]["I_A"], expected)


def test_summary_agrees_with_the_series_it_came_from(exported):
    _, trace = exported
    series, s = trace["series"], trace["episode"]["summary"]
    n, dt = trace["episode"]["n_steps"], trace["meta"]["env"]["dt_ctrl_s"]
    assert s["pump_energy_kJ"] == pytest.approx(sum(series["pump_energy_J"]) / 1e3)
    assert s["peak_T_C"] == pytest.approx(max(series["T_peak_substep_C"]))
    assert s["pct_steps_violating"] == pytest.approx(100.0 * sum(series["violation"]) / n)
    assert s["degree_s_above"] == pytest.approx(sum(series["cost"]) * dt)
    assert s["max_spread_K"] == pytest.approx(max(series["spread_K"]))
    assert s["cost_total"] == pytest.approx(sum(series["cost"]))
    # The reward is normalised pump energy, so the return follows the same series.
    e_norm = trace["meta"]["env"]["P_pump_max_W"] * dt
    assert s["return"] == pytest.approx(-sum(series["pump_energy_J"]) / e_norm)
    assert s["peak_T_C"] >= max(max(row) for row in series["T_cells_C"])


def test_energy_ledger_closes_inside_the_trace(exported):
    """The heat-ledger layer is only honest if the trace's own numbers balance."""
    _, trace = exported
    e = trace["series"]["energy_J"]
    gen, stored = sum(e["gen"]), sum(e["stored_change"])
    assert gen > 0
    assert gen == pytest.approx(stored + sum(e["cool"]) + sum(e["amb"]), rel=1e-9, abs=1e-6)
    assert abs(sum(e["cond"])) < 1e-6 * gen


def test_sensor_readings_are_the_noisy_ones_the_controller_saw(exported):
    """In a partial mode they must differ from truth by about sigma_cell, not match it."""
    _, trace = exported
    env = BatteryPackCoolingEnv(obs_mode="partial", split="test")
    idx = trace["meta"]["env"]["sensor_cells"]
    truth = np.array(trace["series"]["T_cells_C"])[:, idx]
    read = np.array(trace["series"]["sensors"]["cells_C"])
    error = read - truth
    assert abs(error.mean()) < 0.1
    assert error.std() == pytest.approx(env.sigma_cell, rel=0.15)
    assert np.all(np.array(trace["series"]["sensors"]["T_in_C"]) > 0)


def test_full_mode_sensor_readings_are_the_true_temperatures(tmp_path):
    _, trace = export_to(tmp_path, "--obs-mode", "full")
    idx = trace["meta"]["env"]["sensor_cells"]
    truth = np.array(trace["series"]["T_cells_C"])[:, idx]
    read = np.array(trace["series"]["sensors"]["cells_C"])
    # float32 observations, so the inverse scaling round-trips to ~1e-5 K.
    np.testing.assert_allclose(read, truth, atol=1e-4)
    assert {len(row) for row in trace["series"]["sensors"]["cells_C"]} == {len(idx)}


def test_history_mode_decodes_the_newest_frame_of_the_stack():
    """Temperatures move slowly, so only a synthetic stack can prove which frame is read."""
    env = BatteryPackCoolingEnv(obs_mode="partial_history", split="test")
    k = len(env.sensor_idx)

    def frame(T):
        return np.concatenate([np.full(k + 2, (T - env.T_ref) / env.T_scale), [0.0, 0.0, 0.0]])

    obs = np.concatenate([frame(31.0)] * (env.history_len - 1) + [frame(42.0)])
    cells, T_in, T_out = export_trace.sensor_readings(env, obs)
    assert cells == pytest.approx([42.0] * k)
    assert (T_in, T_out) == pytest.approx((42.0, 42.0))


def test_history_mode_exports_like_the_other_partial_modes(tmp_path):
    _, trace = export_to(tmp_path, "--obs-mode", "partial_history")
    idx = trace["meta"]["env"]["sensor_cells"]
    assert {len(row) for row in trace["series"]["sensors"]["cells_C"]} == {len(idx)}
    error = np.array(trace["series"]["sensors"]["cells_C"]) - np.array(
        trace["series"]["T_cells_C"]
    )[:, idx]
    assert error.std() == pytest.approx(0.5, rel=0.3)  # sensor noise, not a decoding error


# ------------------------------------------------------------------- manifest
def test_manifest_round_trips(tmp_path):
    path_a, trace_a = export_to(tmp_path, name="a.json")
    path_b, trace_b = export_to(tmp_path, "--action", "0.9", name="b.json")

    with open(tmp_path / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["schema"] == "kelvin.trace.manifest/1"
    assert [e["file"] for e in manifest["traces"]] == ["a.json", "b.json"]
    for entry, trace in zip(manifest["traces"], (trace_a, trace_b)):
        run = trace["meta"]["run"]
        assert entry["controller"] == run["controller"]
        assert entry["obs_mode"] == run["obs_mode"]
        assert entry["seed"] == run["seed"]
        assert entry["split"] == run["split"]
        assert entry["profile_index"] == run["profile_index"]
        assert entry["n_steps"] == trace["episode"]["n_steps"]
        assert entry["git_commit"] == trace["meta"]["git_commit"]
        assert entry["config_sha256"] == trace["meta"]["config_sha256"]
    assert {e["controller"] for e in manifest["traces"]} == {"constant_a0.30", "constant_a0.90"}
    assert all((tmp_path / e["file"]).exists() for e in manifest["traces"])


def test_manifest_is_rebuilt_not_appended(tmp_path):
    export_to(tmp_path, name="a.json")
    export_to(tmp_path, "--action", "0.9", name="b.json")
    (tmp_path / "b.json").unlink()
    manifest = export_trace.rebuild_manifest(tmp_path)
    assert [e["file"] for e in manifest["traces"]] == ["a.json"]

    # Re-exporting the same trace updates its entry instead of duplicating it.
    export_to(tmp_path, name="a.json")
    with open(tmp_path / "manifest.json", encoding="utf-8") as f:
        assert [e["file"] for e in json.load(f)["traces"]] == ["a.json"]


def test_manifest_ignores_foreign_json(tmp_path):
    (tmp_path / "notes.json").write_text('{"schema": "something/else"}', encoding="utf-8")
    export_to(tmp_path, name="a.json")
    manifest = export_trace.rebuild_manifest(tmp_path)
    assert [e["file"] for e in manifest["traces"]] == ["a.json"]


def test_default_trace_name_follows_the_run(exported):
    _, trace = exported
    assert export_trace.trace_name(trace) == f"constant_a{ACTION:.2f}_partial_seed0_test{PROFILE:02d}"


def test_unknown_controller_is_rejected():
    with pytest.raises(SystemExit):
        export_trace.build_parser().parse_args(["--controller", "pid"])

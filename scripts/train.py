"""M3 training: one RL agent, one observation mode, one seed.

Writes everything a run needs to be reproduced or re-examined into
``results/m3/<run_id>/``:

* ``progress.csv`` -- one row per update: learning curves of return and cost,
  the loss diagnostics, and for PPO-Lagrangian the trajectory of the multiplier.
* ``meta.json`` -- the full hyperparameter set, the seed, the git commit, and
  the library versions.
* ``agent.pt`` -- the weights and the frozen observation normaliser.

Nothing here is an evaluation. Everything in ``progress.csv`` is measured on
*training* profiles, because that is what the agent was rolled out on; held-out
numbers come from ``scripts/evaluate_agents.py``. The two must never be mixed,
which is why the files are separate and the split is recorded in both.

Run with::

    python scripts/train.py --agent ppo_lagrangian --obs-mode full --seed 0
    python scripts/train.py --agent ppo --penalty 10 --seed 0 --total-steps 200000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kelvin_rl.agents import AGENTS, PPOConfig
from kelvin_rl.config import REPO_ROOT
from kelvin_rl.envs import BatteryPackCoolingEnv

TRAIN_SPLIT = "train"
RESULTS_DIR = REPO_ROOT / "results" / "m3"


def run_id(agent: str, obs_mode: str, seed: int, tag: str | None = None) -> str:
    """A directory name that says what the run was, so a results tree is readable."""
    parts = [agent, obs_mode, f"seed{seed}"]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def build_config(args) -> PPOConfig:
    cfg = PPOConfig(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma=args.gamma,
        penalty=args.penalty,
        cost_limit=args.cost_limit,
        lambda_lr=args.lambda_lr,
        norm_obs=not args.no_norm_obs,
    )
    if args.lambda_kp or args.lambda_kd:
        cfg.extra["lambda_kp"] = args.lambda_kp
        cfg.extra["lambda_kd"] = args.lambda_kd
    cfg.extra["obs_mode"] = args.obs_mode
    cfg.extra["split"] = TRAIN_SPLIT
    return cfg


def train(args) -> Path:
    cfg = build_config(args)
    tag = args.tag
    if tag is None and args.agent == "ppo":
        # The penalty is the swept variable in E2, so it belongs in the name.
        tag = f"pen{args.penalty:g}"
    out = Path(args.out_dir or RESULTS_DIR) / run_id(args.agent, args.obs_mode, args.seed, tag)

    def make_env():
        return BatteryPackCoolingEnv(obs_mode=args.obs_mode, split=TRAIN_SPLIT)

    agent = AGENTS[args.agent](make_env, cfg, seed=args.seed, run_dir=out)
    agent.write_meta({"obs_mode": args.obs_mode, "split": TRAIN_SPLIT, "run_id": out.name})
    print(
        f"training {args.agent} obs_mode={args.obs_mode} seed={args.seed} "
        f"for {cfg.n_updates} updates x {cfg.batch_size} steps "
        f"({cfg.n_updates * cfg.batch_size} total) -> {out}",
        flush=True,
    )
    rows = agent.learn(progress_every=args.log_every)
    agent.save(out / "agent.pt")

    last = rows[-1]
    print(
        f"\ndone. final-update training means: return={last['ep_return']:.3f} "
        f"cost={last['ep_cost']:.3f} viol%={last['ep_pct_violating']:.2f} "
        f"energy={last['ep_pump_energy_kJ']:.2f} kJ peak={last['ep_peak_T_C']:.2f} C"
    )
    print("these are TRAINING-profile numbers; evaluate on held-out profiles separately")
    (out / "summary.json").write_text(json.dumps(last, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=sorted(AGENTS), required=True)
    ap.add_argument("--obs-mode", default="full", choices=["full", "partial", "partial_history"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=460_800)  # 160 updates x 2880
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=360)  # one full episode per env
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--penalty", type=float, default=1.0, help="fixed-penalty PPO only")
    ap.add_argument("--cost-limit", type=float, default=1.0, help="PPO-Lagrangian only")
    ap.add_argument("--lambda-lr", type=float, default=0.05, help="PPO-Lagrangian only")
    ap.add_argument("--lambda-kp", type=float, default=0.0, help="PID-Lagrangian ablation")
    ap.add_argument("--lambda-kd", type=float, default=0.0, help="PID-Lagrangian ablation")
    ap.add_argument("--no-norm-obs", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--log-every", type=int, default=10)
    train(ap.parse_args())


if __name__ == "__main__":
    main()

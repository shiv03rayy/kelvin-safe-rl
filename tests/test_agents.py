"""Tests for the PPO agents.

These do not test that PPO *learns* -- that is what the E1/E2 experiments are
for, and a unit test that asserts learning is a slow flaky test. They test the
parts that are easy to get silently wrong and that no learning curve would
reveal: the advantage arithmetic, the truncation bootstrap, the Lagrange
multiplier's dynamics, reproducibility from a seed, and that a saved agent
reloads as the same policy.

The GAE cases are checked against values worked out by hand in the comments,
not against a second implementation -- a test that compares code to a copy of
itself only proves the copy was faithful.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kelvin_rl.agents import AGENTS, PPOConfig
from kelvin_rl.agents.common import RunningNorm, SyncEnvs, compute_gae, explained_variance
from kelvin_rl.agents.ppo import PPO, ActorCritic
from kelvin_rl.agents.ppo_lagrangian import PPOLagrangian
from kelvin_rl.envs.battery_pack import BatteryPackCoolingEnv


def make_env(obs_mode="full"):
    return lambda: BatteryPackCoolingEnv(obs_mode=obs_mode, split="train")


def tiny_cfg(**kw):
    """A config small enough to run in a test, shaped like the real one."""
    base = dict(total_steps=2 * 2 * 8, n_envs=2, n_steps=8, n_epochs=2, n_minibatches=2)
    base.update(kw)
    return PPOConfig(**base)


# ------------------------------------------------------------------- GAE
class TestGAE:
    def test_single_step_no_done_is_the_td_error(self):
        # T=1, one env. delta = r + gamma*V(s1) - V(s0) = 1 + 0.9*2 - 0.5 = 2.3
        adv, ret = compute_gae(
            rewards=np.array([[1.0]]),
            values=np.array([[0.5]]),
            dones=np.array([[0.0]]),
            bootstrap_values=np.array([[0.0]]),
            last_values=np.array([2.0]),
            gamma=0.9,
            gae_lambda=0.95,
        )
        assert adv[0, 0] == pytest.approx(2.3)
        assert ret[0, 0] == pytest.approx(2.8)  # advantage + value

    def test_two_steps_recursion(self):
        # gamma=1, lambda=1, V=0 everywhere, so advantages are reward-to-go:
        # A_1 = 2, A_0 = 1 + 2 = 3.
        adv, _ = compute_gae(
            rewards=np.array([[1.0], [2.0]]),
            values=np.zeros((2, 1)),
            dones=np.zeros((2, 1)),
            bootstrap_values=np.zeros((2, 1)),
            last_values=np.zeros(1),
            gamma=1.0,
            gae_lambda=1.0,
        )
        assert adv[1, 0] == pytest.approx(2.0)
        assert adv[0, 0] == pytest.approx(3.0)

    def test_truncation_bootstraps_from_the_final_observation(self):
        """The regression this whole module exists to prevent.

        Step 0 ends an episode. The value of its final observation is 10. A
        correct implementation uses that 10; one that treats truncation as
        termination uses 0 and reports an advantage 9 lower.
        """
        adv, _ = compute_gae(
            rewards=np.array([[1.0], [0.0]]),
            values=np.array([[0.0], [0.0]]),
            dones=np.array([[1.0], [0.0]]),
            bootstrap_values=np.array([[10.0], [0.0]]),
            last_values=np.zeros(1),
            gamma=0.9,
            gae_lambda=0.95,
        )
        assert adv[0, 0] == pytest.approx(1.0 + 0.9 * 10.0)

    def test_done_cuts_the_recursion_but_not_the_value_target(self):
        """Both halves of the (1 - d_t) factor at once.

        Step 0 is terminal-with-bootstrap, step 1 belongs to the next episode
        and carries a large advantage. That advantage must not leak backwards
        into step 0, while step 0's own one-step target must still include its
        bootstrap.
        """
        adv, _ = compute_gae(
            rewards=np.array([[0.0], [100.0]]),
            values=np.zeros((2, 1)),
            dones=np.array([[1.0], [0.0]]),
            bootstrap_values=np.array([[5.0], [0.0]]),
            last_values=np.zeros(1),
            gamma=1.0,
            gae_lambda=1.0,
        )
        assert adv[0, 0] == pytest.approx(5.0), "advantage leaked across an episode boundary"
        assert adv[1, 0] == pytest.approx(100.0)

    def test_terminated_passes_zero(self):
        """A genuinely terminated episode has no future, so bootstrap is zero."""
        adv, _ = compute_gae(
            rewards=np.array([[3.0]]),
            values=np.zeros((1, 1)),
            dones=np.array([[1.0]]),
            bootstrap_values=np.array([[0.0]]),
            last_values=np.array([99.0]),  # must be ignored: the step is done
            gamma=0.9,
            gae_lambda=0.95,
        )
        assert adv[0, 0] == pytest.approx(3.0)

    def test_discounting_decays_the_sum(self):
        adv, _ = compute_gae(
            rewards=np.ones((3, 1)),
            values=np.zeros((3, 1)),
            dones=np.zeros((3, 1)),
            bootstrap_values=np.zeros((3, 1)),
            last_values=np.zeros(1),
            gamma=0.5,
            gae_lambda=1.0,
        )
        assert adv[0, 0] == pytest.approx(1 + 0.5 + 0.25)

    def test_envs_are_independent(self):
        adv, _ = compute_gae(
            rewards=np.array([[1.0, 10.0], [1.0, 10.0]]),
            values=np.zeros((2, 2)),
            dones=np.array([[0.0, 1.0], [0.0, 0.0]]),
            bootstrap_values=np.zeros((2, 2)),
            last_values=np.zeros(2),
            gamma=1.0,
            gae_lambda=1.0,
        )
        assert adv[0, 0] == pytest.approx(2.0)  # env 0 ran on
        assert adv[0, 1] == pytest.approx(10.0)  # env 1 was cut


# ------------------------------------------------------------------ helpers
class TestRunningNorm:
    def test_matches_batch_statistics(self):
        """Ten incremental updates must agree with one batch computation.

        Not to machine precision: the ``count = 1e-4`` prior that keeps the
        first update from dividing by zero also pulls the mean a little towards
        zero, by about one part in 5e6 here. That bias is a property of the
        standard initialisation, not a bug, and it shrinks as data arrives.
        """
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 2.0, size=(500, 3))
        norm = RunningNorm(3)
        for chunk in np.array_split(x, 10):
            norm.update(chunk)
        assert norm.mean == pytest.approx(x.mean(axis=0), rel=1e-5)
        assert norm.var == pytest.approx(x.var(axis=0), rel=1e-4)
        # The bias is towards zero, and small.
        assert np.all(np.abs(norm.mean) <= np.abs(x.mean(axis=0)))

    def test_output_is_standardised_and_clipped(self):
        norm = RunningNorm(1, clip=2.0)
        norm.update(np.random.default_rng(1).normal(0, 1, size=(1000, 1)))
        assert abs(float(norm(np.array([[0.0]]))[0, 0])) < 0.2
        assert float(norm(np.array([[1e6]]))[0, 0]) == pytest.approx(2.0)

    def test_freezing_stops_adaptation(self):
        norm = RunningNorm(1)
        norm.update(np.zeros((10, 1)))
        before = norm.mean.copy()
        norm.frozen = True
        norm.update(np.full((10, 1), 100.0))
        assert norm.mean == pytest.approx(before)

    def test_round_trips_through_its_state_dict(self):
        norm = RunningNorm(2)
        norm.update(np.random.default_rng(2).normal(size=(50, 2)))
        other = RunningNorm(2)
        other.load_state_dict(norm.state_dict())
        x = np.random.default_rng(3).normal(size=(5, 2))
        assert other(x) == pytest.approx(norm(x))


def test_explained_variance_bounds():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == pytest.approx(1.0)
    assert explained_variance(np.full(4, y.mean()), y) == pytest.approx(0.0)
    assert explained_variance(-y, y) < 0.0


# --------------------------------------------------------------- SyncEnvs
class TestSyncEnvs:
    def test_shapes_and_cost_are_reported(self):
        envs = SyncEnvs(make_env(), n_envs=3, seed=1)
        obs = envs.reset()
        assert obs.shape == (3, envs.obs_dim)
        obs, reward, cost, done, final_obs, completed = envs.step(np.full((3, 1), 0.5))
        assert obs.shape == (3, envs.obs_dim)
        assert reward.shape == cost.shape == done.shape == (3,)
        assert np.all(reward <= 0.0), "reward is negative pumping energy"
        assert np.all(cost >= 0.0), "cost is degrees above the limit, never negative"
        assert not done.any() and not completed

    def test_seeds_are_distinct_and_derived_from_one_number(self):
        assert SyncEnvs(make_env(), 3, seed=7).seeds == [70_000, 70_001, 70_002]

    def test_autoreset_reports_the_episode_and_keeps_going(self):
        envs = SyncEnvs(make_env(), n_envs=2, seed=0)
        envs.reset()
        n_steps = envs.envs[0].n_ctrl_steps
        for _ in range(n_steps - 1):
            _, _, _, done, _, completed = envs.step(np.full((2, 1), 0.5))
            assert not done.any()
        obs, _, _, done, final_obs, completed = envs.step(np.full((2, 1), 0.5))
        assert done.all()
        assert len(completed) == 2
        ep = completed[0]
        assert ep["length"] == n_steps
        assert not ep["terminated"], "the env should truncate, never terminate"
        assert ep["pump_energy_kJ"] > 0.0
        assert 0.0 <= ep["pct_steps_violating"] <= 100.0
        # obs is already the next episode; final_obs is the one that ended.
        assert not np.allclose(obs, final_obs)
        # Counters reset, so the next episode is not credited with this one's.
        assert envs._ep_len.tolist() == [0, 0]

    def test_episode_counters_restart_cleanly(self):
        envs = SyncEnvs(make_env(), n_envs=1, seed=0)
        envs.reset()
        n_steps = envs.envs[0].n_ctrl_steps
        for _ in range(n_steps):
            envs.step(np.full((1, 1), 1.0))
        _, _, _, _, _, completed = envs.step(np.full((1, 1), 1.0))
        assert not completed  # one step into the new episode
        assert envs._ep_len.tolist() == [1]


# ------------------------------------------------------------------ policy
class TestActorCritic:
    def test_starts_near_mid_flow(self):
        """A fresh policy should be a sane controller, not a coin flip."""
        net = ActorCritic(obs_dim=13, act_dim=1)
        obs = torch.zeros(64, 13)
        with torch.no_grad():
            mean = net.actor_mean(obs)
        assert mean.mean().item() == pytest.approx(0.5, abs=0.05)

    def test_log_prob_is_of_the_sampled_action(self):
        net = ActorCritic(obs_dim=13, act_dim=1)
        obs = torch.zeros(4, 13)
        action, logprob, entropy = net.act(obs)
        recomputed, _ = net.evaluate(obs, action)
        assert recomputed.detach().numpy() == pytest.approx(logprob.detach().numpy(), abs=1e-6)
        assert action.shape == (4, 1)
        assert logprob.shape == entropy.shape == (4,)

    def test_deterministic_act_returns_the_mean(self):
        net = ActorCritic(obs_dim=13, act_dim=1)
        obs = torch.randn(8, 13)
        a1, _, _ = net.act(obs, deterministic=True)
        a2, _, _ = net.act(obs, deterministic=True)
        assert torch.allclose(a1, a2)
        assert torch.allclose(a1, net.actor_mean(obs))

    def test_cost_critic_is_absent_unless_asked_for(self):
        net = ActorCritic(obs_dim=13, act_dim=1, cost_critic=False)
        with pytest.raises(RuntimeError, match="without a cost critic"):
            net.cost_value(torch.zeros(1, 13))
        assert ActorCritic(13, 1, cost_critic=True).cost_value(torch.zeros(2, 13)).shape == (2,)

    def test_the_two_critics_do_not_share_parameters(self):
        """Separate trunks, so the reward and cost heads cannot fight over one."""
        net = ActorCritic(obs_dim=13, act_dim=1, cost_critic=True)
        shared = {id(p) for p in net.critic.parameters()} & {id(p) for p in net.cost_critic.parameters()}
        assert not shared


# ------------------------------------------------------------- the agents
class TestTraining:
    @pytest.mark.parametrize("algo", sorted(AGENTS))
    def test_one_update_runs_and_logs(self, algo, tmp_path):
        agent = AGENTS[algo](make_env(), tiny_cfg(), seed=0, run_dir=tmp_path)
        agent.write_meta()
        rows = agent.learn(progress_every=0)
        assert len(rows) == agent.cfg.n_updates
        row = rows[-1]
        assert row["global_step"] == agent.cfg.n_updates * agent.cfg.batch_size
        assert np.isfinite(row["pg_loss"]) and np.isfinite(row["v_loss"])
        assert row["approx_kl"] >= 0.0, "the k3 KL estimator is non-negative by construction"
        assert (tmp_path / "progress.csv").exists()
        assert (tmp_path / "meta.json").exists()

    @pytest.mark.parametrize("algo", sorted(AGENTS))
    def test_same_seed_same_run(self, algo):
        def run():
            agent = AGENTS[algo](make_env(), tiny_cfg(), seed=3)
            return agent.learn(progress_every=0)[-1]

        a, b = run(), run()
        for key in ("ep_return", "pg_loss", "v_loss", "ep_cost"):
            assert a[key] == pytest.approx(b[key], rel=1e-9, nan_ok=True), f"{key} is not reproducible"

    def test_different_seeds_diverge(self):
        first = AGENTS["ppo"](make_env(), tiny_cfg(), seed=0).learn(progress_every=0)[-1]
        second = AGENTS["ppo"](make_env(), tiny_cfg(), seed=1).learn(progress_every=0)[-1]
        assert first["ep_return"] != second["ep_return"] or first["pg_loss"] != second["pg_loss"]

    @pytest.mark.parametrize("algo", sorted(AGENTS))
    def test_checkpoint_reloads_as_the_same_policy(self, algo, tmp_path):
        agent = AGENTS[algo](make_env(), tiny_cfg(), seed=0)
        agent.learn(progress_every=0)
        path = tmp_path / "agent.pt"
        agent.save(path)
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["algo"] == algo
        clone = AGENTS[algo](make_env(), tiny_cfg(), seed=0)
        clone.net.load_state_dict(ckpt["model"])
        clone.obs_norm.load_state_dict(ckpt["obs_norm"])
        obs = torch.randn(16, agent.obs_dim)
        with torch.no_grad():
            assert torch.allclose(agent.net.actor_mean(obs), clone.net.actor_mean(obs))
            assert torch.allclose(agent.net.value(obs), clone.net.value(obs))

    def test_obs_norm_is_saved_with_the_weights(self, tmp_path):
        """Because an agent evaluated under a different normaliser is a different agent."""
        agent = AGENTS["ppo"](make_env(), tiny_cfg(norm_obs=True), seed=0)
        agent.learn(progress_every=0)
        agent.save(tmp_path / "a.pt")
        state = torch.load(tmp_path / "a.pt", weights_only=False)["obs_norm"]
        assert state["count"] > 1.0
        assert not np.allclose(state["mean"], 0.0)

    def test_penalty_enters_the_shaped_reward(self):
        """Fixed-penalty PPO maximises r - penalty*c, and nothing else."""
        agent = AGENTS["ppo"](make_env(), tiny_cfg(penalty=7.0), seed=0)
        buf, _, _ = agent.collect(agent.envs.reset())
        buf["costs"] = np.ones_like(buf["costs"])
        _, extras = agent.advantages(buf)
        expected = float((buf["rewards"] - 7.0 * buf["costs"]).mean())
        assert extras["shaped_reward_mean"] == pytest.approx(expected)

    def test_partial_obs_mode_changes_the_input_width(self):
        full = AGENTS["ppo"](make_env("full"), tiny_cfg(), seed=0)
        partial = AGENTS["ppo"](make_env("partial"), tiny_cfg(), seed=0)
        assert partial.obs_dim < full.obs_dim
        partial.learn(progress_every=0)  # and still trains


# ---------------------------------------------------------- the multiplier
class TestLagrangian:
    def cfg(self, **kw):
        return tiny_cfg(**{"cost_limit": 1.0, "lambda_lr": 0.5, **kw})

    def agent(self, **kw):
        return PPOLagrangian(make_env(), self.cfg(**kw), seed=0)

    def test_cost_above_the_limit_raises_lambda(self):
        agent = self.agent()
        agent.update_lambda([3.0, 3.0])  # mean cost 3, limit 1, error +2
        assert agent.lam == pytest.approx(1.0)  # 0 + 0.5*2

    def test_cost_below_the_limit_leaves_lambda_at_zero(self):
        """The multiplier is a one-sided price: no reward for being over-safe."""
        agent = self.agent()
        for _ in range(5):
            agent.update_lambda([0.0])
        assert agent.lam == 0.0

    def test_lambda_decays_once_the_constraint_is_met(self):
        agent = self.agent()
        agent.update_lambda([5.0])
        raised = agent.lam
        assert raised > 0
        for _ in range(3):
            agent.update_lambda([0.0])
        assert agent.lam < raised

    def test_lambda_is_capped(self):
        agent = self.agent(lambda_max=2.0)
        for _ in range(50):
            agent.update_lambda([100.0])
        assert agent.lam == pytest.approx(2.0)

    def test_lambda_never_goes_negative(self):
        agent = self.agent()
        for _ in range(100):
            agent.update_lambda([0.0])
        assert agent.lam >= 0.0

    def test_no_completed_episodes_leaves_lambda_alone(self):
        agent = self.agent()
        agent.update_lambda([4.0])
        held = agent.lam
        agent.update_lambda([])
        assert agent.lam == held

    def test_proportional_term_reacts_faster_than_the_integral_alone(self):
        plain = self.agent()
        pid = PPOLagrangian(make_env(), self.cfg(extra={"lambda_kp": 1.0}), seed=0)
        plain.update_lambda([3.0])
        pid.update_lambda([3.0])
        assert pid.lam > plain.lam

    def test_default_agent_is_the_plain_lagrangian_method(self):
        agent = self.agent()
        assert agent.lambda_kp == 0.0 and agent.lambda_kd == 0.0

    def test_lambda_mixes_the_two_advantages(self):
        """A = (A_r - lambda*A_c)/(1+lambda): at lambda=0 the cost is ignored."""
        agent = self.agent()
        buf, _, _ = agent.collect(agent.envs.reset())
        buf["costs"] = np.ones_like(buf["costs"])  # a constant cost has no advantage
        agent.lam = 0.0
        adv_free, _ = agent.advantages(buf)
        agent.lam = 10.0
        adv_priced, _ = agent.advantages(buf)
        assert not np.allclose(adv_free, adv_priced)
        # Scaling by 1/(1+lambda) keeps the priced advantage from blowing up.
        assert np.abs(adv_priced).max() < np.abs(adv_free).max() * 5

    def test_cost_returns_are_produced_for_the_cost_critic(self):
        agent = self.agent()
        buf, _, _ = agent.collect(agent.envs.reset())
        _, extras = agent.advantages(buf)
        assert "cost_returns" in extras
        assert extras["cost_returns"].shape == buf["costs"].shape

    def test_lambda_is_logged_every_update(self, tmp_path):
        agent = self.agent()
        rows = agent.learn(progress_every=0)
        assert all("lambda" in row for row in rows)
        assert all(row["cost_limit"] == 1.0 for row in rows)

    def test_multiplier_moves_before_the_policy_does(self):
        """Otherwise the correction is applied one rollout late."""
        agent = self.agent()
        seen = {}
        original = agent.advantages

        def spy(buf):
            seen["lambda_at_policy_step"] = agent.lam
            return original(buf)

        agent.advantages = spy
        buf, _, _ = agent.collect(agent.envs.reset())
        agent._ep_window = [{"cost": 9.0}]
        agent.update(buf)
        assert seen["lambda_at_policy_step"] > 0.0


def test_plain_ppo_has_no_cost_critic_and_no_multiplier():
    """The two agents must differ in exactly the documented ways."""
    assert PPO.uses_cost_critic is False
    assert PPOLagrangian.uses_cost_critic is True
    agent = AGENTS["ppo"](make_env(), tiny_cfg(), seed=0)
    assert agent.net.cost_critic is None
    assert agent.extra_log() == {}

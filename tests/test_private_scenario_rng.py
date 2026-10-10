"""
Regression test: competitor RNG use must not change the scenario stream.

``CompetitionEnv`` used to draw its scenarios from ``self.np_random``, the RNG
gymnasium also hands to subclasses and wrappers. Any draw made by an MDP hook
(e.g. observation noise in ``_get_obs``) or by a stock wrapper (e.g.
``gymnasium.wrappers.StickyAction``) shifted that stream, so from the second
episode on such a submission was scored on different scenarios than the
official "first 1000 episodes of seed 42". Scenarios now come from a private
RNG that only ``reset()`` touches. It is seeded exactly like ``np_random``, so
the stock stream itself is unchanged.

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_private_scenario_rng.py
    python -m pytest tests/test_private_scenario_rng.py
"""

import numpy as np
import gymnasium as gym

from bluesky_gym.envs.competition_env import CompetitionEnv

N_EPISODES = 3
STEPS_PER_EPISODE = 5   # scenarios are drawn in reset(); a few steps let hooks/wrappers draw

# Agent start of the first 3 seed-42 episodes, measured before this fix (commit 0093001).
SEED42_AGENT_STARTS = [(51.3133, 3.6112), (51.2122, 2.4649), (52.3414, 5.1717)]


class NoisyObsEnv(CompetitionEnv):
    """An allowed hook override: observation noise drawn from self.np_random."""

    def _get_obs(self, ac_id):
        obs = super()._get_obs(ac_id)
        obs["airspeed"] = obs["airspeed"] + 0.01 * self.np_random.standard_normal(1)
        return obs


def _scenario_stream(env, seed=42):
    """Scenarios of the first N_EPISODES episodes: reset(seed), then unseeded resets."""
    scenarios = []
    for ep in range(N_EPISODES):
        env.reset(seed=seed if ep == 0 else None)
        scenarios.append(env.unwrapped.scenario)
        for _ in range(STEPS_PER_EPISODE):
            env.step(np.zeros(2))
    env.close()
    return scenarios


def _agent_starts(scenarios):
    return [tuple(round(float(x), 4) for x in s.agents[0].start) for s in scenarios]


def test_stock_stream_unchanged():
    env = CompetitionEnv()
    generator = env.scenario_generator
    stock = _scenario_stream(env)

    rng = np.random.default_rng(42)
    expected = [generator.generate(rng) for _ in range(N_EPISODES)]
    assert stock == expected
    assert _agent_starts(stock) == SEED42_AGENT_STARTS


def test_hook_and_wrapper_draws_do_not_change_scenarios():
    stock = _scenario_stream(CompetitionEnv())
    noisy_obs = _scenario_stream(NoisyObsEnv())
    sticky = _scenario_stream(gym.wrappers.StickyAction(CompetitionEnv(),
                                                        repeat_action_probability=0.25))
    assert noisy_obs == stock, f"{_agent_starts(noisy_obs)} != {_agent_starts(stock)}"
    assert sticky == stock, f"{_agent_starts(sticky)} != {_agent_starts(stock)}"


def test_np_random_still_seeded_by_reset():
    """reset(seed) still seeds np_random for competitors, and scenario
    generation no longer consumes it."""
    env = CompetitionEnv()
    env.reset(seed=123)
    draws = env.np_random.random(3)
    env.close()
    assert np.array_equal(draws, np.random.default_rng(123).random(3))


if __name__ == "__main__":
    test_stock_stream_unchanged()
    test_hook_and_wrapper_draws_do_not_change_scenarios()
    test_np_random_still_seeded_by_reset()
    print("PASS")

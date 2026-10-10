"""
Regression test: the simulated-time budget must not depend on action_frequency.

Both competition envs ran the full ``action_frequency`` sim seconds in every
step and only checked the time limit afterwards. A timed-out episode therefore
ran past ``episode_time_limit`` whenever ``action_frequency`` does not divide
it (3003 s at action_frequency=7, 3008 s at 64), although COMPETITION.md
guarantees that ``action_frequency`` does not change the time budget. The sim
loop now stops as soon as the budget is spent.

The policy keeps turning ([1, 0]), so no aircraft reaches its goal and every
episode ends by truncation.

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_time_budget.py
    python -m pytest tests/test_time_budget.py
"""

import numpy as np
import bluesky as bs

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition_v0 import CompetitionZooEnv

TURN = np.array([1.0, 0.0])   # constant turn: never reaches the goal


def test_sa_time_budget_independent_of_action_frequency():
    for action_frequency in (10, 7, 64):   # 10 divides 3000 (default), 7 and 64 do not
        env = CompetitionEnv(action_frequency=action_frequency)
        env.reset(seed=1)
        t0 = bs.sim.simt
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(TURN)
        sim_seconds = bs.sim.simt - t0
        env.close()

        assert truncated and not terminated
        assert info["flight_time"] == 3000.0, (
            f"action_frequency={action_frequency}: flight_time {info['flight_time']}")
        assert sim_seconds == 3000.0


def test_ma_time_budget_independent_of_action_frequency():
    # a short budget keeps the test fast: 100 s is not a multiple of 7
    env = CompetitionZooEnv(action_frequency=7, episode_time_limit=100.0)
    env.reset(seed=1)
    while env.agents:
        _, _, terminations, truncations, infos = env.step({agent: TURN for agent in env.agents})
    env.close()

    assert env.sim_time == 100.0, f"sim_time {env.sim_time}"
    assert len(infos) == env.n_agents
    assert all(truncations.values()) and not any(terminations.values())
    assert all(info["flight_time"] == 100.0 for info in infos.values())


if __name__ == "__main__":
    test_sa_time_budget_independent_of_action_frequency()
    test_ma_time_budget_independent_of_action_frequency()
    print("PASS")

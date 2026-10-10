"""
Regression tests for ``scripts/record_competition_gifs.py``.

1. A clip covers the whole episode, whatever ``action_frequency`` is. The env
   truncates itself on simulated time (``episode_time_limit``), so the recorder
   must not add a cap counted in decisions (it used to stop after 400 env
   steps, i.e. after 400 s with ``action_frequency=1``). The env is injected
   through ``gymnasium.make`` here, so this check does not depend on test 2.
2. The recorder builds its env through the ``make_env`` edit point, so a
   competitor's env subclass / wrapper stack is what gets recorded.

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_record_competition_gifs.py
    python -m pytest tests/test_record_competition_gifs.py
"""

from unittest import mock

import gymnasium as gym

from scripts import record_competition_gifs as rgif   # sets the headless SDL driver
from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition_v0 import CompetitionZooEnv


class _MarkedEnv(CompetitionEnv):
    """Stand-in for a competitor's env subclass; counts its rendered frames."""
    n_frames = 0

    def render(self):
        self.n_frames += 1
        return super().render()


class _MarkedZooEnv(CompetitionZooEnv):
    """Stand-in for a competitor's multi-agent env; counts its rendered frames."""
    n_frames = 0

    def render(self):
        self.n_frames += 1
        return super().render()


def test_clip_is_not_capped_at_400_decisions():
    # One decision per simulated second and a 600 s budget: the clip must run
    # until the env ends the episode (goal or 600 s), one frame per second.
    # No goal is reachable within 400 s, so a 400-decision cap would bind.
    make, made = gym.make, []

    def make_af1(env_id, **kwargs):
        if env_id == "CompetitionEnv-v0":
            kwargs.update(action_frequency=1, episode_time_limit=600.0)
        made.append(make(env_id, **kwargs))
        return made[-1]

    with mock.patch.object(gym, "make", make_af1):
        frames = rgif.rollout_single_agent(seed=1)
    env = made[-1].unwrapped
    m = env.metrics[env.agent]
    assert env.action_frequency == 1 and env.episode_time_limit == 600.0
    assert m["waypoint_reached"] == 1 or m["flight_time"] >= 600.0, (
        f"clip stopped after {len(frames) - 1} decisions = {m['flight_time']:.0f} s, "
        f"before the 600 s episode ended")
    assert len(frames) == 1 + m["flight_time"] > 401, (len(frames), m["flight_time"])


def test_rollouts_use_make_env():
    assert hasattr(rgif, "make_env"), "record_competition_gifs has no make_env edit point"
    built = []

    def make_env(kind, render_mode="rgb_array", n_agents=10):
        if kind == "sa":
            env = _MarkedEnv(render_mode=render_mode, episode_time_limit=300.0)
        else:
            env = _MarkedZooEnv(render_mode=render_mode, n_agents=n_agents,
                                episode_time_limit=300.0)
        built.append((kind, n_agents, env))
        return env

    with mock.patch.object(rgif, "make_env", make_env):
        sa_frames = rgif.rollout_single_agent(seed=1)
        ma_frames = rgif.rollout_multi_agent(seed=1, n_agents=2)

    assert [(k, n) for k, n, _ in built] == [("sa", 10), ("ma", 2)], built
    sa_env, ma_env = built[0][2], built[1][2]
    assert sa_env.n_frames == len(sa_frames) == 31, (sa_env.n_frames, len(sa_frames))
    assert ma_env.n_frames == len(ma_frames) > 1, (ma_env.n_frames, len(ma_frames))
    assert ma_frames[0].shape == (ma_env.window_height, ma_env.window_width, 3)


if __name__ == "__main__":
    import sys
    failed = 0
    for name, test in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            test()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)

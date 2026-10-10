"""
Regression test for calling ``reset()`` twice without a step in between.

``CompetitionEnv.reset`` queues each intruder's route as BlueSky ``ADDWPT``
stack commands. ``bs.traf.reset()`` clears the aircraft but not the command
stack, so commands still pending when ``reset()`` is called again (an SB3
VecEnv auto-reset followed by ``learn()`` / ``evaluate_policy()``, or a manual
``reset(); reset(seed=s)``) used to run against the new episode's aircraft:
every intruder got the stale waypoints followed by its own route. ``reset()``
now runs its queued commands itself, so a double reset must give exactly the
episode of a single reset, while the normal single-reset flow (and the
``DT 1;FF`` queued in ``__init__``) must stay unchanged.

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_reset_pending_stack.py
    python -m pytest tests/test_reset_pending_stack.py
"""

import math

import numpy as np
import bluesky as bs
from bluesky.stack.stackbase import Stack

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition_v0 import CompetitionZooEnv

# Final info of the first 3 episodes of the official seed-42 stream with the
# neutral [0, 0] action, measured before this fix (commit 0093001).
SEED42_NEUTRAL = [
    dict(total_reward=-99.0059382501526, flight_time=890.0, waypoint_reached=1,
         intrusion_events=0, intrusion_time=0.0, sector_exit_events=0, time_outside_sector=0.0,
         restricted_area_events=1, time_in_restricted_area=100.0),
    dict(total_reward=-82.00219003137397, flight_time=1059.0, waypoint_reached=1,
         intrusion_events=2, intrusion_time=83.0, sector_exit_events=0, time_outside_sector=0.0,
         restricted_area_events=0, time_in_restricted_area=0.0),
    dict(total_reward=-314.0276795922376, flight_time=1059.0, waypoint_reached=1,
         intrusion_events=3, intrusion_time=166.0, sector_exit_events=0, time_outside_sector=0.0,
         restricted_area_events=1, time_in_restricted_area=149.0),
]


def _route_lengths(env):
    return [bs.traf.ap.route[bs.traf.id2idx(ac_id)].nwp for ac_id in env.intruder_ids]


def _run_episode(env):
    """Fly the neutral action to the end of the episode. Returns the intruder
    route lengths after the first step (queued commands have run by then, with
    or without the fix) and the final info."""
    _, _, term, trunc, info = env.step(np.zeros(2))
    route_lengths = _route_lengths(env)
    while not (term or trunc):
        _, _, term, trunc, info = env.step(np.zeros(2))
    return route_lengths, info


def test_double_reset_gives_single_reset_episode():
    env = CompetitionEnv()
    env.reset(seed=1)
    expected_lengths = [len(route.waypoints) - 1 for route in env.scenario.intruder_routes]
    single_lengths, single_info = _run_episode(env)

    env.reset(seed=1000)   # e.g. a VecEnv auto-reset ...
    env.reset(seed=1)      # ... followed by another reset before any step
    double_lengths, double_info = _run_episode(env)
    env.close()

    assert single_lengths == expected_lengths
    assert double_lengths == expected_lengths, (
        f"intruder route lengths {double_lengths}, expected {expected_lengths}")
    assert double_info == single_info, f"{double_info} != {single_info}"


def test_single_reset_flow_unchanged():
    env = CompetitionEnv()
    for ep, expected in enumerate(SEED42_NEUTRAL):
        env.reset(seed=42 if ep == 0 else None)
        _, info = _run_episode(env)
        expected = dict(expected)
        assert math.isclose(info.pop("total_reward"), expected.pop("total_reward"), rel_tol=1e-9)
        assert info == expected, f"episode {ep}: {info} != {expected}"
    env.close()


def test_reset_leaves_no_pending_commands():
    """Both envs: nothing is left on the stack after reset(), and the 'DT 1'
    queued in __init__ is still applied (one env step = action_frequency s)."""
    for env_cls in (CompetitionEnv, CompetitionZooEnv):
        env = env_cls()
        env.reset(seed=1)
        assert not Stack.cmdstack, f"{env_cls.__name__}: pending after reset: {Stack.cmdstack[:3]}"
        t0 = bs.sim.simt
        if env_cls is CompetitionEnv:
            env.step(np.zeros(2))
        else:
            env.step({agent: np.zeros(2) for agent in env.agents})
        assert bs.sim.simt - t0 == env.action_frequency * env.sim_dt
        env.close()


if __name__ == "__main__":
    test_double_reset_gives_single_reset_episode()
    test_single_reset_flow_unchanged()
    test_reset_leaves_no_pending_commands()
    print("PASS")

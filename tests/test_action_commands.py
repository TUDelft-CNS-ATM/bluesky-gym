"""
Regression tests for the BlueSky commands stacked by ``core.actions``.

``HeadingAction`` stacks ``HDG <id> <heading>``. Formatting a float with
``str()`` gives scientific notation for |x| < 1e-4 (e.g. ``-1e-05`` when the
commanded heading is within 1e-4 deg of north), and BlueSky's ``txt2hdg``
rejects that text: the command is dropped and the autopilot keeps its old
target heading. The commands must use fixed-point text, and every other value
must still give exactly the same autopilot target as before.

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_action_commands.py
    python -m pytest tests/test_action_commands.py
"""

import bluesky as bs
from bluesky.tools.aero import kts

from core.actions import MpS2Kt, _bound_angle
from bluesky_gym.envs.competition_env import CompetitionEnv

AC = "KL001"


def _env():
    env = CompetitionEnv()
    env.reset(seed=1)
    return env, bs.traf.id2idx(AC)


def _heading_target(env, i, hdg, action):
    bs.stack.stack(f"HDG {AC} 123")          # known old target
    bs.sim.step()
    assert bs.traf.ap.trk[i] == 123.0
    bs.traf.hdg[i] = hdg
    env.heading_action.execute(AC, action)
    bs.sim.step()                            # processes the stacked command
    return bs.traf.ap.trk[i]


def test_heading_command_near_north_is_applied():
    env, i = _env()
    for hdg, action in [(359.99999, 0.0), (0.00001, 0.0), (45.000001, -1.0)]:
        trk = _heading_target(env, i, hdg, action)
        expected = _bound_angle(hdg + action * env.heading_action.d_heading)
        assert abs(expected) < 1e-4, "test setup: target must be within 1e-4 deg of north"
        assert trk == expected, (
            f"hdg={hdg}, action={action}: autopilot target {trk}, expected {expected} "
            f"(HDG command dropped, old target 123 kept)")
    env.close()


def test_normal_commands_give_unchanged_targets():
    # Same autopilot targets as plain str() formatting, bit for bit.
    env, i = _env()
    for hdg in [0.0, 10.5, 123.456789012345, 179.99, 180.0, 270.123456789, 359.9]:
        for action in [-1.0, -0.37, 0.0, 0.5, 1.0]:
            trk = _heading_target(env, i, hdg, action)
            assert trk == float(str(_bound_angle(hdg + action * env.heading_action.d_heading)))
    for action in [-1.0, -0.37, 0.0, 0.5, 1.0]:
        cas = bs.traf.cas[i]
        env.speed_action.execute(AC, action)
        new_spd = (cas + action * env.speed_action.d_speed / MpS2Kt) * MpS2Kt
        bs.sim.step()
        assert bs.traf.selspd[i] == float(str(new_spd)) * kts, (bs.traf.selspd[i], new_spd)
    env.close()


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

"""
Regression test for ``restricted_area_events`` with overlapping obstacles.

Obstacles may overlap (the generator places them independently), and an
overlapping pair is one contiguous grey region. ``restricted_area_events``
must count entries into restricted airspace, i.e. into the union of all
obstacles, consistent with ``time_in_restricted_area`` (seconds inside the
union): flying from obstacle A straight into an overlapping obstacle B is one
entry, not two. Two separate obstacles are still two entries.

Each case flies a hand-built scenario with the neutral ``[0, 0]`` action and
``action_frequency=1``, so the test can sample the aircraft position after
every 1-second sim step (the metric sampling interval) and compare the scored
metrics with an independent ground truth (rising edges of / seconds inside the
union of the obstacle rectangles).

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_restricted_area_events.py
    python -m pytest tests/test_restricted_area_events.py
"""

import numpy as np
import bluesky as bs

from core.scenario import Scenario, Obstacle, AgentSpec, _nm_to_latlon, _bearing_nm
from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition_v0 import CompetitionZooEnv

CENTER = (52.0, 4.0)
SECTOR_NM = [(100, -100), (100, 100), (-100, 100), (-100, -100)]   # (x=north, y=east)

# Obstacle rectangles in the flat NM frame: (x_min, x_max, y_min, y_max).
# Agents fly due east along a line x = const, from y = -60 to y = +60.
OVERLAPPING = [(-10, 10, -30, -10), (-10, 10, -15, 5)]    # overlap y in [-15, -10]
SEPARATE = [(-10, 10, -30, -15), (-10, 10, 0, 15)]        # 15 NM gap


def _ll(p):
    return _nm_to_latlon(CENTER, p)


def _shift(rects, dx):
    return [(x0 + dx, x1 + dx, y0, y1) for x0, x1, y0, y1 in rects]


def _obstacle(rect):
    x0, x1, y0, y1 = rect
    return Obstacle(vertices=[_ll(p) for p in [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]])


def _agent(ac_id, x):
    start, goal = (x, -60.0), (x, 60.0)
    return AgentSpec(ac_id=ac_id, start=_ll(start), goal=_ll(goal),
                     heading=_bearing_nm(start, goal), speed=150.0)


def _scenario(rects, agents):
    return Scenario(center=CENTER, sector=[_ll(p) for p in SECTOR_NM],
                    obstacles=[_obstacle(r) for r in rects], agents=agents)


class _UnionTruth:
    """Ground truth for one aircraft: rising edges of, and seconds inside,
    the union of axis-aligned lat/lon rectangles (the NM->lat/lon map is
    affine, so the NM rectangles stay rectangles)."""

    def __init__(self, rects):
        self.boxes = []
        for x0, x1, y0, y1 in rects:
            (lat0, lon0), (lat1, lon1) = _ll((x0, y0)), _ll((x1, y1))
            self.boxes.append((lat0, lat1, lon0, lon1))
        self.inside = False
        self.entries = 0
        self.seconds = 0

    def sample(self, ac_id, dt=1):
        idx = bs.traf.id2idx(ac_id)
        lat, lon = bs.traf.lat[idx], bs.traf.lon[idx]
        inside = any(a <= lat <= b and c <= lon <= d for a, b, c, d in self.boxes)
        self.entries += int(inside and not self.inside)
        self.seconds += dt if inside else 0
        self.inside = inside


def _fly_sa(rects):
    env = CompetitionEnv(scenario=_scenario(rects, [_agent("KL001", 0.0)]),
                         action_frequency=1)
    env.reset()
    truth = _UnionTruth(rects)
    truth.sample("KL001", dt=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        truth.sample("KL001")
    return info, truth


def _fly_ma(agent_rects):
    """agent_rects: {ac_id: (x of its eastbound line, obstacle rects on it)}."""
    agents = [_agent(a, x) for a, (x, _) in agent_rects.items()]
    rects = [r for _, rs in agent_rects.values() for r in rs]
    env = CompetitionZooEnv(scenario=_scenario(rects, agents), action_frequency=1)
    env.reset()
    # each agent's truth uses all obstacles, as the env does
    truth = {a: _UnionTruth(rects) for a in agent_rects}
    for a in agent_rects:
        truth[a].sample(a, dt=0)
    final = {}
    while env.agents:
        _, _, terms, truncs, infos = env.step({a: np.zeros(2) for a in env.agents})
        for a in infos:
            if terms[a] or truncs[a]:
                final[a] = infos[a]
        # an agent that reached its goal this step is already deleted; its last
        # sample is at the goal, outside every obstacle, so skipping it is exact
        for a in env.agents:
            truth[a].sample(a)
    return final, truth


def _check(info, truth, expected_entries):
    assert truth.entries == expected_entries, "test geometry: path must cross the obstacles"
    assert info["waypoint_reached"] == 1
    assert info["intrusion_events"] == 0 and info["sector_exit_events"] == 0
    assert info["restricted_area_events"] == truth.entries, (
        f"restricted_area_events={info['restricted_area_events']}, "
        f"entries into restricted airspace={truth.entries}")
    assert truth.seconds > 0
    assert info["time_in_restricted_area"] == truth.seconds, (
        f"time_in_restricted_area={info['time_in_restricted_area']}, "
        f"seconds inside restricted airspace={truth.seconds}")


def test_sa_overlapping_obstacles_count_one_entry():
    info, truth = _fly_sa(OVERLAPPING)
    _check(info, truth, expected_entries=1)


def test_sa_separate_obstacles_count_two_entries():
    info, truth = _fly_sa(SEPARATE)
    _check(info, truth, expected_entries=2)


def test_ma_overlapping_and_separate_obstacles():
    # KL001 crosses an overlapping pair, KL002 (60 NM south, no conflict) a separate pair
    final, truth = _fly_ma({"KL001": (30.0, _shift(OVERLAPPING, 30.0)),
                            "KL002": (-30.0, _shift(SEPARATE, -30.0))})
    assert set(final) == {"KL001", "KL002"}
    _check(final["KL001"], truth["KL001"], expected_entries=1)
    _check(final["KL002"], truth["KL002"], expected_entries=2)


if __name__ == "__main__":
    import sys
    failed = 0
    for name, test in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            test()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print("FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)

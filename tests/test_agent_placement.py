"""
Regression tests for agent start/goal placement in ``core.scenario`` (issue #72).

When the strict placement (10 NM obstacle clearance) fails, the relaxed
placement used to drop the obstacle test entirely, so starts and goals could
land inside obstacles. Over the official evaluation streams (seed 42, first
1000 single-agent and multi-agent scenarios) no start or goal may lie inside an
obstacle and no goal's 5 km capture disc may lie entirely inside obstacles. A
crowded configuration exercises the relaxed placement on most scenarios.

Run from the repo root with ``pytest tests/test_agent_placement.py`` or
``python tests/test_agent_placement.py``.
"""

import math
import sys

from gymnasium.utils import seeding
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import core
from core.scenario import ScenarioGenerator

SEED = 42                  # official evaluation seed
N_SCENARIOS = 1000         # official episode count
CAPTURE_NM = 5.0 / 1.852   # env waypoint-capture radius (5 km)


def _stream(gen, n, seed=SEED):
    """reset(seed) once, then unseeded resets: the env's scenario sequence."""
    rng, _ = seeding.np_random(seed)
    return [gen.generate(rng) for _ in range(n)]


def _to_nm(center, p):
    """(lat, lon) -> the generator's flat (north, east) NM frame."""
    return ((p[0] - center[0]) * 60.0,
            (p[1] - center[1]) * 60.0 * math.cos(math.radians(center[0])))


def _placement_violations(scenarios):
    """(scenario, agent, what) for every start or goal inside an obstacle and every
    capture disc entirely inside obstacles.

    Inside tests use lat/lon polygons, like BlueSky's areafilter; the capture
    disc is checked in the flat NM frame.
    """
    bad = []
    for ep, scen in enumerate(scenarios):
        polys = [Polygon(o.vertices) for o in scen.obstacles]
        blocked_nm = unary_union([Polygon([_to_nm(scen.center, v) for v in o.vertices])
                                  for o in scen.obstacles])
        for agent in scen.agents:
            for what, p in (("start", agent.start), ("goal", agent.goal)):
                if any(poly.contains(Point(p)) for poly in polys):
                    bad.append((ep, agent.ac_id, what))
            disc = Point(_to_nm(scen.center, agent.goal)).buffer(CAPTURE_NM)
            if disc.difference(blocked_nm).area < 1e-9:
                bad.append((ep, agent.ac_id, "capture disc"))
    return bad


def _check(scenarios):
    bad = _placement_violations(scenarios)
    assert not bad, (f"{len(bad)} placement violations in {len({b[0] for b in bad})} scenarios, "
                     f"e.g. {bad[:6]}")


def test_single_agent_stream():
    _check(_stream(ScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0), N_SCENARIOS))


def test_multi_agent_stream():
    _check(_stream(ScenarioGenerator(n_obstacles=5, n_intruders=0, n_agents=10, agent_speed=150.0),
                   N_SCENARIOS))


def test_crowded_obstacles_use_relaxed_placement_safely():
    """Large obstacles make the strict placement fail on most scenarios."""
    crowded = dict(n_obstacles=12, n_intruders=0, obstacle_area_range=(1500, 3000))
    _check(_stream(ScenarioGenerator(**crowded), 100, seed=0))
    # an unreachable relaxed clearance forces the last pass ("outside every obstacle")
    _check(_stream(ScenarioGenerator(relaxed_obstacle_clearance_nm=100.0, **crowded), 20, seed=0))


if __name__ == "__main__":
    print(f"testing {core.__file__}")
    failed = 0
    for test in (test_single_agent_stream,
                 test_multi_agent_stream,
                 test_crowded_obstacles_use_relaxed_placement_safely):
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:   # report every test, then fail
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)

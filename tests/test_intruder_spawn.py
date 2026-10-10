"""
Regression tests for intruder spawning in ``core.scenario`` (issue #70).

Over the official single-agent evaluation stream (seed 42, first 1000
scenarios) no intruder may spawn within the spawn clearance of the agent's
start, so no episode begins in loss of separation, and no route speed may
exceed the A320 maximum CAS that BlueSky enforces. One check runs the real
environment (in a subprocess: one BlueSky simulator per process).

Run from the repo root with ``pytest tests/test_intruder_spawn.py`` or
``python tests/test_intruder_spawn.py`` (the env check runs in the current
directory with the same interpreter).
"""

import functools
import math
import subprocess
import sys

from gymnasium.utils import seeding

import core
from core.scenario import ScenarioGenerator
from core.tools import kwikqdrdist

SEED = 42                   # official evaluation seed
N_SCENARIOS = 1000          # official episode count
SPAWN_CLEARANCE_NM = 10.0   # ScenarioGenerator default
INTRUSION_NM = 5.0          # separation minimum scored by the env
A320_VMAX = 163.0           # m/s, BlueSky OpenAP A320 maximum CAS at 350 m
N_ENV_EPISODES = 25         # real-env check (episodes 2, 8 and 23 failed before the fix)


@functools.lru_cache(maxsize=None)
def _sa_stream():
    """The evaluation scenarios: reset(seed=42) once, then unseeded resets."""
    rng, _ = seeding.np_random(SEED)
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0)
    return [gen.generate(rng) for _ in range(N_SCENARIOS)]


def _to_nm(center, p):
    """(lat, lon) -> the generator's flat (north, east) NM frame."""
    return ((p[0] - center[0]) * 60.0,
            (p[1] - center[1]) * 60.0 * math.cos(math.radians(center[0])))


def test_intruders_spawn_clear_of_agent_start():
    bad = []
    for ep, scen in enumerate(_sa_stream()):
        start = scen.agents[0].start
        s_nm = _to_nm(scen.center, start)
        for route in scen.intruder_routes:
            spawn = route.waypoints[0]
            p_nm = _to_nm(scen.center, spawn)
            d_frame = math.hypot(p_nm[0] - s_nm[0], p_nm[1] - s_nm[1])
            d_scored = float(kwikqdrdist(*start, *spawn)[1])
            if d_frame < SPAWN_CLEARANCE_NM - 1e-6 or d_scored <= INTRUSION_NM:
                bad.append((ep, route.ac_id, round(d_scored, 2)))
    assert not bad, (f"{len(bad)} intruders in {len({b[0] for b in bad})} scenarios spawn within "
                     f"{SPAWN_CLEARANCE_NM} NM of the agent start, e.g. {bad[:5]}")


def test_intruder_speeds_within_a320_envelope():
    speeds = [r.speed for scen in _sa_stream() for r in scen.intruder_routes]
    fast = [v for v in speeds if v > A320_VMAX]
    assert not fast, (f"{len(fast)}/{len(speeds)} intruder routes are faster than the "
                      f"A320 vmax {A320_VMAX} m/s (max {max(fast):.1f})")


def test_spawn_remap_is_uniform_on_clear_window():
    """A uniform draw remapped out of the clearance is uniform on the clear part."""
    from core.scenario import _remap_outside

    width, lo, hi, n = 0.4, 0.1, 0.25, 4000
    grid = [(k + 0.5) / n * width for k in range(n)]           # uniform on [0, width]
    out = sorted(_remap_outside(u, lo, hi, width) for u in grid)
    assert all(not lo < u < hi for u in out)
    # equally spaced over [0, lo] + [hi, width]: compare with that set's quantiles
    free = width - (hi - lo)
    for k, u in enumerate(out):
        q = (k + 0.5) / n * free
        expected = q if q < lo else hi + (q - lo)
        assert abs(u - expected) < 3 * free / n


ENV_CHECK = r"""
import sys
import gymnasium as gym
import bluesky_gym
bluesky_gym.register_envs()
env = gym.make("CompetitionEnv-v0")
bad = []
for ep in range(int(sys.argv[1])):
    _, info = env.reset(seed=42 if ep == 0 else None)
    if info["intrusion_events"]:
        bad.append(ep)
env.close()
print("EPISODES_WITH_INTRUSION_AT_RESET", bad)
"""


def test_env_episodes_start_without_intrusion():
    proc = subprocess.run([sys.executable, "-c", ENV_CHECK, str(N_ENV_EPISODES)],
                          capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("EPISODES_WITH_INTRUSION_AT_RESET")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert lines[-1] == "EPISODES_WITH_INTRUSION_AT_RESET []", (
        f"official episodes with intrusion_events > 0 straight after reset(): {lines[-1].split(' ', 1)[1]}")


if __name__ == "__main__":
    print(f"testing {core.__file__}")
    failed = 0
    for test in (test_intruders_spawn_clear_of_agent_start,
                 test_intruder_speeds_within_a320_envelope,
                 test_spawn_remap_is_uniform_on_clear_window,
                 test_env_episodes_start_without_intrusion):
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:   # report every test, then fail
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)

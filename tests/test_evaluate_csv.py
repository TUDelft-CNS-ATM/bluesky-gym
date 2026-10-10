"""
Regression tests for the evaluation harness (scripts/evaluate_competition.py).

* Every CSV row identifies its episode: ``episode``, ``agent_id`` (multi-agent)
  and ``scenario_hash`` (a short, stable hash of the episode's scenario), next
  to the original ``episode_index`` row number.
* The CSV is written before the summary is printed, and the console output is
  ASCII, so a non-UTF-8 console (e.g. a cp932 Windows code page with output
  redirected to a file) cannot lose a finished run.
* ``--env both --out X.csv`` writes ``X_sa.csv`` and ``X_ma.csv`` instead of
  silently ignoring ``--out``.

The first four tests run in-process without the simulator. The last two run
the harness end to end in subprocesses (2 single-agent episodes; then 1
single-agent and 1 multi-agent episode), from a temporary directory so nothing
is written into the repo.

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_evaluate_csv.py
    python -m pytest tests/test_evaluate_csv.py
"""

import contextlib
import copy
import csv
import functools
import importlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from core.scenario import ScenarioGenerator, agent_callsigns

HARNESS = importlib.util.find_spec("scripts.evaluate_competition").origin
REPO_ROOT = os.path.dirname(os.path.dirname(HARNESS))
HEX16 = re.compile(r"[0-9a-f]{16}")


def _harness():
    return importlib.import_module("scripts.evaluate_competition")


def _fake_records(ev, n=3):
    return [{**{k: float(i) for k in ev.METRIC_KEYS},
             "episode": i, "agent_id": "KL001", "scenario_hash": "0" * 16}
            for i in range(n)]


@contextlib.contextmanager
def _patched(module, **attrs):
    saved = {k: getattr(module, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def _main_out_paths(ev, argv, stdout=None):
    """Run main() with evaluate() stubbed; return the CSV paths it would write."""
    calls = []
    with _patched(ev, evaluate=lambda kind, n, n_agents, model, out: calls.append(out)), \
            _patched(sys, argv=["evaluate_competition", *argv]), \
            contextlib.redirect_stdout(stdout or io.StringIO()):
        ev.main()
    return calls


# ----------------------------------------------------------- in-process tests
def test_scenario_hash_stable_and_sensitive():
    ev = _harness()
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=10)
    a = gen.generate(np.random.default_rng(0))
    h = ev.scenario_hash(a)
    assert HEX16.fullmatch(h), h
    assert ev.scenario_hash(gen.generate(np.random.default_rng(0))) == h
    noisy = copy.deepcopy(a)   # float noise far below the 9-decimal rounding
    lat, lon = noisy.agents[0].start
    noisy.agents[0].start = (lat + 1e-12, lon)
    assert ev.scenario_hash(noisy) == h
    moved = copy.deepcopy(a)   # ~0.1 m shift of one obstacle vertex
    lat, lon = moved.obstacles[0].vertices[0]
    moved.obstacles[0].vertices[0] = (lat + 1e-6, lon)
    assert ev.scenario_hash(moved) != h
    assert ev.scenario_hash(gen.generate(np.random.default_rng(1))) != h


def test_csv_written_before_summary():
    ev = _harness()

    def broken_console(records, label):
        raise UnicodeEncodeError("cp932", "—", 0, 1, "simulated console error")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "run.csv")
        with _patched(ev, run_single_agent=lambda n, act: _fake_records(ev),
                      summarize=broken_console), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                ev.evaluate("sa", 3, 10, None, out)
            except UnicodeEncodeError:
                pass
        assert os.path.exists(out), "a console error while printing the summary lost the CSV"


def test_console_output_is_ascii():
    ev = _harness()
    console = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
    with contextlib.redirect_stdout(console):
        ev.summarize(_fake_records(ev), "Single-agent")
        ev.summarize(_fake_records(ev), "Multi-agent")
        _main_out_paths(ev, ["--env", "sa", "--episodes", "2"], console)   # NOTE line
        console.flush()
    text = console.buffer.getvalue().decode("ascii")
    assert "| Metric | Mean | Std |" in text and "+/-" in text and "NOTE:" in text


def test_both_out_derives_per_env_paths():
    ev = _harness()
    assert _main_out_paths(ev, ["--env", "both", "--out", "res/x.csv"]) == [
        "res/x_sa.csv", "res/x_ma.csv"]
    assert _main_out_paths(ev, ["--env", "sa", "--out", "res/x.csv"]) == ["res/x.csv"]
    assert _main_out_paths(ev, ["--env", "both"]) == [None, None]   # default names


# ------------------------------------------------- end-to-end harness runs
@functools.lru_cache(maxsize=None)
def _harness_run(args, encoding=None):
    """Run the harness once in a temp dir; return (returncode, stdout, stderr,
    {csv file name: (fieldnames, rows)})."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [REPO_ROOT, env.get("PYTHONPATH")] if p)
    if encoding:
        env["PYTHONIOENCODING"] = encoding
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.run([sys.executable, "-m", "scripts.evaluate_competition", *args],
                              cwd=d, env=env, capture_output=True, timeout=900)
        csvs = {}
        for name in sorted(os.listdir(d)):
            if name.endswith(".csv"):
                with open(os.path.join(d, name), newline="") as f:
                    reader = csv.DictReader(f)
                    csvs[name] = (reader.fieldnames, list(reader))
    stdout = proc.stdout.decode(encoding or "utf-8", errors="replace")
    return proc.returncode, stdout, proc.stderr.decode(errors="replace"), csvs


def _assert_ok(run):
    rc, _, stderr, _ = run
    assert rc == 0, f"harness exited {rc}:\n{stderr[-1500:]}"


def test_sa_run_on_cp932_console_writes_ids():
    ev = _harness()
    # stdout is a pipe, so PYTHONIOENCODING=cp932 reproduces a redirected
    # Japanese-locale Windows console
    run = _harness_run(("--env", "sa", "--episodes", "2", "--out", "run.csv"), "cp932")
    _assert_ok(run)
    _, stdout, _, csvs = run
    assert list(csvs) == ["run.csv"]
    fields, rows = csvs["run.csv"]
    assert fields == ["episode_index", "episode", "scenario_hash", *ev.METRIC_KEYS], fields
    assert [(r["episode_index"], r["episode"]) for r in rows] == [("0", "0"), ("1", "1")]
    hashes = [r["scenario_hash"] for r in rows]
    assert all(HEX16.fullmatch(h) for h in hashes), hashes
    assert hashes[0] != hashes[1], "different episodes must have different hashes"
    assert "| Metric | Mean | Std |" in stdout


def test_both_run_writes_per_env_csvs_with_ids():
    ev = _harness()
    run = _harness_run(("--env", "both", "--episodes", "1", "--out", "both.csv"))
    _assert_ok(run)
    csvs = run[3]
    assert list(csvs) == ["both_ma.csv", "both_sa.csv"], list(csvs)

    fields, rows = csvs["both_ma.csv"]
    assert fields == ["episode_index", "episode", "agent_id", "scenario_hash",
                      *ev.METRIC_KEYS], fields
    assert len(rows) == ev.N_AGENTS_MA
    assert {r["episode"] for r in rows} == {"0"}
    assert sorted(r["agent_id"] for r in rows) == sorted(agent_callsigns(ev.N_AGENTS_MA))
    ma_hashes = {r["scenario_hash"] for r in rows}
    assert len(ma_hashes) == 1 and HEX16.fullmatch(next(iter(ma_hashes)))

    # the same first seed-42 scenario hashes identically across runs (and
    # console encodings), and equals the hash of the generated scenario itself
    _, sa_rows = csvs["both_sa.csv"]
    first = _harness_run(("--env", "sa", "--episodes", "2", "--out", "run.csv"), "cp932")
    _assert_ok(first)
    assert sa_rows[0]["scenario_hash"] == first[3]["run.csv"][1][0]["scenario_hash"]
    expected = ScenarioGenerator(n_obstacles=5, n_intruders=10).generate(np.random.default_rng(42))
    assert sa_rows[0]["scenario_hash"] == ev.scenario_hash(expected)
    assert sa_rows[0]["scenario_hash"] not in ma_hashes


if __name__ == "__main__":
    failed = 0
    for name, test in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            test()
            print(f"PASS {name}")
        except Exception as e:   # report every test, also on import-level API gaps
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)

"""
bluesky_gym/experiment/compare_runs.py
----------------------------------------
Load and compare training evaluation curves from multiple runs.

Each run must have been trained with session.track_training_evals = True,
which writes training_evals.csv to the model save path.

Output
------
  Console  - per-run summary table (final checkpoint metrics)
  Console  - per-timestep comparison table (optional, --full)
  CSV      - merged comparison table → comparison_<timestamp>.csv

Usage
-----
  # Compare two specific runs
  python compare_runs.py --runs 20260401_120000 20260401_130000

  # Compare all runs (auto-discover)
  python compare_runs.py --all

  # Show the full per-timestep table in the console too
  python compare_runs.py --runs 20260401_120000 20260401_130000 --full

  # Save merged CSV to a custom location
  python compare_runs.py --all --out ./results/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from datetime import datetime
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_all_training_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    """Return [(run_id, csv_path), ...] for every training_evals.csv found."""
    pattern = os.path.join(base, "*/*/*/training_evals.csv")
    results = []
    for path in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(path))
        results.append((run_id, path))
    return results


def find_training_csv(run_id: str, base: str = "./experiments") -> str:
    """Locate training_evals.csv for a specific run_id."""
    pattern = os.path.join(base, f"*/*/models/{run_id}/training_evals.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No training_evals.csv found for run_id='{run_id}'.\n"
            f"Make sure the run used track_training_evals=true and has finished."
        )
    return matches[0]


def load_training_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "timestep":     int(row["timestep"]),
                "mean_reward":  float(row["mean_reward"]),
                "std_reward":   _safe_float(row.get("std_reward")),
                "success_rate": _safe_float(row.get("success_rate")),
            })
    return rows


def _safe_float(v: Optional[str]) -> float:
    if v is None or v.strip() in ("", "nan", "None"):
        return float("nan")
    return float(v)


# ---------------------------------------------------------------------------
# Summary stats per run
# ---------------------------------------------------------------------------

def run_summary(run_id: str, rows: list[dict]) -> dict:
    if not rows:
        return {"run_id": run_id, "n_evals": 0}

    rewards = [r["mean_reward"]  for r in rows]
    success = [r["success_rate"] for r in rows]
    final   = rows[-1]
    best_idx= int(np.argmax(rewards))

    return {
        "run_id":              run_id,
        "n_evals":             len(rows),
        "final_timestep":      final["timestep"],
        "final_mean_reward":   final["mean_reward"],
        "final_success_rate":  final["success_rate"],
        "best_mean_reward":    rewards[best_idx],
        "best_at_timestep":    rows[best_idx]["timestep"],
        "peak_success_rate":   float(np.nanmax(success)) if success else float("nan"),
        "mean_reward_overall": float(np.mean(rewards)),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)

def _fmt_pct(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v:.1%}"


def print_summary_table(summaries: list[dict]) -> None:
    cols = [
        ("run_id",             16, lambda v: str(v)),
        ("n_evals",             7, lambda v: str(v)),
        ("final_timestep",     14, lambda v: f"{v:,}"),
        ("final_mean_reward",  16, lambda v: _fmt(v)),
        ("final_success_rate", 18, lambda v: _fmt_pct(v)),
        ("best_mean_reward",   16, lambda v: _fmt(v)),
        ("best_at_timestep",   15, lambda v: f"{v:,}"),
        ("peak_success_rate",  17, lambda v: _fmt_pct(v)),
    ]
    col_w  = {name: max(w, len(name)) for name, w, _ in cols}
    sep    = "─" * (sum(col_w[n] + 2 for n, *_ in cols) + 1)
    header = "  ".join(f"{name:>{col_w[name]}}" for name, *_ in cols)

    print(f"\n{sep}")
    print("  TRAINING CURVE COMPARISON")
    print(sep)
    print(header)
    print(sep)
    for s in summaries:
        row = "  ".join(
            f"{fmt(s.get(name, float('nan'))):>{col_w[name]}}"
            for name, _, fmt in cols
        )
        print(row)
    print(sep)
    print()


def print_full_table(run_ids: list[str], all_rows: list[list[dict]]) -> None:
    all_timesteps = sorted({r["timestep"] for rows in all_rows for r in rows})
    indexed = [{r["timestep"]: r for r in rows} for rows in all_rows]

    col_w = 10
    header_parts = ["timestep".rjust(12)]
    for rid in run_ids:
        short = rid[-8:]
        header_parts += [
            f"{'rew_' + short:>{col_w}}",
            f"{'sr_'  + short:>{col_w}}",
        ]

    sep = "─" * (len("  ".join(header_parts)) + 2)
    print(sep)
    print("  FULL TRAINING CURVES (rew = mean_reward, sr = success_rate)")
    print(sep)
    print("  ".join(header_parts))
    print(sep)

    for ts in all_timesteps:
        parts = [f"{ts:>12,}"]
        for idx_map in indexed:
            row = idx_map.get(ts)
            if row:
                parts += [
                    f"{_fmt(row['mean_reward']):>{col_w}}",
                    f"{_fmt_pct(row['success_rate']):>{col_w}}",
                ]
            else:
                parts += [f"{'—':>{col_w}}", f"{'—':>{col_w}}"]
        print("  ".join(parts))
    print(sep)
    print()


def save_merged_csv(
    run_ids:  list[str],
    all_rows: list[list[dict]],
    path:     str,
) -> None:
    all_timesteps = sorted({r["timestep"] for rows in all_rows for r in rows})
    indexed = [{r["timestep"]: r for r in rows} for rows in all_rows]

    fieldnames = ["timestep"]
    for rid in run_ids:
        fieldnames += [
            f"{rid}__mean_reward",
            f"{rid}__std_reward",
            f"{rid}__success_rate",
        ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts in all_timesteps:
            row: dict = {"timestep": ts}
            for rid, idx_map in zip(run_ids, indexed):
                r = idx_map.get(ts)
                row[f"{rid}__mean_reward"]  = r["mean_reward"]  if r else ""
                row[f"{rid}__std_reward"]   = r["std_reward"]   if r else ""
                row[f"{rid}__success_rate"] = r["success_rate"] if r else ""
            writer.writerow(row)

    print(f"📄 Merged CSV → {path}")

# ---------------------------------------------------------------------------
# Programmatic API & CLI
# ---------------------------------------------------------------------------

def compare(
    runs: list[str] | str | None = None,
    discover_all: bool = False,
    full: bool = False,
    out: str | None = None,
) -> None:
    """Programmatic entry point for comparing training runs."""
    if discover_all:
        discovered = find_all_training_csvs()
        if not discovered:
            print("No training_evals.csv files found under ./experiments/")
            return
        run_ids   = [r for r, _ in discovered]
        csv_paths = [p for _, p in discovered]
        print(f"Found {len(run_ids)} run(s): {', '.join(run_ids)}")
    else:
        if not runs:
            print("❌ Error: Must provide either a list of runs or discover_all=True.")
            return
        run_ids   = runs if isinstance(runs, list) else runs.split(",")
        csv_paths = [find_training_csv(rid) for rid in run_ids]

    all_rows: list[list[dict]] = []
    for rid, path in zip(run_ids, csv_paths):
        rows = load_training_csv(path)
        all_rows.append(rows)
        print(f"  Loaded {len(rows):>4} eval checkpoints from run {rid}")

    summaries = [run_summary(rid, rows) for rid, rows in zip(run_ids, all_rows)]
    print_summary_table(summaries)

    if full:
        print_full_table(run_ids, all_rows)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out or f"./experiments/comparison_{ts}.csv"
    save_merged_csv(run_ids, all_rows, out_path)


def run_compare_cli(experiment_cls=None) -> None:
    """Standalone CLI entry point (experiment_cls is ignored but required by runner.py signature)."""
    p = argparse.ArgumentParser(
        description="Compare training evaluation curves across multiple runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", nargs="+", metavar="RUN_ID")
    group.add_argument("--all",  action="store_true", help="Auto-discover all runs with training_evals.csv.")
    p.add_argument("--full", action="store_true", help="Also print the full per-timestep table.")
    p.add_argument("--out",  type=str, default=None, metavar="PATH")
    
    args = p.parse_args()

    compare(
        runs=args.runs,
        discover_all=args.all,
        full=args.full,
        out=args.out
    )
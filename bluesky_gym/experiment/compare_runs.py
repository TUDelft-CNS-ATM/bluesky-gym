"""
bluesky_gym/experiment/compare_runs.py
----------------------------------------
Load and compare training evaluation curves from multiple runs.

Each run must have been trained with session.track_training_evals = True,
which writes training_evals.csv to the model save path.

Metrics Explained
-----------------
  * Final Metrics (final_mean_reward, final_success_rate): Performance at the very last evaluation checkpoint.
  * Peak Metrics (best_mean_reward, peak_success_rate): The absolute highest evaluation score achieved during the entire training run. best_at_timestep indicates when this peak occurred.
  * Convergence (steps_to_80pct_peak, steps_to_90pct_peak): The first timestep where the mean reward crossed 80% or 90% of its peak reward. Measures how quickly the model learns.
  * Tail Stability (tail_reward_cv, tail_mean_success_rate): Evaluates the final portion of training (controlled by --tail, default 20%). The Coefficient of Variation (CV) measures reward variance (lower is more stable), while the tail mean success rate shows average late-stage reliability.
  * Sample Efficiency (auc_mean_reward): The Area Under the Curve (AUC) for the reward over time, normalised by total timesteps. Higher AUC indicates the model learned faster and spent more of its training time at higher rewards.
  * Late-Stage Regression (peak_to_final_drop): The difference between the peak reward and the final reward. Evaluates if the model degraded or "forgot" how to succeed at the end of training.

Output
------
  Console  - per-run summary table (final checkpoint metrics)
  Console  - convergence & stability analysis table
  Console  - per-timestep comparison table (optional, --full)
  CSV      - merged comparison table → comparison_<timestamp>.csv

Usage
-----
  # Compare two specific runs (discovers individual training_evals.csv files)
  python compare_runs.py --runs 20260401_120000 20260401_130000

  # Compare all runs (auto-discover)
  python compare_runs.py --all

  # Re-load a previously saved merged comparison CSV and reprint all tables
  python compare_runs.py --from-csv ./experiments/comparison_20260401_120000.csv

  # Show the full per-timestep table in the console too
  python compare_runs.py --runs 20260401_120000 20260401_130000 --full

  # Save merged CSV to a custom location
  python compare_runs.py --all --out ./results/comparison.csv

  # Tune the convergence-detection window (default: last 20% of evals)
  python compare_runs.py --all --tail 0.15
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
# Loading — individual training_evals.csv files
# ---------------------------------------------------------------------------

def find_all_training_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    """Return [(run_id, csv_path), ...] for every training_evals.csv found."""
    pattern = os.path.join(base, "*/*/logs/*/training_evals.csv")
    results = []
    for path in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(path))
        results.append((run_id, path))
    return results


def find_training_csv(run_id: str, base: str = "./experiments") -> str:
    """Locate training_evals.csv for a specific run_id."""
    pattern = os.path.join(base, f"*/*/logs/{run_id}/training_evals.csv")
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


# ---------------------------------------------------------------------------
# Loading — previously saved merged comparison CSV
# ---------------------------------------------------------------------------

def load_merged_csv(path: str) -> tuple[list[str], list[list[dict]]]:
    """
    Parse a merged comparison CSV (as written by save_merged_csv) back into
    the canonical (run_ids, all_rows) format used everywhere else.

    The merged CSV has columns:
        timestep, <run_id>__mean_reward, <run_id>__std_reward, <run_id>__success_rate, ...

    Returns
    -------
    run_ids  : list of run_id strings in column order
    all_rows : list of per-run row-lists, each entry matching load_training_csv output
    """
    with open(path, newline="") as f:
        reader   = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {path}")
        fieldnames = list(reader.fieldnames)
        raw_rows   = list(reader)

    # Discover run_ids from column names  (strip the __<metric> suffix)
    run_ids: list[str] = []
    seen: set[str] = set()
    for col in fieldnames:
        if "__mean_reward" in col:
            rid = col.replace("__mean_reward", "")
            if rid not in seen:
                run_ids.append(rid)
                seen.add(rid)

    if not run_ids:
        raise ValueError(
            f"No '<run_id>__mean_reward' columns found in {path}.\n"
            f"Make sure it was produced by save_merged_csv()."
        )

    # Rebuild per-run row lists, skipping timesteps where a run has no data
    all_rows: list[list[dict]] = []
    for rid in run_ids:
        rows: list[dict] = []
        for raw in raw_rows:
            mean_r = raw.get(f"{rid}__mean_reward", "").strip()
            if mean_r == "":
                continue   # this run didn't have a checkpoint at this timestep
            rows.append({
                "timestep":     int(raw["timestep"]),
                "mean_reward":  float(mean_r),
                "std_reward":   _safe_float(raw.get(f"{rid}__std_reward")),
                "success_rate": _safe_float(raw.get(f"{rid}__success_rate")),
            })
        all_rows.append(rows)

    return run_ids, all_rows


def _safe_float(v: Optional[str]) -> float:
    if v is None or str(v).strip() in ("", "nan", "None"):
        return float("nan")
    return float(v)


# ---------------------------------------------------------------------------
# Summary stats per run
# ---------------------------------------------------------------------------

def run_summary(run_id: str, rows: list[dict], tail_frac: float = 0.20) -> dict:
    """
    Compute per-run summary statistics.

    Parameters
    ----------
    tail_frac : float
        Fraction of the eval sequence used to define the "converged tail"
        for stability / plateau metrics.  Default = last 20 %.
    """
    if not rows:
        return {"run_id": run_id, "n_evals": 0}

    rewards = np.array([r["mean_reward"]  for r in rows], dtype=float)
    success = np.array([r["success_rate"] for r in rows], dtype=float)
    steps   = np.array([r["timestep"]     for r in rows], dtype=float)
    final   = rows[-1]

    best_idx  = int(np.argmax(rewards))
    peak_rew  = float(rewards[best_idx])
    peak_sr   = float(np.nanmax(success))

    # ── Convergence: first timestep where reward crosses 80 / 90 % of peak ─
    threshold_90 = 0.90 * peak_rew
    threshold_80 = 0.80 * peak_rew
    cross_90 = _first_crossing(steps, rewards, threshold_90)
    cross_80 = _first_crossing(steps, rewards, threshold_80)

    # ── Tail stability (coefficient of variation over last tail_frac evals) ─
    tail_n = max(1, int(len(rows) * tail_frac))
    tail_rewards = rewards[-tail_n:]
    tail_success = success[-tail_n:]
    tail_cv = (float(np.std(tail_rewards)) / abs(float(np.mean(tail_rewards)))
               if np.mean(tail_rewards) != 0 else float("nan"))
    tail_mean_sr = float(np.nanmean(tail_success))

    # ── Sample efficiency: AUC (trapezoid) normalised by total timesteps ───
    _trapz  = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc_rew = float(_trapz(rewards, steps) / (steps[-1] - steps[0])) if len(steps) > 1 else float("nan")

    # ── Late-stage regression: final vs best reward ─────────────────────────
    regressed = peak_rew - float(final["mean_reward"])

    return {
        "run_id":                run_id,
        "n_evals":               len(rows),
        "final_timestep":        int(final["timestep"]),
        # Final checkpoint
        "final_mean_reward":     float(final["mean_reward"]),
        "final_std_reward":      float(final["std_reward"]),
        "final_success_rate":    float(final["success_rate"]),
        # Best checkpoint
        "best_mean_reward":      peak_rew,
        "best_at_timestep":      int(rows[best_idx]["timestep"]),
        "peak_success_rate":     peak_sr,
        # Convergence speed
        "steps_to_80pct_peak":   cross_80,
        "steps_to_90pct_peak":   cross_90,
        # Stability in converged tail
        "tail_reward_cv":        tail_cv,        # lower is more stable
        "tail_mean_success_rate": tail_mean_sr,
        # Sample efficiency
        "auc_mean_reward":       auc_rew,
        # Regression from peak to final
        "peak_to_final_drop":    regressed,
    }


def _first_crossing(steps: np.ndarray, values: np.ndarray, threshold: float) -> int:
    """Return the first timestep at which values >= threshold, or -1 if never."""
    idx = np.where(values >= threshold)[0]
    if len(idx) == 0:
        return -1
    return int(steps[idx[0]])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3) -> str:
    if isinstance(v, float) and (np.isnan(v) or v == float("nan")):
        return "n/a"
    if v == -1:
        return "never"
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)


def _fmt_pct(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if v == -1:
        return "never"
    return f"{v:.1%}"


def _fmt_steps(v) -> str:
    if v == -1:
        return "never"
    return f"{v:,}"


def _winner_mask(summaries: list[dict], key: str, higher_is_better: bool = True) -> set[str]:
    """Return the run_id(s) with the best value for a given metric key."""
    vals = [(s["run_id"], s.get(key, float("nan"))) for s in summaries]
    vals = [(rid, v) for rid, v in vals if not (isinstance(v, float) and np.isnan(v)) and v != -1]
    if not vals:
        return set()
    best = max(vals, key=lambda x: x[1]) if higher_is_better else min(vals, key=lambda x: x[1])
    return {best[0]}


def print_summary_table(summaries: list[dict]) -> None:
    """Print a two-section summary: core metrics + convergence/stability."""
    _print_core_table(summaries)
    _print_convergence_table(summaries)


def _print_core_table(summaries: list[dict]) -> None:
    cols = [
        ("run_id",             16, lambda v: str(v),        None),
        ("n_evals",             7, lambda v: str(v),        None),
        ("final_timestep",     14, lambda v: f"{v:,}",      None),
        ("final_mean_reward",  17, lambda v: _fmt(v),       True),
        ("final_success_rate", 18, lambda v: _fmt_pct(v),   True),
        ("best_mean_reward",   16, lambda v: _fmt(v),       True),
        ("best_at_timestep",   15, lambda v: _fmt_steps(v), None),
        ("peak_success_rate",  17, lambda v: _fmt_pct(v),   True),
        ("peak_to_final_drop", 18, lambda v: _fmt(v),       False),
    ]
    _print_table("CORE TRAINING METRICS", summaries, cols)


def _print_convergence_table(summaries: list[dict]) -> None:
    cols = [
        ("run_id",                16, lambda v: str(v),      None),
        ("steps_to_80pct_peak",   18, lambda v: _fmt_steps(v), False),
        ("steps_to_90pct_peak",   18, lambda v: _fmt_steps(v), False),
        ("tail_reward_cv",        15, lambda v: _fmt(v, 4),  False),
        ("tail_mean_success_rate",22, lambda v: _fmt_pct(v), True),
        ("auc_mean_reward",       15, lambda v: _fmt(v, 2),  True),
    ]
    _print_table("CONVERGENCE & STABILITY  (↓ = lower is better)", summaries, cols)


def _print_table(title: str, summaries: list[dict], cols: list) -> None:
    """Generic table printer.  cols = [(key, min_width, fmt_fn, higher_is_better|None)]"""
    winners: dict[str, set[str]] = {}
    for key, _, _, hib in cols:
        if hib is not None:
            winners[key] = _winner_mask(summaries, key, higher_is_better=hib)

    col_w  = {key: max(w, len(key)) for key, w, _, _ in cols}
    sep    = "─" * (sum(col_w[k] + 2 for k, *_ in cols) + 1)
    header = "  ".join(f"{key:>{col_w[key]}}" for key, *_ in cols)

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)

    for s in summaries:
        parts = []
        for key, _, fmt, hib in cols:
            raw   = s.get(key, float("nan"))
            cell  = fmt(raw)
            width = col_w[key]
            if hib is not None and s["run_id"] in winners.get(key, set()):
                cell = f"*{cell}"
            parts.append(f"{cell:>{width}}")
        print("  ".join(parts))

    print(sep)
    if any(hib is not None for _, _, _, hib in cols):
        print("  * = best value for that metric across all runs")
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


def print_head_to_head(summaries: list[dict]) -> None:
    """Print a plain-English verdict per headline metric (2-run comparisons only)."""
    if len(summaries) != 2:
        return

    a, b  = summaries
    metrics = [
        ("peak_success_rate",     True,  "Peak success rate"),
        ("best_mean_reward",      True,  "Best mean reward"),
        ("auc_mean_reward",       True,  "Sample efficiency (AUC)"),
        ("steps_to_90pct_peak",   False, "Speed to 90 % of peak reward"),
        ("tail_reward_cv",        False, "Tail stability (CV)"),
        ("final_mean_reward",     True,  "Final mean reward"),
    ]
    sep = "─" * 70
    print(sep)
    print("  HEAD-TO-HEAD COMPARISON")
    print(sep)
    for key, hib, label in metrics:
        va, vb = a.get(key, float("nan")), b.get(key, float("nan"))
        na = isinstance(va, float) and np.isnan(va)
        nb = isinstance(vb, float) and np.isnan(vb)
        if na and nb:
            verdict = "n/a"
        elif na:
            verdict = f"  {b['run_id']} wins  (no data for {a['run_id']})"
        elif nb:
            verdict = f"  {a['run_id']} wins  (no data for {b['run_id']})"
        elif va == vb:
            verdict = "  tie"
        elif va == -1 and vb == -1:
            verdict = "  both never reached threshold"
        elif va == -1:
            verdict = f"  {b['run_id']} wins  ({a['run_id']} never reached threshold)"
        elif vb == -1:
            verdict = f"  {a['run_id']} wins  ({b['run_id']} never reached threshold)"
        else:
            winner  = a if (hib and va > vb) or (not hib and va < vb) else b
            verdict = f"  {winner['run_id']} wins"
        print(f"  {label:<38} {verdict}")
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


def save_summary_csv(summaries: list[dict], path: str) -> None:
    """Export per-run summary stats alongside the merged CSV."""
    if not summaries:
        return
    keys = list(summaries[0].keys())
    summary_path = path.replace(".csv", "_summary.csv")
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"📄 Summary  CSV → {summary_path}")


# ---------------------------------------------------------------------------
# Shared core: print tables + optionally save outputs
# ---------------------------------------------------------------------------

def _run_comparison(
    run_ids:  list[str],
    all_rows: list[list[dict]],
    full:     bool,
    out:      Optional[str],
    tail:     float,
    save:     bool = True,
) -> None:
    summaries = [run_summary(rid, rows, tail_frac=tail) for rid, rows in zip(run_ids, all_rows)]
    print_summary_table(summaries)

    if len(summaries) == 2:
        print_head_to_head(summaries)

    if full:
        print_full_table(run_ids, all_rows)

    if save:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out or f"./experiments/results/comparison_{ts}.csv"
        save_merged_csv(run_ids, all_rows, out_path)
        save_summary_csv(summaries, out_path)


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

def compare(
    runs:         list[str] | str | None = None,
    discover_all: bool  = False,
    from_csv:     str | None = None,
    full:         bool  = False,
    out:          str | None = None,
    tail:         float = 0.20,
) -> None:
    """
    Compare training runs.  Three mutually exclusive data sources:

      runs         - list of run_id strings  (discovers individual CSVs)
      discover_all - scan ./experiments/ for all training_evals.csv files
      from_csv     - path to a previously saved merged comparison CSV
    """
    if from_csv:
        print(f"📂 Loading merged CSV: {from_csv}")
        run_ids, all_rows = load_merged_csv(from_csv)
        print(f"  Found {len(run_ids)} run(s): {', '.join(run_ids)}")
        for rid, rows in zip(run_ids, all_rows):
            print(f"  Loaded {len(rows):>4} eval checkpoints from run {rid}")
        # Don't re-save when reading back an existing CSV
        _run_comparison(run_ids, all_rows, full=full, out=out, tail=tail, save=False)
        return

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
            print("❌ Error: Must provide runs, discover_all=True, or from_csv=<path>.")
            return
        run_ids   = runs if isinstance(runs, list) else runs.split(",")
        csv_paths = [find_training_csv(rid) for rid in run_ids]

    all_rows = []
    for rid, path in zip(run_ids, csv_paths):
        rows = load_training_csv(path)
        all_rows.append(rows)
        print(f"  Loaded {len(rows):>4} eval checkpoints from run {rid}")

    _run_comparison(run_ids, all_rows, full=full, out=out, tail=tail, save=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_compare_cli(experiment_cls=None) -> None:
    """Standalone CLI entry point."""
    p = argparse.ArgumentParser(
        description="Compare training evaluation curves across multiple runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs",     nargs="+", metavar="RUN_ID",
                        help="One or more run IDs to compare.")
    source.add_argument("--all",      action="store_true",
                        help="Auto-discover all runs with training_evals.csv.")
    source.add_argument("--from-csv", metavar="PATH",
                        help="Re-load a previously saved merged comparison CSV and reprint all tables.")

    p.add_argument("--full", action="store_true",
                   help="Also print the full per-timestep table.")
    p.add_argument("--out",  type=str, default=None, metavar="PATH",
                   help="Path for the merged output CSV (ignored with --from-csv).")
    p.add_argument("--tail", type=float, default=0.20, metavar="FRAC",
                   help="Fraction of eval history used as the converged tail (default: 0.20).")

    args = p.parse_args()

    compare(
        runs=args.runs,
        discover_all=args.all,
        from_csv=getattr(args, "from_csv", None),
        full=args.full,
        out=args.out,
        tail=args.tail,
    )
"""
bluesky_gym/experiment/plot.py
---------------------------------
Plotting utilities for training curves and evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy matplotlib import so the module can be imported without a display
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return plt

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Data loaders & Path Finders
# ---------------------------------------------------------------------------

def _load_training_csv(path: str) -> list[dict]:
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
    if v is None or str(v).strip() in ("", "nan", "None"):
        return float("nan")
    return float(v)


def _find_training_csv(run_id: str, base: str = "./experiments") -> str:
    pattern = os.path.join(base, f"*/*/models/{run_id}/training_evals.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No training_evals.csv for run_id='{run_id}'. "
            f"Was session.track_training_evals=True during training?"
        )
    return matches[0]


def _find_all_training_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    pattern = os.path.join(base, "*/*/*/training_evals.csv")
    results = []
    for path in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(path))
        results.append((run_id, path))
    return results


def _find_eval_files(run_id: str, extension: str, base: str = "./experiments") -> list[str]:
    """Finds all eval files (YAML or CSV) for a specific run_id."""
    pattern = os.path.join(base, f"*/*/models/{run_id}/eval_{run_id}_*.{extension}")
    matches = glob.glob(pattern)
    if not matches:
        print(f"⚠️ No eval {extension} files found for run_id='{run_id}'")
    return sorted(matches)


def _load_eval_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _load_eval_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "episode":      int(row["episode"]),
                "group":        row["group"],
                "is_success":   row["is_success"].lower() in ("true", "1", "yes"),
                "total_reward": float(row["total_reward"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def plot_training_curves(
    run_ids:  list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    smooth:   int           = 1,
    title:    Optional[str] = None,
) -> None:
    plt = _plt()
    has_success = any(not math.isnan(r["success_rate"]) for rows in all_rows for r in rows)
    n_panels = 2 if has_success else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
    ax_rew = axes[0, 0]
    ax_sr  = axes[0, 1] if has_success else None

    def _smooth(vals: list[float], w: int) -> np.ndarray:
        if w <= 1: return np.array(vals, dtype=float)
        kernel = np.ones(w) / w
        return np.convolve(vals, kernel, mode="same")

    for i, (run_id, rows) in enumerate(zip(run_ids, all_rows)):
        if not rows: continue
        color, label = _color(i), run_id
        steps = np.array([r["timestep"] for r in rows])
        rew_s = _smooth([r["mean_reward"] for r in rows], smooth)
        ax_rew.plot(steps, rew_s, color=color, label=label, linewidth=1.8)

        if ax_sr is not None:
            sr = np.array([r["success_rate"] for r in rows], dtype=float)
            valid_sr = ~np.isnan(sr)
            if valid_sr.any():
                sr_s = _smooth(sr[valid_sr], smooth)
                ax_sr.plot(steps[valid_sr], sr_s, color=color, label=label, linewidth=1.8)

    ax_rew.set_title("Training Reward")
    ax_rew.legend(fontsize=8)
    if ax_sr: ax_sr.set_title("Training Success Rate")
    
    fig.tight_layout()
    _save_or_show(fig, out_dir, "training_curves.png", plt)


def plot_eval_summary(
    labels:     list[str],
    yaml_data:  list[dict],
    metric:     str           = "success_rate",
    out_dir:    Optional[str] = None,
    title:      Optional[str] = None,
) -> None:
    plt = _plt()
    all_groups = sorted({g for d in yaml_data for g in d.get("per_group", {}).keys()})
    if not all_groups: return

    n_runs, n_groups = len(labels), len(all_groups)
    bar_w, x = 0.8 / n_runs, np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.2 + 2), 5))

    for i, (label, d) in enumerate(zip(labels, yaml_data)):
        per_group = d.get("per_group", {})
        vals = [per_group.get(g, {}).get(metric, float("nan")) for g in all_groups]
        ax.bar(x + (i - (n_runs - 1) / 2) * bar_w, vals, width=bar_w * 0.9, color=_color(i), label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    _save_or_show(fig, out_dir, "eval_summary.png", plt)


def plot_eval_episodes(
    labels:    list[str],
    all_rows:  list[list[dict]],
    out_dir:   Optional[str] = None,
    title:     Optional[str] = None,
) -> None:
    plt = _plt()
    n_runs = len(labels)
    fig, axes = plt.subplots(1, n_runs, figsize=(max(5, 5 * n_runs), 5), squeeze=False, sharey=True)

    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        ax = axes[0, i]
        by_group = {}
        for r in rows: by_group.setdefault(r["group"], []).append(r["total_reward"])
        groups = sorted(by_group.keys())
        ax.boxplot([by_group[g] for g in groups], labels=groups)
        ax.set_title(label)

    fig.tight_layout()
    _save_or_show(fig, out_dir, "eval_episodes.png", plt)


def _save_or_show(fig, out_dir: Optional[str], filename: str, plt) -> None:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved → {path}")
    else:
        plt.show()
    plt.close(fig)

def plot(
    command: str,
    runs: list[str] | None = None,
    discover_all: bool = False,
    files: list[str] | None = None,
    run_id: str | None = None,
    metric: str = "success_rate",
    labels: list[str] | None = None,
    out: str | None = None,
    title: str | None = None,
    smooth: int = 1,
) -> None:
    """Programmatic entry point for plotting."""
    
    if command == "training":
        if discover_all:
            discovered = _find_all_training_csvs()
            run_ids, csv_paths = [r for r, _ in discovered], [c for _, c in discovered]
        else:
            if not runs:
                print("❌ Error: Provide a list of runs or discover_all=True")
                return
            run_ids = runs
            csv_paths = [_find_training_csv(r) for r in run_ids]
            
        all_rows = [_load_training_csv(p) for p in csv_paths]
        plot_training_curves(run_ids, all_rows, out, smooth, title)

    elif command in ["eval-summary", "eval-episodes"]:
        ext = "yaml" if command == "eval-summary" else "csv"
        eval_files = files
        if not eval_files and run_id:
            eval_files = _find_eval_files(run_id, ext)
        
        if not eval_files:
            print(f"❌ Error: Provide either 'files' or a 'run_id' with existing {ext} files.")
            return

        plot_labels = labels or [os.path.basename(f) for f in eval_files]
        
        if command == "eval-summary":
            yaml_data = [_load_eval_yaml(f) for f in eval_files]
            plot_eval_summary(plot_labels, yaml_data, metric, out, title)
        else:
            csv_data = [_load_eval_csv(f) for f in eval_files]
            plot_eval_episodes(plot_labels, csv_data, out, title)
    else:
        print(f"❌ Unknown command: {command}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training curves and evaluation results.")
    sub = p.add_subparsers(dest="command", required=True)

    # Training
    tr = sub.add_parser("training")
    src = tr.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+", metavar="RUN_ID")
    src.add_argument("--all", action="store_true")
    tr.add_argument("--smooth", type=int, default=1)
    tr.add_argument("--out", type=str, default=None)
    tr.add_argument("--title", type=str, default=None)

    # Eval Summary - UPDATED
    es = sub.add_parser("eval-summary")
    es.add_argument("--files", nargs="+", metavar="YAML_PATH")
    es.add_argument("--run-id", type=str, help="Auto-locate eval YAMLs for this run ID.")
    es.add_argument("--metric", type=str, default="success_rate")
    es.add_argument("--labels", nargs="+", default=None)
    es.add_argument("--out", type=str, default=None)
    es.add_argument("--title", type=str, default=None)

    # Eval Episodes - UPDATED
    ep = sub.add_parser("eval-episodes")
    ep.add_argument("--files", nargs="+", metavar="CSV_PATH")
    ep.add_argument("--run-id", type=str, help="Auto-locate eval CSVs for this run ID.")
    ep.add_argument("--labels", nargs="+", default=None)
    ep.add_argument("--out", type=str, default=None)
    ep.add_argument("--title", type=str, default=None)

    return p

def run_plot_cli(experiment_cls=None) -> None:
    """Standalone CLI entry point."""
    args = _build_parser().parse_args()

    if args.command == "training":
        plot(
            command="training", 
            runs=getattr(args, 'runs', None), 
            discover_all=getattr(args, 'all', False),
            smooth=args.smooth, 
            out=args.out, 
            title=args.title
        )
    elif args.command in ["eval-summary", "eval-episodes"]:
        plot(
            command=args.command, 
            files=args.files, 
            run_id=args.run_id, 
            metric=getattr(args, 'metric', 'success_rate'), 
            labels=args.labels, 
            out=args.out, 
            title=args.title
        )
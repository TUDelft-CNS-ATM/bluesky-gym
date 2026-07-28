"""
Plotting API and Command-Line Interface.

This module serves as the primary entry point for generating visual analytics 
across both training and evaluation phases. It parses user commands to locate 
log files (CSV/YAML) and routes the data to the appropriate plotting engines 
defined in `train_plots.py` and `eval_plots.py`.
"""

from __future__ import annotations

import argparse
import os

from .data import _find_training_csv, _find_all_training_csvs, _find_eval_files, _load_eval_csv, _load_eval_yaml, _load_merged_csv, _load_training_csv
from .train_plots import plot_training_curves, plot_comparison_grid
from .eval_plots import plot_eval_summary, plot_eval_episodes, plot_eval_dashboard

def plot(
    command:      str,
    runs:         list[str] | None = None,
    discover_all: bool = False,
    from_csv:     str | None = None,
    files:        list[str] | None = None,
    run_id:       str | None = None,
    metric:       str = "success_rate",
    labels:       list[str] | None = None,
    out:          str | None = None,
    title:        str | None = None,
    smooth:       int = 1,
    grid:         bool = False,
) -> None:
    """
    Programmatic entry point for plotting.

    Parameters
    ----------
    from_csv : str, optional
        Path to a merged comparison CSV (output of compare_runs).  When
        provided for the 'training' command, run IDs and data are loaded
        directly from it — no individual training_evals.csv files needed.
    grid : bool
        When True and command='training', render the richer 2x2 comparison
        grid instead of the simple side-by-side panels.
    """
    if command == "training":
        # ── Resolve data source ──────────────────────────────────────────────
        if from_csv:
            run_ids, all_rows = _load_merged_csv(from_csv)
            print(f"📂 Loaded {len(run_ids)} run(s) from {from_csv}")
        elif discover_all:
            discovered = _find_all_training_csvs()
            run_ids  = [r for r, _ in discovered]
            all_rows = [_load_training_csv(p) for _, p in discovered]
        else:
            if not runs:
                print("❌ Error: Provide --runs, --all, or --from-csv for the training command.")
                return
            run_ids  = runs
            all_rows = [_load_training_csv(_find_training_csv(r)) for r in run_ids]

        # Map labels
        plot_labels = labels if labels and len(labels) == len(run_ids) else run_ids
        if labels and len(labels) != len(run_ids):
            print(f"⚠️ Warning: Provided {len(labels)} labels for {len(run_ids)} runs. Defaulting to Run IDs.")

        if grid or len(run_ids) > 1:
            plot_comparison_grid(plot_labels, all_rows, out, smooth, title)
        else:
            plot_training_curves(plot_labels, all_rows, out, smooth, title)

    elif command == "eval-dashboard":
        # Single-run dashboard: needs one CSV (and optionally one YAML)
        csv_files  = files or (_find_eval_files(run_id, "csv")  if run_id else [])
        yaml_files = _find_eval_files(run_id, "yaml") if run_id else []
        if not csv_files:
            print("❌ Error: Provide --files <CSV> or --run-id for eval-dashboard.")
            return
        dash_labels = labels or [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
        for label, csv_path in zip(dash_labels, csv_files):
            rows      = _load_eval_csv(csv_path)
            yaml_d    = _load_eval_yaml(yaml_files[0]) if yaml_files else None
            plot_eval_dashboard(label, rows, yaml_d, out, title)

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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training curves and evaluation results.")
    sub = p.add_subparsers(dest="command", required=True)

    # ── training subcommand ──────────────────────────────────────────────────
    tr = sub.add_parser("training", help="Plot training reward / success-rate curves.")
    src = tr.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs",     nargs="+", metavar="RUN_ID",
                     help="One or more run IDs.")
    src.add_argument("--all",      action="store_true",
                     help="Auto-discover all runs.")
    src.add_argument("--from-csv", metavar="PATH",
                     help="Load directly from a merged comparison CSV.")
    tr.add_argument("--labels", nargs="+", default=None,
                    help="Custom labels for the legend. Must match the number of runs.")
    tr.add_argument("--smooth", type=int, default=1,
                    help="Rolling-average window size (default: 1 = no smoothing).")
    tr.add_argument("--grid",   action="store_true",
                    help="Render the richer 2×2 comparison grid.")
    tr.add_argument("--out",    type=str, default=None,
                    help="Output directory for saved plots.")
    tr.add_argument("--title",  type=str, default=None)

    # ── eval-summary subcommand ──────────────────────────────────────────────
    es = sub.add_parser("eval-summary", help="Plot evaluation summary data.")
    es.add_argument("--files",  nargs="+", metavar="YAML_PATH")
    es.add_argument("--run-id", type=str)
    es.add_argument("--metric", type=str, default="success_rate")
    es.add_argument("--labels", nargs="+", default=None)
    es.add_argument("--out",    type=str, default=None)
    es.add_argument("--title",  type=str, default=None)

    # ── eval-episodes subcommand ─────────────────────────────────────────────
    ep = sub.add_parser("eval-episodes", help="Plot evaluation episode data.")
    ep.add_argument("--files",  nargs="+", metavar="CSV_PATH")
    ep.add_argument("--run-id", type=str)
    ep.add_argument("--labels", nargs="+", default=None)
    ep.add_argument("--out",    type=str, default=None)
    ep.add_argument("--title",  type=str, default=None)

    # ── eval-dashboard subcommand ─────────────────────────────────────────────
    ed = sub.add_parser("eval-dashboard",
                        help="Single-run dashboard: reward distribution, success rate, timeline, extras.")
    ed.add_argument("--files",  nargs="+", metavar="CSV_PATH",
                    help="One or more episode CSV files (one dashboard per file).")
    ed.add_argument("--run-id", type=str,
                    help="Auto-locate eval CSVs (and YAML) for this run ID.")
    ed.add_argument("--labels", nargs="+", default=None)
    ed.add_argument("--out",    type=str, default=None)
    ed.add_argument("--title",  type=str, default=None)

    return p


def run_plot_cli(experiment_cls=None) -> None:
    """Standalone CLI entry point."""
    args = _build_parser().parse_args()

    if args.command == "training":
        plot(
            command="training",
            runs=getattr(args, "runs", None),
            discover_all=getattr(args, "all", False),
            from_csv=getattr(args, "from_csv", None),
            smooth=args.smooth,
            grid=getattr(args, "grid", False),
            labels=getattr(args, "labels", None),
            out=args.out,
            title=args.title,
        )
    elif args.command in ["eval-summary", "eval-episodes", "eval-dashboard"]:
        plot(
            command=args.command,
            files=args.files,
            run_id=getattr(args, "run_id", None),
            metric=getattr(args, "metric", "success_rate"),
            labels=getattr(args, "labels", None),
            out=args.out,
            title=args.title,
        )
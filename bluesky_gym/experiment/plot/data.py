"""
Data ingestion and file discovery utilities.

Handles the parsing of CSV and YAML artifacts generated during training and 
evaluation. This includes safely loading numeric data, resolving file globs 
from run IDs, and reconstructing multi-run data structures from merged logs.
"""

from __future__ import annotations

import csv
import glob
import os

from typing import Optional


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
    pattern = os.path.join(base, f"*/*/logs/{run_id}/training_evals.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No training_evals.csv for run_id='{run_id}'. "
            f"Was session.track_training_evals=True during training?"
        )
    return matches[0]


def _find_all_training_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    pattern = os.path.join(base, "*/*/logs/*/training_evals.csv")
    results = []
    for path in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(path))
        results.append((run_id, path))
    return results


def _load_merged_csv(path: str) -> tuple[list[str], list[list[dict]]]:
    """Parse a merged comparison CSV back into (run_ids, all_rows)."""
    with open(path, newline="") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        raw_rows   = list(reader)

    run_ids: list[str] = []
    seen: set[str] = set()
    for col in fieldnames:
        if "__mean_reward" in col:
            rid = col.replace("__mean_reward", "")
            if rid not in seen:
                run_ids.append(rid)
                seen.add(rid)

    if not run_ids:
        raise ValueError(f"No '<run_id>__mean_reward' columns found in {path}.")

    all_rows: list[list[dict]] = []
    for rid in run_ids:
        rows: list[dict] = []
        for raw in raw_rows:
            mean_r = raw.get(f"{rid}__mean_reward", "").strip()
            if mean_r == "":
                continue
            rows.append({
                "timestep":     int(raw["timestep"]),
                "mean_reward":  float(mean_r),
                "std_reward":   _safe_float(raw.get(f"{rid}__std_reward")),
                "success_rate": _safe_float(raw.get(f"{rid}__success_rate")),
            })
        all_rows.append(rows)

    return run_ids, all_rows


def _find_eval_files(run_id: str, extension: str, base: str = "./experiments") -> list[str]:
    pattern = os.path.join(base, f"*/*/models/{run_id}/eval_{run_id}_*.{extension}")
    matches = glob.glob(pattern)
    if not matches:
        print(f"⚠️  No eval {extension} files found for run_id='{run_id}'")
    return sorted(matches)


def _load_eval_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)

def _load_eval_csv(path: str) -> list[dict]:
    """Load an episode CSV, preserving ALL columns (including MetricExtractor extras)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: dict = {
                "episode":      int(row["episode"]),
                "group":        row["group"],
                "is_success":   row["is_success"].lower() in ("true", "1", "yes"),
                "total_reward": float(row["total_reward"]),
            }
            # Preserve any extra columns written by MetricExtractor
            fixed = {"episode", "group", "is_success", "total_reward"}
            for k, v in row.items():
                if k not in fixed:
                    rec[k] = _safe_float(v) if v.strip() not in ("", "None") else float("nan")
            rows.append(rec)
    return rows
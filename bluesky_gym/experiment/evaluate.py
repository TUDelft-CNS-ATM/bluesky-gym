"""
bluesky_gym/experiment/evaluate.py
------------------------------------
Generic evaluation script for any trained SB3 + GoalEnv model.
"""

from __future__ import annotations

import argparse
import csv
import os
import yaml
from datetime import datetime
from typing import Any, Callable, Optional, TypedDict, cast
 
import bluesky_gym
import numpy as np
 
from .config import ExperimentConfig


# ---------------------------------------------------------------------------
# MetricExtractor — Updated for Any-type support
# ---------------------------------------------------------------------------

class MetricExtractor:
    """Declares which extra metrics to pull from an episode's final info dict.

    Parameters
    ----------
    extractors : dict[str, Callable[[dict, bool], Any]]
        Maps metric name → function(info, is_success) → Any.
        Now supports non-float return types.

    aggregators : dict[str, Callable[[list[Any]], Any]] | None
        Per-metric aggregation overrides. Defaults to np.nanmean for numbers,
        or a simple 'list' capture for non-numeric data if not specified.

    display : list[str] | None
        Ordered subset of metric names shown in the console table.
    """

    def __init__(
        self,
        extractors:  dict[str, Callable[[dict, bool], Any]],
        aggregators: dict[str, Callable[[list[Any]], Any]] | None = None,
        display:     list[str] | None = None,
    ) -> None:
        self.extractors  = extractors
        self.aggregators = aggregators or {}
        self.display     = display or list(extractors.keys())

    def extract(self, info: dict, is_success: bool) -> dict[str, Any]:
        """Extract metrics from the episode info dict."""
        return {name: fn(info, is_success) for name, fn in self.extractors.items()}

    def aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate extracted metrics across multiple episodes."""
        result: dict[str, Any] = {}
        for name in self.extractors:
            values = [r[name] for r in rows]
            
            # Use custom aggregator if provided; otherwise default based on type
            if name in self.aggregators:
                agg_fn = self.aggregators[name]
            else:
                # Default to mean for numbers, or just returning the list for others
                is_numeric = all(isinstance(v, (int, float, np.number)) for v in values if v is not None)
                agg_fn = cast(Callable, np.nanmean if is_numeric else list)

            try:
                result[name] = agg_fn(values)
            except Exception:
                result[name] = float("nan")
        return result


# ---------------------------------------------------------------------------
# Episode record — Updated for flexibility
# ---------------------------------------------------------------------------

class EpisodeRecord(TypedDict):
    episode:      int
    group:        str
    is_success:   bool
    total_reward: float
    # Extras are dynamically added and can be Any type

def _make_record(
    episode:      int,
    group:        str,
    is_success:   bool,
    total_reward: float,
    extras:       dict[str, Any],
) -> EpisodeRecord:
    rec = {
        "episode":      episode,
        "group":        group,
        "is_success":   is_success,
        "total_reward": total_reward,
        **extras,
    }
    return cast(EpisodeRecord, rec)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg:            ExperimentConfig,
    experiment_cls,
    n_episodes:     int,
    groups:         Optional[list[str]],
    render:         bool,
) -> list[EpisodeRecord]:
    """Run n_episodes episodes; return one EpisodeRecord per episode."""
 
    extractor   = experiment_cls.metric_extractor()
    success_key = cfg.env.success_key
    group_key   = cfg.env.group_key
 
    eval_kwargs = cfg.eval_env_kwargs
    if groups is not None:
        from .config import _inject_groups
        _inject_groups(eval_kwargs, cfg.env.env_kwargs, groups)
 
    render_mode = "human" if render else None
    experiment  = experiment_cls(cfg)
    env         = experiment.make_env(eval_kwargs, render_mode=render_mode)
    model       = cfg.model.get_algorithm().load(f"{cfg.save_path}/final_model", env=env)
 
    records: list[EpisodeRecord] = []
 
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = truncated = False
        info: dict = {}
 
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            if hasattr(action, "shape") and action.shape == ():
                action = action[()]
            obs, _, done, truncated, info = env.step(action)
 
        is_success   = bool(info.get(success_key, False))
        total_reward = float(info.get("total_reward", 0.0))
        group        = str(info.get(group_key, "unknown")) if group_key else "all"
        extras       = extractor.extract(info, is_success) if extractor else {}
 
        rec = _make_record(ep, group, is_success, total_reward, extras)
        records.append(rec)
 
        status    = "✅" if is_success else "❌"
        extra_str = "  ".join(
            f"{k}={_fmt_val(v)}"
            for k, v in extras.items()
            if extractor and k in extractor.display
            and not (isinstance(v, float) and np.isnan(v))
        )
        print(
            f"  Ep {ep:>3}/{n_episodes}  {status}  "
            f"group={group:<6}  reward={total_reward:+.1f}"
            + (f"  {extra_str}" if extra_str else "")
        )
 
    env.close()
    return records


# ---------------------------------------------------------------------------
# Aggregation & Formatting — Optimized for Any types
# ---------------------------------------------------------------------------

def _fmt_val(v) -> str:
    """Smart formatting for arbitrary metric types."""
    if isinstance(v, (float, np.floating)):
        return "n/a" if np.isnan(v) else f"{v:.2f}"
    if isinstance(v, (list, tuple)):
        return f"len={len(v)}"
    return str(v)

def _fmt_pct(v) -> str:
    return "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


def _aggregate_group(
    label:     str,
    recs:      list[EpisodeRecord],
    extractor: MetricExtractor | None,
) -> dict:
    n = len(recs)
    if n == 0:
        base = {"group": label, "n_episodes": 0,
                "success_rate": 0.0,
                "mean_total_reward": float("nan"),
                "std_total_reward":  float("nan")}
        if extractor:
            base.update({k: float("nan") for k in extractor.extractors})
        return base

    rewards = [r["total_reward"] for r in recs]
    base = {
        "group":             label,
        "n_episodes":        n,
        "success_rate":      sum(r["is_success"] for r in recs) / n,
        "mean_total_reward": float(np.mean(rewards)),
        "std_total_reward":  float(np.std(rewards)),
    }
    if extractor:
        # Pass the full list of extra metrics to the flexible aggregator
        extra_rows = [{k: r[k] for k in extractor.extractors} for r in recs]
        base.update(extractor.aggregate(extra_rows))
    return base


def aggregate_metrics(
    records:   list[EpisodeRecord],
    extractor: MetricExtractor | None,
) -> tuple[dict, dict[str, dict]]:
    """Return (overall_summary, per_group_summary_dict)."""
    by_group: dict[str, list[EpisodeRecord]] = {}
    for rec in records:
        by_group.setdefault(rec["group"], []).append(rec)

    overall   = _aggregate_group("overall", records, extractor)
    per_group = {
        g: _aggregate_group(g, recs, extractor)
        for g, recs in sorted(by_group.items())
    }
    return overall, per_group


def print_summary(
    overall:   dict,
    per_group: dict[str, dict],
    extractor: MetricExtractor | None,
) -> None:
    """Print the final evaluation table, handling Any types gracefully."""
    fixed_cols = ["group", "n_episodes", "success_rate",
                  "mean_total_reward", "std_total_reward"]
    extra_cols = extractor.display if extractor else []
    all_cols   = fixed_cols + extra_cols

    col_w  = 14
    sep    = "─" * ((col_w + 2) * len(all_cols) + 1)
    header = "  ".join(f"{c:>{col_w}}" for c in all_cols)

    def _row(m: dict) -> str:
        cells = []
        for c in all_cols:
            v = m.get(c, float("nan"))
            if   c == "group":       cells.append(f"{str(v):>{col_w}}")
            elif c == "n_episodes":  cells.append(f"{int(v):>{col_w}}")
            elif c == "success_rate":cells.append(f"{_fmt_pct(v):>{col_w}}")
            else:                    cells.append(f"{_fmt_val(v):>{col_w}}")
        return "  ".join(cells)

    print(f"\n{sep}")
    print("  EVALUATION SUMMARY")
    print(sep)
    print(header)
    print(sep)
    for m in per_group.values():
        print(_row(m))
    print(sep)
    print(_row(overall))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_csv(records: list[EpisodeRecord], path: str) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"📄 Episode CSV  → {path}")


def save_yaml_summary(overall: dict, per_group: dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _clean(val: Any) -> Any:
        """Deep clean types for YAML serialization (e.g., converting NaNs)."""
        if isinstance(val, dict):
            return {k: _clean(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_clean(v) for v in val]
        
        # Handle NumPy scalars (converts np.float64 -> float, np.int64 -> int)
        if hasattr(val, "item") and not isinstance(val, (list, np.ndarray)):
            val = val.item()
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return None
        return val

    payload = {
        "overall":   _clean(overall),
        "per_group": {g: _clean(m) for g, m in per_group.items()},
    }
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f"📊 Metrics YAML → {path}")

def evaluate(
    experiment_cls,
    run_id: str,
    episodes: int | None = None,
    groups: list[str] | None = None,
    render: bool = True,
) -> tuple[dict, dict]:
    """
    Programmatic API to evaluate a trained model.
    Returns (overall_metrics, per_group_metrics).
    """
    import bluesky_gym
    bluesky_gym.register_envs()

    # 1. Load Config
    cfg = ExperimentConfig.load(
        run_id,
        model_config_cls=experiment_cls.model_config_cls,
        env_config_cls=experiment_cls.env_config_cls,
    )

    n_episodes = episodes or cfg.session.eval_episodes
    eval_groups = groups or cfg.session.eval_groups
    algo_name = cfg.model.algorithm.__name__ if cfg.model.algorithm else "Unspecified"

    print(f"\n🔍 Evaluating run  {cfg.run_id}")
    print(f"   env     = {cfg.env.env_name}")
    print(f"   algo    = {algo_name}")
    print(f"   n_eps   = {n_episodes}")
    print(f"   groups  = {eval_groups or 'All'}\n")

    # 2. Run Inference
    records = run_evaluation(
        cfg=cfg,
        experiment_cls=experiment_cls,
        n_episodes=n_episodes,
        groups=eval_groups,
        render=render,
    )

    # 3. Process Metrics
    extractor = experiment_cls.metric_extractor()
    overall, per_group = aggregate_metrics(records, extractor)
    print_summary(overall, per_group, extractor)

    # 4. Save Artifacts
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"eval_{cfg.run_id}_{ts}"
    save_csv(records, os.path.join(cfg.save_path, f"{stem}.csv"))
    save_yaml_summary(overall, per_group, os.path.join(cfg.save_path, f"{stem}.yaml"))

    return overall, per_group


def run_evaluate_cli(experiment_cls):
    """CLI wrapper for the evaluate function."""
    p = argparse.ArgumentParser(
        description="Evaluate a trained model with detailed metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id",    type=str, required=True, help="The ID of the run to evaluate.")
    p.add_argument("--episodes",  type=int, default=None,  help="Override default episode count.")
    p.add_argument("--groups",    nargs="+", default=None, metavar="GROUP", help="Specific groups to test.")
    p.add_argument("--no-render", action="store_true",     help="Disable UI rendering.")
    
    args = p.parse_args()

    evaluate(
        experiment_cls=experiment_cls,
        run_id=args.run_id,
        episodes=args.episodes,
        groups=args.groups,
        render=not args.no_render
    )
"""
Evaluation plotting suite.

Generates visual analytics for post-training model evaluations. This includes 
single-run performance dashboards, cross-run success rate summaries, and 
detailed episode-by-episode timelines. It dynamically accommodates custom 
metrics extracted via the `MetricExtractor` class.
"""

from __future__ import annotations

from .style import _apply_style, _color, _plt, _get_ticker, _smooth, _save_or_show

import math
import numpy as np
from typing import Optional

def _extra_numeric_keys(rows: list[dict], yaml_data: dict | None = None) -> list[str]:
    """
    Return extra numeric column names beyond the four fixed fields.
    
    If yaml_data is provided, it cross-references the 'overall' keys to ensure
    we only plot metrics that were explicitly aggregated.
    """
    fixed = {"episode", "group", "is_success", "total_reward"}
    if not rows:
        return []

    # 1. Identify candidates from the CSV row
    candidates = [
        k for k in rows[0] 
        if k not in fixed
        and isinstance(rows[0][k], (int, float))
        and not (isinstance(rows[0][k], float) and math.isnan(rows[0][k]))
    ]

    # 2. If we have YAML metadata, refine the list
    if yaml_data and "overall" in yaml_data:
        # The YAML 'overall' section contains keys like 'mean_total_reward', 
        # 'std_total_reward', and our extras.
        yaml_keys = yaml_data["overall"].keys()
        
        # We only keep candidates that exist in the YAML 
        # (The MetricExtractor output keys match the CSV column names)
        candidates = [k for k in candidates if k in yaml_keys]

    return candidates

def plot_eval_dashboard(
    label:   str,
    rows:    list[dict],
    yaml_data: dict | None = None,
    out_dir: Optional[str] = None,
    title:   Optional[str] = None,
) -> None:
    """
    Single-run evaluation dashboard with dynamic grid sizing.

    Always shows:
      1. Reward distribution by group (violin + jitter)
      2. Success rate by group (horizontal bar)
      3. Episode timeline (scatter + rolling mean)
      
    Then dynamically appends:
      - A bar chart for *each* extra metric extracted.
      - Or a reward histogram if no extra metrics exist.
    """
    _apply_style()
    plt = _plt()
    ticker = _get_ticker()

    groups       = sorted({r["group"] for r in rows})
    n_groups     = len(groups)
    group_colors = {g: _color(i) for i, g in enumerate(groups)}
    extras       = _extra_numeric_keys(rows, yaml_data)
    by_group     = {g: [r for r in rows if r["group"] == g] for g in groups}

    # ── Dynamic Grid Calculation ─────────────────────────────────────────────
    # Base 3 plots + either 1 plot per extra metric OR 1 fallback histogram
    n_plots = 3 + (len(extras) if extras else 1)
    
    # Force columns to be either 2 or 3 for optimal viewing
    cols = 3 if n_plots >= 5 or n_plots == 3 else 2
    rows_grid = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 6.5, rows_grid * 4.5))
    
    # Flatten axes for easy sequential iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    ax_idx = 0

    # ── 1. Reward violin + jitter ────────────────────────────────────────────
    ax_viol = axes[ax_idx]; ax_idx += 1
    reward_data = [np.array([r["total_reward"] for r in by_group[g]]) for g in groups]
    vp = ax_viol.violinplot(reward_data, positions=range(n_groups),
                             showmedians=True, showextrema=False)
    for pc, g in zip(vp["bodies"], groups):
        pc.set_facecolor(group_colors[g])
        pc.set_alpha(0.55)
    vp["cmedians"].set_color("#333333")
    vp["cmedians"].set_linewidth(2)
    for j, g in enumerate(groups):
        jx = np.random.default_rng(j).uniform(-0.12, 0.12, len(by_group[g]))
        ys = [r["total_reward"] for r in by_group[g]]
        cs = [("#2ecc71" if r["is_success"] else "#e74c3c") for r in by_group[g]]
        ax_viol.scatter(j + jx, ys, c=cs, s=18, alpha=0.7, zorder=3, linewidths=0)
    ax_viol.set_xticks(range(n_groups))
    ax_viol.set_xticklabels(groups)
    ax_viol.set_ylabel("Total Reward")
    ax_viol.set_title("Reward Distribution by Group")
    
    from matplotlib.lines import Line2D
    ax_viol.legend(
        handles=[Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7, label="success"),
                 Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=7, label="failure")],
        fontsize=8, loc="upper right",
    )

    # ── 2. Success rate horizontal bars ──────────────────────────────────────
    ax_sr = axes[ax_idx]; ax_idx += 1
    sr_vals = [sum(r["is_success"] for r in by_group[g]) / len(by_group[g]) for g in groups]
    n_eps   = [len(by_group[g]) for g in groups]
    bars = ax_sr.barh(range(n_groups), sr_vals, color=[group_colors[g] for g in groups],
                      alpha=0.75, zorder=3)
    for j, (bar, sr, n) in enumerate(zip(bars, sr_vals, n_eps)):
        ax_sr.text(min(sr + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                   f"{sr:.0%}  (n={n})", va="center", fontsize=8, color="#333333")
    overall_sr = sum(r["is_success"] for r in rows) / len(rows)
    ax_sr.axvline(overall_sr, color="#444444", linewidth=1.5, linestyle="--", label=f"overall {overall_sr:.0%}")
    ax_sr.set_yticks(range(n_groups))
    ax_sr.set_yticklabels(groups)
    ax_sr.set_xlim(0, 1.15)
    ax_sr.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_sr.set_title("Success Rate by Group")
    ax_sr.legend(fontsize=8)

    # ── 3. Episode timeline ──────────────────────────────────────────────────
    ax_time = axes[ax_idx]; ax_idx += 1
    eps     = [r["episode"]      for r in rows]
    rewards = [r["total_reward"] for r in rows]
    colors  = [("#2ecc71" if r["is_success"] else "#e74c3c") for r in rows]
    ax_time.scatter(eps, rewards, c=colors, s=20, alpha=0.75, zorder=3, linewidths=0)
    if len(rows) >= 5:
        rm = _smooth(np.array(rewards, dtype=float), max(3, len(rows) // 10))
        ax_time.plot(eps, rm, color="#444444", linewidth=1.5, label="rolling mean", zorder=4)
        ax_time.legend(fontsize=8)
    ax_time.set_xlabel("Episode")
    ax_time.set_ylabel("Total Reward")
    ax_time.set_title("Episode Timeline  (● success  ● failure)")

    # ── 4. Dynamic Extra Metrics (or Fallback Histogram) ─────────────────────
    if extras:
        # Generate a bar chart for every extra metric we tracked
        for key in extras:
            ax_ext = axes[ax_idx]
            ax_idx += 1
            
            vals = [np.nanmean([r[key] for r in by_group[g]]) for g in groups]
            stds = [np.nanstd( [r[key] for r in by_group[g]]) for g in groups]
            
            bars2 = ax_ext.bar(range(n_groups), vals, yerr=stds,
                               color=[group_colors[g] for g in groups],
                               alpha=0.75, capsize=4, zorder=3,
                               error_kw=dict(elinewidth=1, ecolor="#666666"))
            for bar, v in zip(bars2, vals):
                ax_ext.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1,
                            f"{v:.2f}", ha="center", fontsize=7, color="#333333")
            ax_ext.set_xticks(range(n_groups))
            ax_ext.set_xticklabels(groups)
            ax_ext.set_ylabel(key.replace("_", " ").title())
            ax_ext.set_title(f"{key.replace('_', ' ').title()} by Group  (mean ± std)")
    else:
        # Fallback: overall reward histogram coloured by group
        ax_hist = axes[ax_idx]; ax_idx += 1
        for g in groups:
            vals = [r["total_reward"] for r in by_group[g]]
            ax_hist.hist(vals, bins=12, alpha=0.55, color=group_colors[g], label=g, zorder=3)
        ax_hist.set_xlabel("Total Reward")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Reward Histogram by Group")
        ax_hist.legend(fontsize=8)

    # ── Clean up empty subplots ──────────────────────────────────────────────
    for i in range(ax_idx, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title or f"Evaluation Dashboard — {label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, out_dir, f"eval_dashboard_{label}.png", plt)

def plot_eval_summary(
    labels:    list[str],
    yaml_data: list[dict],
    metric:    str           = "success_rate",
    out_dir:   Optional[str] = None,
    title:     Optional[str] = None,
) -> None:
    """
    Cross-run grouped bar chart for one metric, with mean±std error bars
    drawn from the YAML overall section.  Overall value annotated as a
    dashed line per run.
    """
    _apply_style()
    plt = _plt()
    ticker = _get_ticker()

    all_groups = sorted({g for d in yaml_data for g in d.get("per_group", {}).keys()})
    if not all_groups:
        return

    n_runs, n_groups = len(labels), len(all_groups)
    bar_w = 0.75 / n_runs
    x = np.arange(n_groups)

    # Infer std key: e.g. success_rate → no std in YAML; mean_total_reward → std_total_reward
    std_key_map = {"mean_total_reward": "std_total_reward"}
    std_key = std_key_map.get(metric)

    is_pct = metric in ("success_rate",)
    fmt_v  = (lambda v: f"{v:.0%}") if is_pct else (lambda v: f"{v:.2f}")

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.4 + 2), 5))

    for i, (label, d) in enumerate(zip(labels, yaml_data)):
        per_group = d.get("per_group", {})
        overall   = d.get("overall",   {})

        vals = np.array([per_group.get(g, {}).get(metric, np.nan) for g in all_groups])
        errs = None
        if std_key:
            errs = np.array([per_group.get(g, {}).get(std_key, np.nan) for g in all_groups])

        offset = (i - (n_runs - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, vals,
            width=bar_w * 0.9,
            color=_color(i), alpha=0.80,
            yerr=errs if errs is not None else None,
            capsize=3,
            error_kw=dict(elinewidth=1, ecolor="#555555"),
            label=label, zorder=3,
        )
        # Value labels
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.01 if is_pct else 0.05),
                        fmt_v(v),
                        ha="center", va="bottom", fontsize=7, color="#333333")

        # Overall dashed line
        ov = overall.get(metric, np.nan)
        if not np.isnan(ov):
            ax.axhline(ov, color=_color(i), linewidth=1.2, linestyle="--",
                       alpha=0.6, label=f"{label} overall ({fmt_v(ov)})")

    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    if is_pct:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.15)
    ax.set_title(title or f"Eval — {metric.replace('_', ' ').title()} by Group")
    ax.legend(fontsize=8, ncol=min(n_runs * 2, 4))
    fig.tight_layout()
    _save_or_show(fig, out_dir, "eval_summary.png", plt)


def plot_eval_episodes(
    labels:   list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    title:    Optional[str] = None,
) -> None:
    """
    Per-run 2x2 comparison grid:
      [0,0]  Reward boxplots by group (one subplot per run, shared y)
      [0,1]  Success rate grouped bars across runs
      [1,0]  Episode timeline scatter for all runs overlaid
      [1,1]  Reward distributions (overlapping histograms per run)
    """
    _apply_style()
    plt  = _plt()
    ticker = _get_ticker()

    all_groups = sorted({r["group"] for rows in all_rows for r in rows})
    n_runs = len(labels)

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)
    ax_box  = fig.add_subplot(gs[0, 0])
    ax_sr   = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # ── [0,0]  Reward boxplots — all runs side-by-side per group ────────────
    x = np.arange(len(all_groups))
    bw = 0.7 / n_runs
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        by_group = {g: [r["total_reward"] for r in rows if r["group"] == g]
                    for g in all_groups}
        offset = (i - (n_runs - 1) / 2) * bw
        bp = ax_box.boxplot(
            [by_group.get(g, [0]) for g in all_groups],
            positions=x + offset,
            widths=bw * 0.85,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=_color(i), linewidth=1.2),
            capprops=dict(color=_color(i), linewidth=1.2),
            flierprops=dict(marker=".", color=_color(i), alpha=0.5, markersize=4),
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(_color(i))
            patch.set_alpha(0.65)
        # Invisible bar for legend
        ax_box.bar(0, 0, color=_color(i), alpha=0.65, label=label)

    ax_box.set_xticks(x)
    ax_box.set_xticklabels(all_groups, rotation=20, ha="right")
    ax_box.set_ylabel("Total Reward")
    ax_box.set_title("Reward Distribution by Group")
    ax_box.legend(fontsize=8)

    # ── [0,1]  Success rate grouped bars ─────────────────────────────────────
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        by_group = {g: [r for r in rows if r["group"] == g] for g in all_groups}
        sr_vals  = [sum(r["is_success"] for r in by_group.get(g, [])) /
                    max(1, len(by_group.get(g, []))) for g in all_groups]
        offset   = (i - (n_runs - 1) / 2) * bw
        bars     = ax_sr.bar(x + offset, sr_vals, width=bw * 0.85,
                             color=_color(i), alpha=0.80, label=label, zorder=3)
        for bar, v in zip(bars, sr_vals):
            ax_sr.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.015,
                       f"{v:.0%}", ha="center", fontsize=7, color="#333333")

    ax_sr.set_xticks(x)
    ax_sr.set_xticklabels(all_groups, rotation=20, ha="right")
    ax_sr.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_sr.set_ylim(0, 1.15)
    ax_sr.set_title("Success Rate by Group")
    ax_sr.legend(fontsize=8)

    # ── [1,0]  Episode timeline overlay ──────────────────────────────────────
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        eps     = [r["episode"]      for r in rows]
        rewards = np.array([r["total_reward"] for r in rows], dtype=float)
        rm      = _smooth(rewards, max(3, len(rows) // 10))
        ax_time.plot(eps, rm, color=_color(i), linewidth=2.0, label=label, zorder=3)
        ax_time.fill_between(eps, rewards, rm,
                             color=_color(i), alpha=0.08, linewidth=0)

    ax_time.set_xlabel("Episode")
    ax_time.set_ylabel("Total Reward")
    ax_time.set_title("Episode Timeline (rolling mean)")
    ax_time.legend(fontsize=8)

    # ── [1,1]  Reward histograms ──────────────────────────────────────────────
    all_rewards = [r["total_reward"] for rows in all_rows for r in rows]
    bins = np.linspace(min(all_rewards), max(all_rewards), 20)
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        rewards = [r["total_reward"] for r in rows]
        ax_hist.hist(rewards, bins=bins, color=_color(i), alpha=0.55,
                     label=label, zorder=3)
        # Median line
        med = float(np.median(rewards))
        ax_hist.axvline(med, color=_color(i), linewidth=1.5, linestyle="--")

    ax_hist.set_xlabel("Total Reward")
    ax_hist.set_ylabel("Episodes")
    ax_hist.set_title("Reward Distribution (dashed = median)")
    ax_hist.legend(fontsize=8)

    fig.suptitle(title or "Evaluation Comparison", fontsize=13, fontweight="bold")
    _save_or_show(fig, out_dir, "eval_episodes.png", plt)


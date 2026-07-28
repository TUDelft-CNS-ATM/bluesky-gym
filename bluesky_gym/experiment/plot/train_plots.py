"""
Training plotting suite.

Generates learning curves and comparative grids to analyze model performance 
during the training phase. Key features include rolling mean smoothing, 
standard deviation banding, and multi-run gap analysis for identifying 
policy consistency and convergence speeds.
"""

from __future__ import annotations

from .style import _apply_style, _color, _plt, _get_ticker, _smooth, _shade_std, _save_or_show

import math
import numpy as np
from typing import Optional

def plot_training_curves(
    labels:  list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    smooth:   int           = 1,
    title:    Optional[str] = None,
) -> None:
    """
    Plot mean reward (+ ±1 std band) and success rate curves for each run,
    with best-checkpoint markers and a convergence annotation.
    """
    _apply_style()
    plt = _plt()
    ticker = _get_ticker()


    has_success = any(
        not math.isnan(r["success_rate"])
        for rows in all_rows for r in rows
    )
    n_panels = 2 if has_success else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False
    )
    ax_rew = axes[0, 0]
    ax_sr  = axes[0, 1] if has_success else None

    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        if not rows:
            continue
        color = _color(i)
        steps   = np.array([r["timestep"]    for r in rows])
        rewards = np.array([r["mean_reward"] for r in rows], dtype=float)
        stds    = np.array([r["std_reward"]  for r in rows], dtype=float)

        rew_s = _smooth(rewards, smooth)

        # ── Reward panel ────────────────────────────────────────────────────
        ax_rew.plot(steps, rew_s, color=color, label=label, linewidth=2.0, zorder=3)
        _shade_std(ax_rew, steps, rewards, stds, color, smooth_w=smooth)

        # Best-checkpoint star marker
        best_idx = int(np.argmax(rew_s))
        ax_rew.scatter(
            steps[best_idx], rew_s[best_idx],
            color=color, s=80, zorder=5, marker="*",
            edgecolors="white", linewidths=0.5,
        )

        # ── Success rate panel ───────────────────────────────────────────────
        if ax_sr is not None:
            sr = np.array([r["success_rate"] for r in rows], dtype=float)
            valid = ~np.isnan(sr)
            if valid.any():
                sr_s = _smooth(sr[valid], smooth)
                ax_sr.plot(
                    steps[valid], sr_s,
                    color=color, label=label, linewidth=2.0, zorder=3,
                )
                ax_sr.fill_between(
                    steps[valid], 0, sr_s,
                    color=color, alpha=0.08, linewidth=0,
                )

    # Axes decoration — reward
    ax_rew.set_title(title or "Mean Reward")
    ax_rew.set_xlabel("Environment Steps")
    ax_rew.set_ylabel("Mean Reward")
    ax_rew.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
    )
    ax_rew.legend(title="Run", title_fontsize=8)

    # Axes decoration — success rate
    if ax_sr is not None:
        ax_sr.set_title("Success Rate")
        ax_sr.set_xlabel("Environment Steps")
        ax_sr.set_ylabel("Success Rate")
        ax_sr.set_ylim(0, 1.05)
        ax_sr.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax_sr.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
        )
        ax_sr.legend(title="Run", title_fontsize=8)

    _add_star_note(fig)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "training_curves.png", plt)


def plot_comparison_grid(
    labels:  list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    smooth:   int           = 1,
    title:    Optional[str] = None,
) -> None:
    """
    2 x 2 grid giving a richer view of the comparison:
      [0,0] Mean reward curves + std bands
      [0,1] Success rate curves
      [1,0] Reward std over time  (measures policy consistency)
      [1,1] Rolling gap: reward_A - reward_B  (2-run mode only; else skipped)
    """
    _apply_style()
    plt = _plt()
    ticker = _get_ticker()

    has_success = any(
        not math.isnan(r["success_rate"])
        for rows in all_rows for r in rows
    )
    two_runs = len(labels) == 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_rew, ax_sr   = axes[0, 0], axes[0, 1]
    ax_std, ax_diff = axes[1, 0], axes[1, 1]

    smoothed_rewards: list[tuple[np.ndarray, np.ndarray]] = []

    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        if not rows:
            smoothed_rewards.append((np.array([]), np.array([])))
            continue
        color = _color(i)
        steps   = np.array([r["timestep"]    for r in rows])
        rewards = np.array([r["mean_reward"] for r in rows], dtype=float)
        stds    = np.array([r["std_reward"]  for r in rows], dtype=float)
        sr      = np.array([r["success_rate"] for r in rows], dtype=float)

        rew_s = _smooth(rewards, smooth)
        smoothed_rewards.append((steps, rew_s))

        # [0,0] Reward + std band
        ax_rew.plot(steps, rew_s, color=color, label=label, linewidth=2.0, zorder=3)
        _shade_std(ax_rew, steps, rewards, stds, color, smooth_w=smooth)
        best_idx = int(np.argmax(rew_s))
        ax_rew.scatter(
            steps[best_idx], rew_s[best_idx],
            color=color, s=80, zorder=5, marker="*",
            edgecolors="white", linewidths=0.5,
        )

        # [0,1] Success rate
        if has_success:
            valid = ~np.isnan(sr)
            if valid.any():
                sr_s = _smooth(sr[valid], smooth)
                ax_sr.plot(steps[valid], sr_s, color=color, label=label, linewidth=2.0)
                ax_sr.fill_between(steps[valid], 0, sr_s, color=color, alpha=0.08, linewidth=0)

        # [1,0] Reward std over time
        valid_std = ~np.isnan(stds)
        if valid_std.any():
            std_s = _smooth(stds[valid_std], smooth)
            ax_std.plot(steps[valid_std], std_s, color=color, label=label, linewidth=1.8, linestyle="--")

    # [1,1] Rolling reward gap (run 0 - run 1)
    if two_runs:
        s0, r0 = smoothed_rewards[0]
        s1, r1 = smoothed_rewards[1]
        if len(s0) and len(s1):
            # Interpolate both to a common timestep grid
            common = np.intersect1d(s0, s1)
            if len(common) > 1:
                idx0 = np.isin(s0, common)
                idx1 = np.isin(s1, common)
                gap  = r0[idx0] - r1[idx1]
                gap_s = _smooth(gap, smooth)
                ax_diff.axhline(0, color="#AAAAAA", linewidth=1.0, linestyle=":")
                ax_diff.fill_between(
                    common, gap_s, 0,
                    where=gap_s >= 0, color=_color(0), alpha=0.25, linewidth=0,
                    label=f"{labels[0]} ahead",
                )
                ax_diff.fill_between(
                    common, gap_s, 0,
                    where=gap_s <= 0,  color=_color(1), alpha=0.25, linewidth=0,
                    label=f"{labels[1]} ahead",
                )
                ax_diff.plot(common, gap_s, color="#444444", linewidth=1.5)
                ax_diff.set_title(f"Reward Gap  ({labels[0]} - {labels[1]})")
                ax_diff.set_xlabel("Environment Steps")
                ax_diff.set_ylabel("Δ Mean Reward")
                ax_diff.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
                )
                ax_diff.legend(title="Run",fontsize=8)
            else:
                ax_diff.text(0.5, 0.5, "No overlapping timesteps",
                             ha="center", va="center", transform=ax_diff.transAxes,
                             color="#888888")
        else:
            ax_diff.set_visible(False)
    else:
        # More than 2 runs: reuse panel for a reward range band
        all_steps = sorted({r["timestep"] for rows in all_rows for r in rows})
        if all_steps:
            step_arr = np.array(all_steps)
            all_rew_at_step = []
            for rows in all_rows:
                idx_map = {r["timestep"]: r["mean_reward"] for r in rows}
                all_rew_at_step.append([idx_map.get(ts, np.nan) for ts in all_steps])
            mat = np.array(all_rew_at_step, dtype=float)
            lo  = np.nanmin(mat, axis=0)
            hi  = np.nanmax(mat, axis=0)
            mid = np.nanmean(mat, axis=0)
            ax_diff.fill_between(step_arr, lo, hi, color="#888888", alpha=0.2, label="min–max range")
            ax_diff.plot(step_arr, mid, color="#444444", linewidth=1.8, label="mean across runs")
            ax_diff.set_title("Reward Range Across Runs")
            ax_diff.set_xlabel("Environment Steps")
            ax_diff.set_ylabel("Mean Reward")
            ax_diff.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
            )
            ax_diff.legend(title="Run",fontsize=8)

    # Axes decoration
    fmt_x = ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))

    ax_rew.set_title("Mean Reward (± 1 std)")
    ax_rew.set_xlabel("Environment Steps")
    ax_rew.set_ylabel("Mean Reward")
    ax_rew.xaxis.set_major_formatter(fmt_x)
    ax_rew.legend(title="Run", title_fontsize=8)

    if has_success:
        ax_sr.set_title("Success Rate")
        ax_sr.set_xlabel("Environment Steps")
        ax_sr.set_ylabel("Success Rate")
        ax_sr.set_ylim(0, 1.05)
        ax_sr.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax_sr.xaxis.set_major_formatter(fmt_x)
        ax_sr.legend(title="Run", title_fontsize=8)
    else:
        ax_sr.set_visible(False)

    ax_std.set_title("Reward Std Dev Over Time  (policy consistency)")
    ax_std.set_xlabel("Environment Steps")
    ax_std.set_ylabel("Std Dev of Reward")
    ax_std.xaxis.set_major_formatter(fmt_x)
    ax_std.legend(title="Run", title_fontsize=8)

    _add_star_note(fig)
    fig.suptitle(title or "Training Comparison", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "training_comparison_grid.png", plt)


def _add_star_note(fig) -> None:
    fig.text(
        0.01, -0.01, "★ = best checkpoint",
        fontsize=7, color="#888888", ha="left",
    )
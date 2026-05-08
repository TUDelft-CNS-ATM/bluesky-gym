"""
Matplotlib styling and utility functions.

Provides a unified, publication-ready aesthetic for all framework plots. 
Includes custom color palettes, standard deviation shading wrappers, array 
smoothing functions, and file-saving helpers.
"""

from __future__ import annotations

import math
import os
from typing import Optional

def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return plt


def _mpl():
    import matplotlib as mpl
    return mpl

def _get_ticker():
    return _mpl().ticker # type: ignore

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]

# TODO: Add more colors
def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _apply_style() -> None:
    """Apply a clean, publication-ready style to all subsequent plots."""
    plt = _plt()
    plt.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "#F8F8F8",
        "axes.edgecolor":       "#CCCCCC",
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "grid.color":           "white",
        "grid.linewidth":       1.0,
        "grid.linestyle":       "-",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.color":          "#555555",
        "ytick.color":          "#555555",
        "axes.labelcolor":      "#333333",
        "axes.titleweight":     "bold",
        "axes.titlesize":       11,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
        "legend.fontsize":      8,
        "figure.dpi":           120,
        "savefig.dpi":          150,
        "savefig.bbox":         "tight",
        "font.family":          "sans-serif",
    })

def _smooth(vals, w: int):
    import numpy as np
    arr = np.array(vals, dtype=float)
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _shade_std(ax, steps, rewards, stds, color, smooth_w=1):
    """Draw a ±1 std shaded band around the reward curve."""
    import numpy as np
    valid = ~np.isnan(stds)
    if not valid.any():
        return
    s = steps[valid]
    r = _smooth(rewards[valid], smooth_w)
    d = _smooth(stds[valid],    smooth_w)
    ax.fill_between(s, r - d, r + d, color=color, alpha=0.15, linewidth=0)

def _save_or_show(fig, out_dir: Optional[str], filename: str, plt):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
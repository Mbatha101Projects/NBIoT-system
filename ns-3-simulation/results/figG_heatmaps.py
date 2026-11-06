#!/usr/bin/env python3
"""
figG_heatmaps.py — Heatmap figures for Group F

WHY HEATMAPS: the line-chart figures (figF_lifetime_grid, figF_delay_grid)
correctly show that T3412 dominates the outcome so strongly that T3324's
own effect becomes invisible on a shared axis (confirmed numerically: at
T3412=86400s, all 6 T3324 values span just 0.03% of the chart's range).

A heatmap sidesteps this entirely: colour, not vertical position, encodes
the value, so T3324's real (if small) horizontal gradient within each row
remains visible regardless of how large T3412's overall range is.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(SCRIPT_DIR, "results_group_f.csv"))
print(f"[INFO] Loaded {len(df)} rows")

FIXED_INTERVAL = 7200
CE_NAMES = {0: "CE-A (Good)", 1: "CE-B (Medium)", 2: "CE-C (Poor)"}


def heatmap_grid(metric_col, title, cbar_label, fname, log_metric=False, cmap="viridis"):
    sub = df[df["interval"] == FIXED_INTERVAL]
    ce_modes = sorted(sub["ceMode"].unique())
    t3412_vals = sorted(sub["T3412"].unique())
    t3324_vals = sorted(sub["T3324"].unique())

    fig, axes = plt.subplots(1, len(ce_modes), figsize=(5.5*len(ce_modes), 5.5))
    if len(ce_modes) == 1:
        axes = [axes]

    # Shared colour scale across all three panels so CE modes are
    # directly comparable to each other, not just internally consistent
    vals = sub[metric_col].values
    vmin, vmax = (np.log10(vals.min()), np.log10(vals.max())) if log_metric else (vals.min(), vals.max())

    im = None
    for ax, ce in zip(axes, ce_modes):
        ce_sub = sub[sub["ceMode"] == ce]
        grid = ce_sub.pivot(index="T3324", columns="T3412", values=metric_col)
        grid = grid.reindex(index=t3324_vals, columns=t3412_vals)
        plot_data = np.log10(grid.values) if log_metric else grid.values
        im = ax.imshow(plot_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(t3412_vals)))
        ax.set_xticklabels([f"{t/3600:.1f}h" if t >= 3600 else f"{t}s" for t in t3412_vals], rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(t3324_vals)))
        ax.set_yticklabels([f"{t}s" for t in t3324_vals], fontsize=11)
        ax.set_xlabel("T3412", fontsize=13)
        ax.set_title(CE_NAMES.get(ce, f"CE={ce}"), fontsize=14)
        # Annotate each cell with its actual value -- this is what makes
        # T3324's real, small gradient checkable directly, not just implied
        for i in range(len(t3324_vals)):
            for j in range(len(t3412_vals)):
                v = grid.values[i, j]
                if not np.isnan(v):
                    txt = f"{v:.1f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                            color="white" if (plot_data[i,j]-vmin)/(vmax-vmin) < 0.6 else "black")
    axes[0].set_ylabel("T3324", fontsize=13)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label=cbar_label)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(cbar_label, fontsize=12)
    fig.suptitle(f"{title} (traffic interval = {FIXED_INTERVAL}s)")
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


heatmap_grid("lifetime_yr", "Group F: Battery Lifetime Heatmap", "log10(lifetime, years)",
             "figG_lifetime_heatmap.png", log_metric=True, cmap="viridis")

heatmap_grid("latency_s", "Group F: Expected Buffering Delay Heatmap", "log10(delay, seconds)",
             "figG_delay_heatmap.png", log_metric=True, cmap="magma")

print("[DONE]")

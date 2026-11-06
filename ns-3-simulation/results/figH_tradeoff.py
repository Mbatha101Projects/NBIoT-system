#!/usr/bin/env python3
"""
figH_tradeoff.py — The energy-vs-delay tradeoff frontier

WHY THIS FIGURE: every other figure shows lifetime OR delay against a
timer value. This one shows them against EACH OTHER directly -- for
every one of the 42 (T3412, T3324) combinations, where does it land on
the lifetime-vs-delay plane? This is the actual tradeoff this entire
project studies, made explicit as a single picture: points toward the
top-left are "good" (long life, low delay); points toward the
bottom-right are the opposite. Points are coloured by T3412 specifically,
since that's the dominant driver established in every other figure --
this figure exists to show WHERE on the tradeoff curve each T3412 choice
lands, with T3324 visible as the minor scatter within each colour cluster.
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
FIXED_INTERVAL = 7200
CE_NAMES = {0: "CE-A (Good)", 1: "CE-B (Medium)", 2: "CE-C (Poor)"}

sub = df[df["interval"] == FIXED_INTERVAL]
ce_modes = sorted(sub["ceMode"].unique())
t3412_vals = sorted(sub["T3412"].unique())
cmap = plt.get_cmap("plasma")
colors = {t: cmap(i/max(1, len(t3412_vals)-1)) for i, t in enumerate(t3412_vals)}

fig, axes = plt.subplots(1, len(ce_modes), figsize=(6*len(ce_modes), 5.5), sharex=True, sharey=True)

for ax, ce in zip(axes, ce_modes):
    ce_sub = sub[sub["ceMode"] == ce]
    for t3412 in t3412_vals:
        line = ce_sub[ce_sub["T3412"] == t3412].sort_values("T3324")
        ax.scatter(line["latency_s"]/3600.0, line["lifetime_yr"],
                  color=colors[t3412], s=45, edgecolors="white", linewidths=0.6,
                  label=f"T3412={t3412/3600:.2f}h" if t3412 >= 3600 else f"T3412={t3412}s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expected Delay (hours, log scale)")
    ax.set_title(CE_NAMES.get(ce, f"CE={ce}"))
    ax.grid(True, alpha=0.3, which="both")

axes[0].set_ylabel("Battery Lifetime (years, log scale)")
axes[-1].legend(title="T3412", loc="lower right", fontsize=7, ncol=1)
fig.suptitle(f"Group F: The Energy-Delay Tradeoff Frontier — every (T3412,T3324) combination\n"
            f"(traffic interval = {FIXED_INTERVAL}s; colour = T3412, scatter within a colour = T3324's minor effect)")
fig.tight_layout()
path = os.path.join(FIG_DIR, "figH_tradeoff_frontier.png")
fig.savefig(path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"[SAVED] {path}")

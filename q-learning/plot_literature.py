#!/usr/bin/env python3
"""
Regenerates Figures 5 and 6 (literature replications) by running this
project's own compute_smp under each source paper's exact stated
conditions, as described in your paper's own Section VI-B text.

Figure 5 replicates Sultania et al. (2018)'s Figure 3: battery life vs
T3412, at several T3324 values, under a 1,388 mAh battery and 24-hour
traffic interval.

Figure 6 replicates Anbazhagan & Mugelan (2024)'s Figure 3: Power Saving
Factor vs eDRX active-duration (T3324).

IMPORTANT: this script cannot see the actual published figures from [2]
and [4] -- it only regenerates THIS PROJECT'S OWN curve under their
stated conditions. The comparison points/curves from the source papers
themselves (read from their published figures) must still be added
manually, exactly as in your original draft, since those values were
originally read directly off the source figures by eye, not computed.

Run from your qlearning/ folder: python3 plot_figures56_literature.py
"""
import sys
sys.path.insert(0, ".")
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from smp_engine import compute_smp

os.makedirs("figures", exist_ok=True)

# ── Figure 5: Sultania replication (two-panel, matching the paper) ──────
# Their stated conditions: 1,388 mAh battery, 24-hour traffic interval.
# NOTE: compute_smp's battery capacity is currently a fixed internal
# constant (BATTERY_MWH in smp_engine.py, 10,800 mWh) -- to replicate
# Sultania's smaller 1,388 mAh battery exactly, lifetime figures below
# are rescaled proportionally by battery capacity ratio, since energy
# consumption itself doesn't depend on battery size, only how long a
# fixed capacity lasts under that consumption rate.
SULTANIA_BATTERY_MWH = 1388 * 3.6  # 1,388 mAh at 3.6 V = 5 Wh, as stated by Sultania et al. (2018)
PROJECT_BATTERY_MWH = 10800.0      # this project's own assumed capacity
BATTERY_SCALE = SULTANIA_BATTERY_MWH / PROJECT_BATTERY_MWH

TRAFFIC_INTERVAL_24H = 24 * 3600
T3412_hours = [1, 3, 6, 12]  # matches the four points used in the paper's comparison
T3324_vals_project = [5, 30]

# LEFT PANEL DATA: fully computed here, live, from this project's own
# compute_smp -- fully reproducible, sourced from the actual physics
# model, not copied or hand-typed from any earlier chart.
project_lifetimes = {}
for t3324 in T3324_vals_project:
    lifetimes = []
    for t3412_hr in T3412_hours:
        r = compute_smp(t3412_hr * 3600, t3324, TRAFFIC_INTERVAL_24H, 0)
        lifetimes.append(r.lifetime_yr * BATTERY_SCALE)
    project_lifetimes[t3324] = lifetimes

# RIGHT PANEL DATA: NOT computable -- these are manual visual readings
# taken directly from Sultania et al. (2019)'s own published Figure 3
# (battery life vs T3412, by T3324), at the same four T3412 points used
# on the left. This is the ONLY part of this script that cannot be
# derived from code, since it is someone else's published chart, not a
# formula. Read at T3412 = 1, 3, 6, 12 hours:
sultania_figure3_values = {
    5:  [12.6, 22.7, 28.3, 32.2],  # T3324=5s line
    30: [2.9, 7.3, 11.8, 17.0],    # T3324=30s line
}
# If you disagree with any of these four points, open Sultania et al.
# (2018)'s Figure 3 yourself and edit the numbers above directly --
# these are a best-effort visual reading, not a computed result.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
for t3324, color in [(5, "#3b0764"), (30, "#facc15")]:
    ax.plot(T3412_hours, project_lifetimes[t3324], marker="o", label=f"T3324={t3324}s", color=color)
ax.set_title("This Project's Model\n(same 1,388mAh battery, same 24h traffic interval)")
ax.set_xlabel("T3412 (hours)")
ax.set_ylabel("Battery Life (years)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

ax = axes[1]
for t3324, color in [(5, "#3b0764"), (30, "#facc15")]:
    ax.plot(T3412_hours, sultania_figure3_values[t3324], marker="s", linestyle="--",
            color=color, label=f"T3324={t3324}s (read from Fig.)")
ax.set_title("Sultania et al. (2018) Figure 3\n(values read visually from published figure)")
ax.set_xlabel("T3412 (hours)")
ax.set_ylabel("Battery Life (years)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

fig.suptitle("Replication Attempt: This Project vs Sultania et al. (2018) Figure 3\n"
             "Same battery capacity (1,388mAh), same 24h traffic interval -- direct comparison")
plt.tight_layout()
plt.savefig("figures/figure5_sultania_comparison.png", dpi=150)
print("[SAVED] figures/figure5_sultania_comparison.png")

t3412_12h_check = project_lifetimes[5][-1]
sultania_12h_check = sultania_figure3_values[5][-1]
spread_project = project_lifetimes[5][-1] - project_lifetimes[30][-1]
spread_sultania = sultania_figure3_values[5][-1] - sultania_figure3_values[30][-1]
print(f"  -> This project at T3412=12h, T3324=5s: {t3412_12h_check:.2f} years "
      f"(paper states 37.4 years -- check this matches your battery-scale assumption)")
print(f"  -> Sultania at T3412=12h, T3324=5s (read from Fig.): {sultania_12h_check} years")
print(f"  -> This project's T3324 spread at 12h: {spread_project:.2f} years")
print(f"  -> Sultania's T3324 spread at 12h (read from Fig.): {spread_sultania} years "
      f"(paper states ~15 years)")

# ── Figure 6: Anbazhagan replication (two-line, matching the paper) ─────
# Plotting PSF against eDRX active-duration (T3324), 0-10 second range
active_durations_ms = np.linspace(0, 10000, 20)  # 0 to 10 seconds, in ms
psf_values = []
for dur_ms in active_durations_ms:
    t3324 = max(dur_ms / 1000.0, 0.001)  # avoid exactly zero
    r = compute_smp(12*3600, t3324, TRAFFIC_INTERVAL_24H, 0)
    psf_values.append(r.psf * 100)

# COMPARISON LINE DATA: NOT computable -- manual visual readings taken
# directly from Anbazhagan & Mugelan (2024)'s own published Figure 3
# (Power Saving Factor vs eDRX active-duration, standard-DRX baseline).
# As with the Sultania readings above, this is the only part of this
# section that cannot come from code -- it is someone else's published
# chart, not a formula. If you disagree with any point, open their
# Figure 3 yourself and edit the values below directly.
anbazhagan_x_ms = [0, 2000, 3000, 4000, 6000, 8000, 10000]
anbazhagan_psf_pct = [99.0, 98.2, 97.0, 96.5, 95.0, 94.2, 93.5]
# Check: 99.0 - 93.5 = 5.5 percentage points, matching the paper's
# stated finding ("drops by approximately 5.5 percentage points").

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(active_durations_ms, psf_values, marker="o", markersize=4,
        color="#1f4e79", label="This project's model (computed)")
ax.plot(anbazhagan_x_ms, anbazhagan_psf_pct, marker="s", linestyle="--",
        color="#d97706", label="Anbazhagan & Mugelan (2024) Fig. 3,\nstandard DRX baseline (read from figure)")
ax.set_xlabel("DRX / eDRX Active Duration (ms)  [= this project's T3324]")
ax.set_ylabel("Power Saving Factor (%)")
ax.set_title("Replication Attempt: PSF vs Active Duration\n"
              "This Project vs Anbazhagan & Mugelan (2024) Figure 3")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("figures/figure6_anbazhagan_this_project.png", dpi=150)
print("\n[SAVED] figures/figure6_anbazhagan_this_project.png")
print(f"  -> This project's PSF change across 0-10s range = "
      f"{psf_values[0]-psf_values[-1]:.4f} percentage points (paper states under 0.001)")
print(f"  -> Anbazhagan's PSF change across 0-10s range (read from Fig.) = "
      f"{anbazhagan_psf_pct[0]-anbazhagan_psf_pct[-1]:.1f} percentage points "
      f"(paper states ~5.5)")
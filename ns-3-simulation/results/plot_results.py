#!/usr/bin/env python3
"""
plot_results.py  —  NB-IoT SMP Benchmark Figure Generator  (v4)

University of Zululand | Honours CS | 2026
Student    : Mawande Mbatha (202396115)
Supervisor : Dr Paul Tarwireyi

FIGURES PRODUCED
────────────────────────────────────────────────────────────────────────────
figA1_T3412_sweep.png      Group A1: Energy, Lifetime, PSF vs T3412 (3-panel)
figA2_T3324_sweep.png      Group A2: Energy, Lifetime, PSF vs T3324 (3-panel)
figB_ce_comparison.png     Group B:  Lifetime & PSF bar chart by CE mode
figC_traffic_profiles.png  Group C:  TC1-TC5 side-by-side lifetime + PSF
figD_tau_sweep.png         Group D:  Energy/lifetime "cliff" at TAU onset
figX_summary_scatter.png   All rows: p_avg_mw vs lifetime coloured by group
figF_lifetime_grid.png     Group F:  Lifetime vs T3412, 6 T3324 lines, 3 CE panels
figF_delay_grid.png        Group F:  Delay vs T3412, 6 T3324 lines, 3 CE panels
(figA1-figX only generate if results.csv is present;
 figF_* only generate if results_group_f.csv is present — both are optional)

USAGE — WINDOWS / VS CODE
──────────────────────────
Put this script in the SAME FOLDER as your CSV files (results.csv,
results_group_f.csv, etc — whichever ones you have). No specific folder
structure or naming is required beyond that. Then, in VS Code's terminal
(View -> Terminal), from that folder, run:

    pip install pandas matplotlib numpy
    python plot_results.py

Figures are written to a "figures" subfolder created next to the script.
If a particular CSV isn't present in the folder, that CSV's figures are
simply skipped with a message — the script does not require ALL of them
to be present at once.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — simple and portable: everything relative to THIS script's own
# location, wherever that happens to be (Windows, VS Code, anywhere).
# No assumption about ns-3-dev, analysis/ subfolders, or any fixed layout.
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH        = os.path.join(SCRIPT_DIR, "results.csv")
CSV_GROUP_F_PATH = os.path.join(SCRIPT_DIR, "results_group_f.csv")

HAVE_MAIN    = os.path.exists(CSV_PATH)
HAVE_GROUP_F = os.path.exists(CSV_GROUP_F_PATH)

if not HAVE_MAIN and not HAVE_GROUP_F:
    sys.exit(f"[ERROR] Neither results.csv nor results_group_f.csv found in "
             f"{SCRIPT_DIR}\nPlace at least one of them in the same folder "
             f"as this script.")

if HAVE_MAIN:
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded results.csv: {len(df)} rows")
else:
    df = None
    print(f"[INFO] results.csv not found — skipping figA1-figX (that's fine "
          f"if you only wanted the Group F figures)")

if HAVE_GROUP_F:
    df_f = pd.read_csv(CSV_GROUP_F_PATH)
    print(f"[INFO] Loaded results_group_f.csv: {len(df_f)} rows")
else:
    df_f = None
    print(f"[INFO] results_group_f.csv not found — skipping figF_* figures")

# Compatibility for v2 fields
if HAVE_MAIN:
    if "tauFired" not in df.columns: df["tauFired"] = 0
    if "p_avg_mw" not in df.columns: df["p_avg_mw"] = np.nan
    if "group"    not in df.columns: df["group"]    = "?"

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED SMP ANALYTICAL ENGINE (mirrors nbiot-smp-benchmark.cc v3)
# Used for overlay curves — must stay in sync with the CC file.
#
# v3 changes vs v2 (see nbiot-smp-benchmark.cc header for full citations):
#   1. P_ACT raised from flat 0.015 mW to cycle-weighted composite 0.267 mW.
#      Source: Sørensen et al. (2022) Table X paging-burst components
#      (E_onDur=0.326mJ, t_onDur=1.445ms, P_sleep_eDRX=0.0122mW) weighted
#      over a 1.28s paging cycle (Anbazhagan & Mugelan 2024, Table 1,
#      shortest DRX cycle). Validated against Sultania et al. (2019)
#      Fig.3 and Anbazhagan Fig.5 — see CC file header for full discussion
#      of the remaining cross-device gap vs Sultania's SARA-N2-based
#      EIdle=21.6mW, which is NOT matched exactly (different chipset).
#   2. SYNC is no longer scaled by tx_frac. A device exiting PSM performs
#      a full re-sync every T3412 expiry regardless of pending data
#      (Anbazhagan & Mugelan 2024, Algorithm 1, line 32-34). v2's scaling
#      of SYNC by tx_frac caused a ~80-92yr lifetime overestimate when
#      checked against Sultania (2019) Fig.3 conditions (T3412 << interval).
# ─────────────────────────────────────────────────────────────────────────────
P_PSM    = 0.0095   # 0.0108 mW — SIM7070G
P_SYNC   = 72.73         # mW — Sørensen Table X
P_RRC    = 21.337        # mW — cDRX idle
P_TX     = 742.858       # mW — Sørensen Table X
P_TXGAP  = 153.6         # mW — TX gap
P_RX     = 21.337        # mW — cDRX idle
P_ACT    = 0.267         # mW — cycle-weighted eDRX composite (v3, was 0.015)
BATT     = 10800.0       # mWh
P_REF    = P_TX          # always-on reference

G_SYNC   = 0.88623       # Γ(1.5)
G_TX     = 0.90274       # Γ(5/3)
L_SYNC   = 2.483         # s λ_base SYNC
L_TX     = 0.0554        # s λ_base TX
P_RETX   = 0.10
RRC_M    = 1.0
RX_F     = 0.015         # s

CE_TX_S  = [1.0, 16.0, 128.0]
CE_SY_S  = [1.0,  2.5,   6.0]
E_TAU_B  = 210.0         # mJ at CE-A

def _tx_eff(ce):
    t   = (L_TX * CE_TX_S[ce] * G_TX) / (1 - P_RETX)
    ng  = int(t / 0.256)
    tot = t + ng * 0.040
    e   = t * P_TX + ng * 0.040 * P_TXGAP
    return e / tot if tot > 0 else P_TX

def smp(T3412, T3324, interval, ce=0):
    tf = min(1.0, T3412 / interval)
    E  = dict(
        PSM    = T3412,
        SYNC   = L_SYNC * CE_SY_S[ce] * G_SYNC,        # v3: NOT scaled by tf
        RRC    = RRC_M * tf,
        TX     = (L_TX * CE_TX_S[ce] * G_TX) / (1-P_RETX) * tf,
        RX     = RX_F * tf,
        ACTIVE = T3324,
    )
    Pw = dict(PSM=P_PSM, SYNC=P_SYNC, RRC=P_RRC,
              TX=_tx_eff(ce), RX=P_RX, ACTIVE=P_ACT)
    sE   = sum(E.values())
    pi   = {s: E[s]/sE for s in E}
    pavg = sum(pi[s]*Pw[s] for s in pi)
    tau  = interval > T3412
    if tau:
        pavg += (E_TAU_B * CE_SY_S[ce]) / (T3412 * 1000.0)
    lt  = (BATT / pavg) / 8760.0 if pavg > 0 else 999.0
    psf = 1.0 - pavg / P_REF

    # v4: expected PSM/eDRX buffering delay (Sultania et al. 2019, Eq.30).
    # Mirrors nbiot-smp-benchmark.cc Section 7b exactly — must stay in sync.
    # Only PSM and ACTIVE are sleep-capable states a packet can be
    # "buffered" during; expected wait within a sleep window of duration T
    # is T/2 (uniform arrival assumption).
    numerator   = pi["PSM"] * (E["PSM"] / 2.0) + pi["ACTIVE"] * (E["ACTIVE"] / 2.0)
    denominator = pi["PSM"] + pi["ACTIVE"]
    delay_s     = (numerator / denominator) if denominator > 0 else 0.0

    return pavg, lt, psf * 100, tau, delay_s

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
CE_COL = {0:"#1565C0", 1:"#E65100", 2:"#B71C1C"}
CE_LAB = {0:"CE-A (Good, 50 m)", 1:"CE-B (Medium, 200 m)", 2:"CE-C (Poor, 500 m)"}
GRP_COL = {"A1":"#1565C0","A2":"#2E7D32","B":"#E65100","C":"#6A1B9A","D":"#BF360C"}

plt.rcParams.update({
    "font.family":"DejaVu Sans","axes.titlesize":12,"axes.labelsize":10,
    "legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
    "figure.dpi":150, "axes.grid":True, "grid.alpha":0.3,
})

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")

def annotate_tau(ax, sub, x_col, y_col):
    """Mark points where TAU fired with a red X."""
    tau_rows = sub[sub["tauFired"] == 1]
    if not tau_rows.empty:
        ax.scatter(tau_rows[x_col], tau_rows[y_col],
                   marker="x", color="crimson", s=90, zorder=6,
                   linewidths=2, label="TAU fired")

# ═════════════════════════════════════════════════════════════════════════════
# FIG A1 — T3412 SWEEP  (Group A1, rows 1-25)
#   Three stacked panels: (1) Energy per cycle, (2) Lifetime, (3) PSF
#   X-axis: T3412 in hours, log scale
#   Analytical curve overlaid on NS-3 scatter points
# ═════════════════════════════════════════════════════════════════════════════
def figA1():
    a1 = df[df["group"]=="A1"].copy().sort_values("T3412")
    if a1.empty:
        print("[WARN] No Group A1 data"); return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        "Fig A1: T3412 (PSM Sleep Duration) Sweep — CE-A, T3324=20s fixed\n"
        "25 configurations, T3412 from 60s to 86400s (log scale)\n"
        "Analytical curve (lines) vs NS-3 simulation (markers)",
        fontsize=11, y=1.01
    )

    # Analytical overlay
    T_range = np.logspace(np.log10(60), np.log10(86400), 400)
    E_an, L_an, P_an = [], [], []
    for t in T_range:
        pavg, lt, psf, _, _ = smp(T3412=t, T3324=20, interval=t, ce=0)
        # energy per cycle = pavg [mW] × cycle_length [s] / 1000 [→ mJ]
        cycle_s = t + 2.2 + 1.0 + (L_TX*G_TX/(1-P_RETX)) + RX_F + 20
        E_an.append(pavg * cycle_s / 1000 * 1000)  # mJ
        L_an.append(lt)
        P_an.append(psf)

    ax0, ax1, ax2 = axes

    # Panel 1 — Energy per cycle
    ax0.plot(T_range/3600, E_an, color="#1565C0", lw=2, label="Analytical")
    ax0.scatter(a1["T3412"]/3600, a1["energy_mWh"]*3600,  # mWh -> mJ (1 mWh = 3600 mJ)
                color="#1565C0", s=60, zorder=5, edgecolors="white",
                linewidths=0.8, label="NS-3 result")
    annotate_tau(ax0, a1.assign(energy_mJ=a1["energy_mWh"]*3600), "T3412", "energy_mJ")
    ax0.set_ylabel("Energy per Cycle (mJ)")
    ax0.set_title("Panel 1 — Energy consumed per transmit cycle")
    ax0.legend()
    ax0.set_xscale("log")

    # Panel 2 — Lifetime
    ax1.plot(T_range/3600, L_an, color="#2E7D32", lw=2, label="Analytical")
    ax1.scatter(a1["T3412"]/3600, a1["lifetime_yr"],
                color="#2E7D32", s=60, zorder=5, edgecolors="white", linewidths=0.8,
                label="NS-3 result")
    annotate_tau(ax1, a1, "T3412", "lifetime_yr")
    ax1.axhline(10, color="grey", ls="--", lw=1, alpha=0.7, label="3GPP 10-yr target")
    ax1.set_ylabel("Battery Lifetime (years)")
    ax1.set_title("Panel 2 — Battery lifetime vs sleep duration")
    ax1.legend()

    # Panel 3 — PSF
    ax2.plot(T_range/3600, P_an, color="#E65100", lw=2, label="Analytical")
    ax2.scatter(a1["T3412"]/3600, a1["PSF_pct"],
                color="#E65100", s=60, zorder=5, edgecolors="white", linewidths=0.8,
                label="NS-3 result")
    annotate_tau(ax2, a1, "T3412", "PSF_pct")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Power Saving Factor (%)")
    ax2.set_title("Panel 3 — PSF (negative = PSM worse than always-on)")
    ax2.set_xlabel("T3412 PSM Sleep Duration (hours, log scale)")
    ax2.legend()

    for ax in axes:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2g}"))

    fig.tight_layout()
    save(fig, "figA1_T3412_sweep.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG A2 — T3324 SWEEP  (Group A2, rows 26-50)
#   Same three-panel layout as A1 but X-axis is T3324
#   KEY RESULT: panels should be nearly flat — proves ACTIVE ≈ 0 mW (v2)
#   If v1 model were used, energy would rise steeply — clear before/after
# ═════════════════════════════════════════════════════════════════════════════
def figA2():
    a2 = df[df["group"]=="A2"].copy().sort_values("T3324")
    if a2.empty:
        print("[WARN] No Group A2 data"); return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        "Fig A2: T3324 (eDRX/Active Window) Sweep — CE-A, T3412=3600s fixed\n"
        "25 configurations, T3324 from 5s to 3595s\n"
        "EXPECTED: near-flat curves — T3324 barely affects energy in corrected model\n"
        "(v1 model predicted strong rise: ACTIVE at 12.6 mW was 1000× too high)",
        fontsize=10, y=1.02
    )

    # Analytical overlay
    T3324_range = np.linspace(5, 3595, 400)
    E_an, L_an, P_an = [], [], []
    for t3324 in T3324_range:
        pavg, lt, psf, _, _ = smp(T3412=3600, T3324=t3324, interval=3600, ce=0)
        cycle_s = 3600 + 2.2 + 1.0 + (L_TX*G_TX/(1-P_RETX)) + RX_F + t3324
        E_an.append(pavg * cycle_s / 1000 * 1000)
        L_an.append(lt)
        P_an.append(psf)

    ax0, ax1, ax2 = axes

    # v1 reference line for contrast
    # In v1: P_ACTIVE=12.6 mW — energy rises linearly with T3324
    # Show it as a dashed "what v1 predicted" line
    E_v1 = []
    for t3324 in T3324_range:
        # v1 power values
        pi_act_v1 = t3324 / (3600 + 2.2 + 1.0 + 0.95 + 0.5 + t3324)
        # rough contribution estimate
        pavg_v1_act = pi_act_v1 * 12.6   # ACTIVE contribution only
        E_v1.append(pavg_v1_act * (3600 + t3324) / 1000 * 1000)

    ax0.plot(T3324_range, E_an, color="#1565C0", lw=2, label="v2 Analytical (corrected)")
    ax0.scatter(a2["T3324"], a2["energy_mWh"]*3600,  # mWh -> mJ
                color="#1565C0", s=60, zorder=5, edgecolors="white",
                linewidths=0.8, label="NS-3 result")
    ax0.set_ylabel("Energy per Cycle (mJ)")
    ax0.set_title("Panel 1 — Energy vs T3324\n(near-flat = v2 ACTIVE correction confirmed)")
    ax0.legend()

    ax1.plot(T3324_range, L_an, color="#2E7D32", lw=2, label="v2 Analytical")
    ax1.scatter(a2["T3324"], a2["lifetime_yr"],
                color="#2E7D32", s=60, zorder=5, edgecolors="white", linewidths=0.8,
                label="NS-3 result")
    ax1.axhline(10, color="grey", ls="--", lw=1, alpha=0.7, label="3GPP 10-yr target")
    ax1.set_ylabel("Battery Lifetime (years)")
    ax1.set_title("Panel 2 — Lifetime vs T3324 (near-flat = T3324 choice barely matters)")
    ax1.legend()

    ax2.plot(T3324_range, P_an, color="#E65100", lw=2, label="v2 Analytical")
    ax2.scatter(a2["T3324"], a2["PSF_pct"],
                color="#E65100", s=60, zorder=5, edgecolors="white", linewidths=0.8,
                label="NS-3 result")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Power Saving Factor (%)")
    ax2.set_title("Panel 3 — PSF vs T3324")
    ax2.set_xlabel("T3324 Active/eDRX Window Duration (seconds)")
    ax2.legend()

    fig.tight_layout()
    save(fig, "figA2_T3324_sweep.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG B — CE MODE COMPARISON  (Group B, 9 configs)
# ═════════════════════════════════════════════════════════════════════════════
def figB():
    b = df[df["group"]=="B"].copy()
    if b.empty:
        print("[WARN] No Group B data"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fig B: CE Mode Comparison — T3412 ∈ {10min, 1h, 4h} × CE-A/B/C",
                 fontsize=12)

    t3412_vals = sorted(b["T3412"].unique())
    t_labels   = {600:"T3412=10min", 3600:"T3412=1h", 14400:"T3412=4h"}
    x          = np.arange(len(t3412_vals))
    w          = 0.25
    offsets    = [-w, 0, w]

    for i, ce in enumerate([0,1,2]):
        sub = b[b["ceMode"]==ce].set_index("T3412")
        lts  = [float(sub.loc[t, "lifetime_yr"]) if t in sub.index else 0 for t in t3412_vals]
        psfs = [float(sub.loc[t, "PSF_pct"])     if t in sub.index else 0 for t in t3412_vals]
        bars1 = ax1.bar(x + offsets[i], lts,  w, color=CE_COL[ce], alpha=0.85, label=CE_LAB[ce])
        bars2 = ax2.bar(x + offsets[i], psfs, w, color=CE_COL[ce], alpha=0.85)
        for bar, v in zip(bars1, lts):
            if v > 0:
                ax1.text(bar.get_x()+bar.get_width()/2, v+0.1,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars2, psfs):
            ypos = v+0.5 if v >= 0 else v-3
            ax2.text(bar.get_x()+bar.get_width()/2, ypos,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x); ax1.set_xticklabels([t_labels[t] for t in t3412_vals])
    ax1.set_ylabel("Lifetime (years)"); ax1.set_title("Battery Lifetime")
    ax1.axhline(10, color="grey", ls="--", lw=1, alpha=0.7, label="3GPP 10-yr")
    ax1.legend()

    ax2.set_xticks(x); ax2.set_xticklabels([t_labels[t] for t in t3412_vals])
    ax2.set_ylabel("PSF (%)"); ax2.set_title("Power Saving Factor")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "figB_ce_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG C — TRAFFIC PROFILES  (Group C)
# ═════════════════════════════════════════════════════════════════════════════
def figC():
    c = df[df["group"]=="C"].copy()
    if c.empty:
        print("[WARN] No Group C data"); return

    order  = ["TC1_SmartMeter","TC2_EnvSensor","TC3a_Agri_dry",
              "TC3b_Agri_irr","TC4_Wildlife_CEC","TC5_Industrial"]
    labels = ["TC1\nSmart Meter\n(CE-A 2h)", "TC2\nEnv Sensor\n(CE-A 1h)",
              "TC3a\nAgri Dry\n(CE-B 2h)", "TC3b\nAgri Irrig\n(CE-B 15m)",
              "TC4\nWildlife\n(CE-C 5m)", "TC5\nIndustrial\n(CE-A 30m)"]
    ce_cols = [CE_COL[0],CE_COL[0],CE_COL[1],CE_COL[1],CE_COL[2],CE_COL[0]]

    lts, psfs, taus = [], [], []
    for lbl in order:
        row = c[c["label"]==lbl]
        lts.append(float(row["lifetime_yr"].iloc[0]) if not row.empty else 0)
        psfs.append(float(row["PSF_pct"].iloc[0])    if not row.empty else 0)
        taus.append(int(row["tauFired"].iloc[0])      if not row.empty else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fig C: Traffic Profile Comparison — TC1 to TC5", fontsize=12)

    x = np.arange(len(order)); w = 0.6
    b1 = ax1.bar(x, lts, w, color=ce_cols, alpha=0.85)
    for bar, lt, tau in zip(b1, lts, taus):
        ax1.text(bar.get_x()+bar.get_width()/2, max(lt,0)+0.05,
                 f"{lt:.2f}", ha="center", va="bottom", fontsize=8)
        if tau:
            ax1.text(bar.get_x()+bar.get_width()/2, max(lt,0)+0.2,
                     "TAU", ha="center", va="bottom", fontsize=7,
                     color="crimson", fontweight="bold")
    ax1.axhline(10, color="grey", ls="--", lw=1, alpha=0.7, label="3GPP 10-yr")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Lifetime (years)"); ax1.set_title("Battery Lifetime")
    ax1.legend()

    b2 = ax2.bar(x, psfs, w, color=ce_cols, alpha=0.85)
    for bar, psf in zip(b2, psfs):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 psf + (0.5 if psf >= 0 else -3),
                 f"{psf:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("PSF (%)"); ax2.set_title("Power Saving Factor\n(negative = PSM worse than always-on)")

    legend_patches = [
        mpatches.Patch(color=CE_COL[0], label="CE-A"),
        mpatches.Patch(color=CE_COL[1], label="CE-B"),
        mpatches.Patch(color=CE_COL[2], label="CE-C"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5,-0.05), fontsize=9)
    fig.tight_layout()
    save(fig, "figC_traffic_profiles.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG D — TAU THRESHOLD SWEEP  (Group D)
# Shows the energy/lifetime "cliff" as T3412 crosses the interval boundary
# ═════════════════════════════════════════════════════════════════════════════
def figD():
    d = df[df["group"]=="D"].copy().sort_values("T3412")
    if d.empty:
        print("[WARN] No Group D data"); return

    d = d.assign(energy_mJ=d["energy_mWh"] * 3600)  # mWh -> mJ for this figure's units

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Fig D: TAU Threshold Effect — interval fixed at 7200s, T3412 varies\n"
        "When T3412 < interval: TAU fires every T3412 seconds (energy cliff)\n"
        "When T3412 ≥ interval: no TAU — energy drops dramatically",
        fontsize=11
    )

    t_vals = d["T3412"].values
    colors = ["crimson" if tau else "#1565C0"
              for tau in d["tauFired"].values]

    metrics = [
        ("energy_mJ",   "Energy per Cycle (mJ)", "Panel 1 — Energy"),
        ("lifetime_yr", "Lifetime (years)",        "Panel 2 — Lifetime"),
        ("PSF_pct",     "PSF (%)",                 "Panel 3 — PSF"),
    ]

    for ax, (col, ylabel, title) in zip(axes, metrics):
        bars = ax.bar(range(len(t_vals)), d[col].values,
                      color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(t_vals)))
        ax.set_xticklabels(
            [f"T3412\n={v}s\n{'TAU' if c=='crimson' else 'no TAU'}"
             for v,c in zip(t_vals, colors)],
            fontsize=8
        )
        ax.set_ylabel(ylabel); ax.set_title(title)
        if col == "lifetime_yr":
            ax.axhline(10, color="grey", ls="--", lw=1, alpha=0.7)
        if col == "PSF_pct":
            ax.axhline(0, color="black", lw=0.8)
        for bar, v in zip(bars, d[col].values):
            ax.text(bar.get_x()+bar.get_width()/2, max(v,0)+abs(d[col].max())*0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    legend_patches = [
        mpatches.Patch(color="crimson",   label="TAU fired (T3412 < 7200s)"),
        mpatches.Patch(color="#1565C0",   label="No TAU (T3412 ≥ 7200s)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5,-0.04), fontsize=9)
    fig.tight_layout()
    save(fig, "figD_tau_sweep.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG E — LITERATURE COMPARISON: Lifetime vs T3412, 6 lines for T3324
#   X-axis : T3412 in hours, ticks at 0, 3, 6, 9, 12, 15
#   Y-axis : Battery lifetime in years, ticks at 5, 10, 15, 20, 25
#   6 lines, one per T3324 value: 5s, 10s, 15s, 20s, 25s, 30s
#
#   Purpose: directly comparable in shape/axis layout to literature curves
#   such as Sørensen et al. (2022) Fig. 7/8 and Andres-Maldonado et al.
#   (2019) Fig. 3, to sanity-check whether this model's T3412/T3324
#   relationship produces a similar lifetime curve shape and range.
#
#   NS-3-measured points (Group E, 36 configs) are plotted as markers.
#   The analytical SMP curve is overlaid as a solid line through a denser
#   T3412 range, so the discrete NS-3 points can be checked against the
#   continuous relationship the model predicts.
# ═════════════════════════════════════════════════════════════════════════════
def figE():
    e = df[df["group"]=="E"].copy()
    if e.empty:
        print("[WARN] No Group E data — run the sweep first"); return

    # T3412 was stored in seconds in the CSV; convert to hours for the x-axis.
    # The 0h anchor point was simulated at 300s (5 min) — see run_benchmark.sh.
    e["T3412_hr"] = e["T3412"] / 3600.0
    # Snap the 300s anchor back to a displayed 0h tick
    e.loc[e["T3412"] == 300, "T3412_hr"] = 0.0

    t3324_list = sorted(e["T3324"].unique())
    cmap       = plt.get_cmap("viridis")
    colors     = {t: cmap(i / max(1, len(t3324_list)-1)) for i, t in enumerate(t3324_list)}
    markers    = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Analytical overlay: dense T3412 range per T3324 line ──────────────────
    T3412_dense_hr = np.linspace(0.05, 15, 300)   # avoid exact 0 (div-by-zero)
    T3412_dense_s  = T3412_dense_hr * 3600

    for i, t3324 in enumerate(t3324_list):
        lt_curve = []
        for t_s in T3412_dense_s:
            _, lt, _, _, _ = smp(T3412=t_s, T3324=t3324,
                              interval=54000, ce=0)
            lt_curve.append(lt)
        ax.plot(T3412_dense_hr, lt_curve,
                color=colors[t3324], linewidth=1.8, alpha=0.85, zorder=3)

    # ── NS-3 measured points ──────────────────────────────────────────────────
    for i, t3324 in enumerate(t3324_list):
        sub = e[e["T3324"] == t3324].sort_values("T3412_hr")
        ax.plot(sub["T3412_hr"], sub["lifetime_yr"],
                color=colors[t3324], marker=markers[i % len(markers)],
                markersize=8, linewidth=0, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"T3324 = {t3324}s")

    ax.set_xlabel("T3412 — PSM Sleep Duration (hours)")
    ax.set_ylabel("Battery Lifetime (years)")
    ax.set_title(
        "Fig E: Battery Lifetime vs T3412, by T3324 Configuration\n"
        "CE-A, interval=15h fixed (no TAU) — solid lines: analytical SMP, "
        "markers: NS-3 result\n"
        "Compare shape/range against literature lifetime-vs-T3412 curves"
    )

    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.set_yticks([5, 10, 15, 20, 25])
    ax.set_xlim(-0.3, 15.5)
    ax.set_ylim(0, max(28, e["lifetime_yr"].max() * 1.05))

    ax.axhline(10, color="grey", ls="--", lw=1, alpha=0.6,
              label="3GPP 10-yr target")
    ax.legend(title="T3324 (Active/eDRX window)", loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "figE_literature_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG X — SUMMARY SCATTER  (all configs)
# p_avg_mw vs lifetime_yr, coloured by group, sized by T3412
# ═════════════════════════════════════════════════════════════════════════════
def figX():
    valid = df.dropna(subset=["p_avg_mw","lifetime_yr"]).copy()
    if valid.empty:
        print("[WARN] No p_avg_mw data for summary scatter"); return

    fig, ax = plt.subplots(figsize=(11, 7))
    for grp, sub in valid.groupby("group"):
        col = GRP_COL.get(grp, "#555555")
        ax.scatter(sub["p_avg_mw"], sub["lifetime_yr"],
                   color=col, s=60, alpha=0.8,
                   edgecolors="white", linewidths=0.5,
                   label=f"Group {grp}", zorder=4)

    # Annotate outliers (top/bottom 5 by lifetime)
    for _, row in pd.concat([
        valid.nlargest(5, "lifetime_yr"),
        valid.nsmallest(5, "lifetime_yr")
    ]).drop_duplicates().iterrows():
        ax.annotate(row["label"],
                    (row["p_avg_mw"], row["lifetime_yr"]),
                    fontsize=5.5, alpha=0.75,
                    xytext=(4,2), textcoords="offset points")

    ax.set_xlabel("Average Power (mW, from corrected SMP v2)")
    ax.set_ylabel("Battery Lifetime (years)")
    ax.set_title("Fig X: Summary — All Configurations\n"
                 "p_avg_mw vs Lifetime, coloured by group")
    ax.axhline(10, color="grey", ls="--", lw=1, alpha=0.6, label="3GPP 10-yr target")
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    save(fig, "figX_summary_scatter.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG F — GROUP F FULL FACTORIAL GRID (504 rows: 7 T3412 x 6 T3324 x 4
# interval x 3 CE). This is a genuinely 4-dimensional dataset, which cannot
# be shown on one 2D chart without holding some variables fixed. These two
# figures each fix ONE representative traffic interval (7200s / 2 hours,
# chosen as a reasonably central, realistic value across the sweep's range)
# and show T3412 on the x-axis, one line per T3324 value, with three
# side-by-side panels — one per coverage mode — so all three CE levels are
# visible together for direct comparison.
#
# To see a DIFFERENT interval slice, change FIGF_INTERVAL below and re-run —
# the full 504-row dataset supports any of the 4 interval values used in the
# original sweep (300, 1800, 7200, 86400 seconds).
# ═════════════════════════════════════════════════════════════════════════════
FIGF_INTERVAL = 7200  # seconds (2 hours) — change this to view a different slice

def _figF_common_setup():
    sub = df_f[df_f["interval"] == FIGF_INTERVAL].copy()
    if sub.empty:
        available = sorted(df_f["interval"].unique())
        print(f"[WARN] No Group F rows at interval={FIGF_INTERVAL}. "
              f"Available interval values in this CSV: {available}")
        return None
    return sub

def figF_lifetime_grid():
    sub = _figF_common_setup()
    if sub is None:
        return
    ce_modes = sorted(sub["ceMode"].unique())
    t3324_list = sorted(sub["T3324"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {t: cmap(i/max(1,len(t3324_list)-1)) for i, t in enumerate(t3324_list)}

    fig, axes = plt.subplots(1, len(ce_modes), figsize=(6*len(ce_modes), 5.5), sharey=True)
    if len(ce_modes) == 1:
        axes = [axes]
    ce_names = {0: "CE-A (Good)", 1: "CE-B (Medium)", 2: "CE-C (Poor)"}

    for ax, ce in zip(axes, ce_modes):
        ce_sub = sub[sub["ceMode"] == ce]
        for t3324 in t3324_list:
            line = ce_sub[ce_sub["T3324"] == t3324].sort_values("T3412")
            ax.plot(line["T3412"]/3600.0, line["lifetime_yr"],
                    marker="o", markersize=4, color=colors[t3324],
                    label=f"T3324={t3324}s")
        ax.set_xscale("log")
        ax.set_xlabel("T3412 (hours)")
        ax.set_title(ce_names.get(ce, f"CE={ce}"))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Battery Lifetime (years)")
    axes[-1].legend(title="T3324", loc="upper left", fontsize=8)
    fig.suptitle(f"Group F: Lifetime vs T3412, by T3324 and Coverage Mode "
                f"(traffic interval = {FIGF_INTERVAL}s)")
    fig.tight_layout()
    save(fig, "figF_lifetime_grid.png")

def figF_delay_grid():
    sub = _figF_common_setup()
    if sub is None:
        return
    if "latency_s" not in sub.columns:
        print("[WARN] No latency_s column in results_group_f.csv — skipping figF_delay_grid "
              "(this column was added by backfill_latency_column.py or CC file v4+)")
        return
    ce_modes = sorted(sub["ceMode"].unique())
    t3324_list = sorted(sub["T3324"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {t: cmap(i/max(1,len(t3324_list)-1)) for i, t in enumerate(t3324_list)}

    fig, axes = plt.subplots(1, len(ce_modes), figsize=(6*len(ce_modes), 5.5), sharey=True)
    if len(ce_modes) == 1:
        axes = [axes]
    ce_names = {0: "CE-A (Good)", 1: "CE-B (Medium)", 2: "CE-C (Poor)"}

    for ax, ce in zip(axes, ce_modes):
        ce_sub = sub[sub["ceMode"] == ce]
        for t3324 in t3324_list:
            line = ce_sub[ce_sub["T3324"] == t3324].sort_values("T3412")
            ax.plot(line["T3412"]/3600.0, line["latency_s"]/3600.0,
                    marker="o", markersize=4, color=colors[t3324],
                    label=f"T3324={t3324}s")
        ax.set_xscale("log")
        ax.set_xlabel("T3412 (hours)")
        ax.set_title(ce_names.get(ce, f"CE={ce}"))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Expected Buffering Delay (hours)")
    axes[-1].legend(title="T3324", loc="upper left", fontsize=8)
    fig.suptitle(f"Group F: Expected Delay vs T3412, by T3324 and Coverage Mode "
                f"(traffic interval = {FIGF_INTERVAL}s)")
    fig.tight_layout()
    save(fig, "figF_delay_grid.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG F2 — T3324's EFFECT, ISOLATED AT A FIXED, SMALL T3412
#
# WHY THIS FIGURE EXISTS: figF_lifetime_grid/figF_delay_grid plot the FULL
# T3412 range (300s to 86400s) on their y-axis, because that's what the
# grid is FOR. But T3324's own effect on lifetime/delay is small relative
# to that huge T3412-driven range — at the smallest T3412 (where T3324
# matters MOST, proportionally), the difference between T3324=5s and
# T3324=30s is under 4% in relative terms, and under 0.1 years in
# absolute terms — completely invisible next to a 0-90 year axis.
#
# This is NOT a data bug — it's the same validated finding as figA2
# elsewhere in this project (T3412 dominates; T3324 has a real but
# comparatively small effect, consistent with Anbazhagan & Mugelan's own
# reported PSF sensitivity to their eDRX timer). This figure isolates
# T3324 the same way figA2 does: fix T3412 at ONE value, zoom the y-axis
# in tightly enough to actually show the real variation.
# ═════════════════════════════════════════════════════════════════════════════
FIGF2_T3412 = 300  # seconds — smallest value in the grid, where T3324's
                    # proportional effect is largest, per direct calculation

def figF_T3324_effect():
    if df_f is None:
        return
    sub = df_f[(df_f["T3412"] == FIGF2_T3412) & (df_f["interval"] == FIGF_INTERVAL)].copy()
    if sub.empty:
        available_t3412 = sorted(df_f["T3412"].unique())
        print(f"[WARN] No Group F rows at T3412={FIGF2_T3412}, interval={FIGF_INTERVAL}. "
              f"Available T3412 values: {available_t3412}")
        return

    has_delay = "latency_s" in sub.columns
    ce_modes = sorted(sub["ceMode"].unique())
    ce_names = {0: "CE-A (Good)", 1: "CE-B (Medium)", 2: "CE-C (Poor)"}
    ce_colors = {0: "#1565C0", 1: "#EF6C00", 2: "#B00020"}

    ncols = 2 if has_delay else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ce in ce_modes:
        line = sub[sub["ceMode"] == ce].sort_values("T3324")
        axes[0].plot(line["T3324"], line["lifetime_yr"], marker="o",
                    color=ce_colors.get(ce, "#333"), label=ce_names.get(ce, f"CE={ce}"))
    axes[0].set_xlabel("T3324 (seconds)")
    axes[0].set_ylabel("Battery Lifetime (years)")
    axes[0].set_title(f"Lifetime vs T3324\n(T3412 fixed at {FIGF2_T3412}s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    if has_delay:
        for ce in ce_modes:
            line = sub[sub["ceMode"] == ce].sort_values("T3324")
            axes[1].plot(line["T3324"], line["latency_s"], marker="o",
                        color=ce_colors.get(ce, "#333"), label=ce_names.get(ce, f"CE={ce}"))
        axes[1].set_xlabel("T3324 (seconds)")
        axes[1].set_ylabel("Expected Buffering Delay (seconds)")
        axes[1].set_title(f"Delay vs T3324\n(T3412 fixed at {FIGF2_T3412}s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

    fig.suptitle(f"Group F: T3324's Own Effect, Isolated (T3412={FIGF2_T3412}s, "
                f"interval={FIGF_INTERVAL}s)\n"
                f"Y-axis zoomed to the real range — this is the same relationship "
                f"already shown at full T3412 scale, just visible now")
    fig.tight_layout()
    save(fig, "figF_T3324_effect.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[PLOT] Generating figures...")
    if HAVE_MAIN:
        figA1()
        figA2()
        figB()
        figC()
        figD()
        figE()
        figX()
    else:
        print("[SKIP] results.csv not present — skipped figA1 through figX")

    if HAVE_GROUP_F:
        figF_lifetime_grid()
        figF_delay_grid()
        figF_T3324_effect()
    else:
        print("[SKIP] results_group_f.csv not present — skipped figF_*")

    print(f"[PLOT] Done. Figures in {FIG_DIR}")
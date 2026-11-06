#!/usr/bin/env python3
"""
Point 12 (reviewer feedback): Monte Carlo uncertainty analysis on the
power constants.

The power values used throughout this project (P_PSM_MW, P_SYNC_MW,
etc.) are real hardware measurements from Sørensen et al., but have
been used as fixed, exact numbers everywhere. Real measurements carry
natural uncertainty. This tests: if each constant varied within a
realistic range, how much would the actual results (lifetime, PSF)
shift as a consequence?

UNCERTAINTY ASSUMPTION: +/-10% around each measured value, applied
independently per constant per trial. This is a standard, conservative
engineering assumption for hardware power measurements -- NOT a figure
sourced from Sørensen et al. directly, since their paper does not
report explicit measurement error bars in the material available to
this project. Flagged clearly as an assumption, not a citation.

Does not modify smp_engine.py -- overrides its module-level constants
temporarily, matching the same safe pattern used throughout this
project's other sensitivity investigations.

Run from your qlearning/ folder: python3 monte_carlo_uncertainty.py
"""
import sys
sys.path.insert(0, ".")
import csv
import os
import numpy as np
import smp_engine

UNCERTAINTY_PCT = 0.10  # +/-10%, a labeled assumption, not a sourced figure
N_TRIALS = 2000

# The original, measured values -- kept here so each trial perturbs
# around the TRUE original, not a previous trial's already-perturbed one
ORIGINAL = {
    "P_PSM_MW": smp_engine.P_PSM_MW,
    "P_SYNC_MW": smp_engine.P_SYNC_MW,
    "P_RRC_MW": smp_engine.P_RRC_MW,
    "P_TX_MW": smp_engine.P_TX_MW,
    "P_TX_GAP_MW": smp_engine.P_TX_GAP_MW,
    "P_RX_MW": smp_engine.P_RX_MW,
    "P_ACTIVE_MW": smp_engine.P_ACTIVE_MW,
}

# Representative test configs -- one per coverage mode, a realistic
# mid-range timer setting, to keep results focused and interpretable
# rather than reporting uncertainty for all 30 states individually
TEST_CONFIGS = [
    ("CE-A, T3412=3600s, T3324=20s, interval=3600s", 3600, 20, 3600, 0),
    ("CE-B, T3412=3600s, T3324=20s, interval=3600s", 3600, 20, 3600, 1),
    ("CE-C, T3412=3600s, T3324=20s, interval=3600s", 3600, 20, 3600, 2),
]

rng = np.random.default_rng(42)

print(f"Monte Carlo uncertainty analysis: {N_TRIALS} trials, +/-{UNCERTAINTY_PCT*100:.0f}% "
      f"per constant (assumption, not a sourced figure)\n")

os.makedirs("results", exist_ok=True)
raw_path = "results/monte_carlo_uncertainty_raw.csv"
summary_path = "results/monte_carlo_uncertainty_summary.csv"

raw_rows = []
summary_rows = []

for label, T3412, T3324, interval, ce in TEST_CONFIGS:
    lifetimes = []
    psfs = []
    for trial in range(N_TRIALS):
        # Perturb every constant independently, each trial fresh from the original
        perturbed_values = {}
        for name, original_value in ORIGINAL.items():
            perturbed = original_value * (1 + rng.uniform(-UNCERTAINTY_PCT, UNCERTAINTY_PCT))
            setattr(smp_engine, name, perturbed)
            perturbed_values[name] = perturbed

        result = smp_engine.compute_smp(T3412, T3324, interval, ce)
        lifetimes.append(result.lifetime_yr)
        psfs.append(result.psf)

        # Save every individual trial -- real, checkable evidence, not
        # just the final summary statistics
        raw_rows.append({
            "config_label": label, "trial": trial,
            "lifetime_yr": result.lifetime_yr, "psf": result.psf,
            **{f"perturbed_{k}": v for k, v in perturbed_values.items()},
        })

    # Restore originals before moving to the next config
    for name, original_value in ORIGINAL.items():
        setattr(smp_engine, name, original_value)

    # Baseline (unperturbed) result, for direct comparison
    baseline = smp_engine.compute_smp(T3412, T3324, interval, ce)

    lifetimes = np.array(lifetimes)
    psfs = np.array(psfs)
    print(f"=== {label} ===")
    print(f"  Baseline (original constants): lifetime={baseline.lifetime_yr:.2f}yr  PSF={baseline.psf*100:.2f}%")
    print(f"  Monte Carlo lifetime: mean={lifetimes.mean():.2f}yr  std={lifetimes.std():.2f}yr  "
          f"90% range=[{np.percentile(lifetimes,5):.2f}, {np.percentile(lifetimes,95):.2f}]yr")
    print(f"  Monte Carlo PSF:      mean={psfs.mean()*100:.2f}%  std={psfs.std()*100:.2f}%  "
          f"90% range=[{np.percentile(psfs,5)*100:.2f}%, {np.percentile(psfs,95)*100:.2f}%]")
    print()

    summary_rows.append({
        "config_label": label, "baseline_lifetime_yr": baseline.lifetime_yr,
        "baseline_psf_pct": baseline.psf*100,
        "mc_lifetime_mean": lifetimes.mean(), "mc_lifetime_std": lifetimes.std(),
        "mc_lifetime_p5": np.percentile(lifetimes,5), "mc_lifetime_p95": np.percentile(lifetimes,95),
        "mc_psf_mean_pct": psfs.mean()*100, "mc_psf_std_pct": psfs.std()*100,
        "mc_psf_p5_pct": np.percentile(psfs,5)*100, "mc_psf_p95_pct": np.percentile(psfs,95)*100,
    })

with open(raw_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
    writer.writeheader()
    writer.writerows(raw_rows)

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"[SAVED] Every individual trial (6,000 rows): {raw_path}")
print(f"[SAVED] Summary statistics per config: {summary_path}")
#!/usr/bin/env python3
"""
Point 13 (reviewer feedback): transfer learning across hardware.

Tests whether the policy already trained on the N211 chip's power
profile still performs well when the UNDERLYING PHYSICS uses a
genuinely different, real device's measured constants -- the BG96
chip, values found directly inside LENA-NB's nb-iot-energy.h during
the NS-3 investigation.

This is NOT a retrain -- it deliberately keeps your existing, already-
trained q_table.npy exactly as-is, and asks: do ITS decisions still
make sense under different real hardware? That is what "transfer"
actually means here.

STATE MAPPING (documented honestly, not all matches are equally solid):
  PSM    -> BG96 PSM     (direct match)
  TX     -> BG96 Uplink  (direct match)
  RX     -> BG96 Downlink (direct match)
  ACTIVE -> BG96 eDRX    (direct match, same underlying concept)
  SYNC   -> BG96 Idle    (APPROXIMATION -- BG96 has no separate sync state)
  RRC    -> BG96 Idle    (APPROXIMATION -- same reasoning as SYNC)

Run from your qlearning/ folder: python3 transfer_learning_bg96.py
"""
import sys
sys.path.insert(0, ".")
import csv
import os
import numpy as np
import smp_engine
from agent import QLearningAgent
from environment import (
    ACTIONS, CE_MODES, TRAFFIC_BUCKETS, bucket_to_representative_interval,
    normalize_lifetime, normalize_delay, DELAY_WEIGHT_BY_BUCKET,
)

# ── Real BG96 constants, converted to mW, from LENA-NB's nb-iot-energy.h ──
BG96_PSM_MW = 3.8 * 3.9e-6 * 1000
BG96_DRX_MW = 3.8 * 1.56e-3 * 1000
BG96_EDRX_MW = 3.8 * 0.81e-3 * 1000
BG96_UPLINK_MW = 3.8 * 155e-3 * 1000
BG96_DOWNLINK_MW = 80e-3 * 1000
BG96_IDLE_MW = 3.8 * 0.81e-3 * 1000

# The N211 (original) constants, kept for switching back
ORIGINAL = {
    "P_PSM_MW": smp_engine.P_PSM_MW,
    "P_SYNC_MW": smp_engine.P_SYNC_MW,
    "P_RRC_MW": smp_engine.P_RRC_MW,
    "P_TX_MW": smp_engine.P_TX_MW,
    "P_RX_MW": smp_engine.P_RX_MW,
    "P_ACTIVE_MW": smp_engine.P_ACTIVE_MW,
}

BG96_MAPPED = {
    "P_PSM_MW": BG96_PSM_MW,
    "P_SYNC_MW": BG96_IDLE_MW,      # approximation, documented above
    "P_RRC_MW": BG96_IDLE_MW,       # approximation, documented above
    "P_TX_MW": BG96_UPLINK_MW,
    "P_RX_MW": BG96_DOWNLINK_MW,
    "P_ACTIVE_MW": BG96_EDRX_MW,
}

T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
N_STATES = len(CE_MODES) * len(TRAFFIC_BUCKETS)
CORRECTNESS_GAP_THRESHOLD = 0.02


def true_optimal_under_current_constants():
    """Computed fresh each time it's called -- reflects whatever
    smp_engine's constants are currently set to (N211 or BG96)."""
    optimal = {}
    for ce in CE_MODES:
        for bucket in range(len(TRAFFIC_BUCKETS)):
            interval = bucket_to_representative_interval(bucket)
            w_dl = 1.0 * DELAY_WEIGHT_BY_BUCKET[bucket]
            rewards = {}
            for T3412 in T3412_vals:
                r = smp_engine.compute_smp(T3412, 20, interval, ce)
                nlt = normalize_lifetime(r.lifetime_yr)
                ndl = normalize_delay(r.delay_seconds)
                rewards[T3412] = 0.7 * nlt - w_dl * ndl
            optimal[(ce, bucket)] = (max(rewards, key=rewards.get), rewards)
    return optimal


def audit(agent, optimal):
    results = []
    n_correct = 0
    for ce in CE_MODES:
        for bucket in range(len(TRAFFIC_BUCKETS)):
            state_idx = ce * len(TRAFFIC_BUCKETS) + bucket
            action_idx = int(np.argmax(agent.q_table[state_idx]))
            learned_T3412 = ACTIONS[action_idx][0]
            true_best_T3412, rewards = optimal[(ce, bucket)]
            true_best_reward = rewards[true_best_T3412]
            learned_reward = rewards.get(learned_T3412, true_best_reward)
            gap = true_best_reward - learned_reward
            ok = gap <= CORRECTNESS_GAP_THRESHOLD
            if ok:
                n_correct += 1
            results.append((ce, bucket, learned_T3412, true_best_T3412, gap, ok))
    return n_correct, results


# ── Load the EXISTING, already-trained agent -- not retraining ──
agent = QLearningAgent(n_states=N_STATES, n_actions=42)
agent.load("results/q_table.npy")

print("Auditing the N211-trained policy under N211's OWN constants (baseline)...")
n211_optimal = true_optimal_under_current_constants()
n211_correct, n211_results = audit(agent, n211_optimal)
print(f"  {n211_correct}/{N_STATES} correct under N211 (should match what's already established)\n")

print("Switching smp_engine to BG96's real, measured constants...")
for name, value in BG96_MAPPED.items():
    setattr(smp_engine, name, value)

print("Auditing the SAME, UNCHANGED N211-trained policy under BG96's constants...")
bg96_optimal = true_optimal_under_current_constants()
bg96_correct, bg96_results = audit(agent, bg96_optimal)
print(f"  {bg96_correct}/{N_STATES} correct under BG96 (this is the actual transfer test)\n")

# Restore N211 constants
for name, value in ORIGINAL.items():
    setattr(smp_engine, name, value)

print("="*70)
print("TRANSFER LEARNING RESULT")
print("="*70)
print(f"  N211 (device the policy was trained on): {n211_correct}/{N_STATES} correct")
print(f"  BG96 (different, real device, same policy, NOT retrained): {bg96_correct}/{N_STATES} correct")
retention_pct = bg96_correct / n211_correct * 100 if n211_correct > 0 else 0
print(f"  Retention: {retention_pct:.1f}% of N211 performance retained on BG96")

# Save full results
os.makedirs("results", exist_ok=True)
with open("results/transfer_learning_bg96.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ce_mode", "bucket", "learned_T3412", "n211_true_best", "n211_gap", "n211_ok",
                      "bg96_true_best", "bg96_gap", "bg96_ok"])
    for (ce, b, learned, n211_best, n211_gap, n211_ok), (_, _, _, bg96_best, bg96_gap, bg96_ok) in zip(n211_results, bg96_results):
        writer.writerow([ce, b, learned, n211_best, round(n211_gap,4), n211_ok,
                          bg96_best, round(bg96_gap,4), bg96_ok])
print(f"\n[SAVED] Full per-state comparison: results/transfer_learning_bg96.csv")
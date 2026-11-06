#!/usr/bin/env python3
"""
Point 4 (reviewer feedback), log-spaced version: replaces uniform-width
traffic buckets with log-spaced ones, giving fine resolution where
realistic NB-IoT reporting actually happens (minutes range) without
needing thousands of buckets to also cover the long end (up to 24h).

First bucket deliberately starts at 0, not an arbitrary floor -- fixes
a real edge-case bug found during design (anything below the first log
boundary would otherwise fall through to the wrong bucket).

Run from your qlearning/ folder: python3 bucket_sensitivity_logspaced.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import environment
from train import train
from environment import ACTIONS, CE_MODES

# ── Log-spaced 10-bucket scheme ──────────────────────────────────────────
N_BUCKETS = 10
LO, HI = 60, 86400
_raw = [LO * (HI/LO)**(i/N_BUCKETS) for i in range(N_BUCKETS+1)]
_boundaries = [0] + _raw[1:]  # first bucket covers 0 upward, not an arbitrary floor

NEW_TRAFFIC_BUCKETS = [
    (_boundaries[i], _boundaries[i+1], f"bucket_{i}_{int(_boundaries[i])}-{int(_boundaries[i+1])}s")
    for i in range(N_BUCKETS)
]
NEW_DELAY_WEIGHTS = {i: 0.45 for i in range(N_BUCKETS)}  # flat, informed by Point 6

environment.TRAFFIC_BUCKETS = NEW_TRAFFIC_BUCKETS
environment.DELAY_WEIGHT_BY_BUCKET = NEW_DELAY_WEIGHTS

print(f"New state space: {len(CE_MODES)} CE modes x {N_BUCKETS} log-spaced buckets "
      f"= {len(CE_MODES)*N_BUCKETS} states")
print("\nBucket boundaries:")
for lo, hi, name in NEW_TRAFFIC_BUCKETS:
    print(f"  {name}")

print("\nTraining fresh agent under the log-spaced scheme...")
agent = train(n_episodes=60000, seed=42, verbose=False)
print("Done.\n")


def query(ce, interval_s):
    bucket = environment.traffic_to_bucket(interval_s)
    state_idx = ce * N_BUCKETS + bucket
    action_idx = int(np.argmax(agent.q_table[state_idx]))
    T3412, T3324 = ACTIONS[action_idx]
    bucket_name = NEW_TRAFFIC_BUCKETS[bucket][2]
    return bucket_name, T3412, T3324


print("="*70)
print("TEST 1: realistic scale (100s vs 200s) -- should now differ")
print("="*70)
for interval in [100, 200]:
    bucket_name, T3412, T3324 = query(0, interval)
    print(f"  CE-A, interval={interval:>4}s -> {bucket_name:20s} T3412={T3412:>6}s T3324={T3324}s")

print("\n" + "="*70)
print("TEST 2: unrealistic scale (1s vs 10s) -- should NOT differ,")
print("        both fall in the same 'as frequent as NB-IoT gets' bucket")
print("="*70)
for interval in [1, 10]:
    bucket_name, T3412, T3324 = query(0, interval)
    print(f"  CE-A, interval={interval:>4}s -> {bucket_name:20s} T3412={T3412:>6}s T3324={T3324}s")

print("\n" + "="*70)
print("TEST 3: original 300s example, revisited")
print("="*70)
for interval in [100, 400, 700]:
    bucket_name, T3412, T3324 = query(0, interval)
    print(f"  CE-A, interval={interval:>4}s -> {bucket_name:20s} T3412={T3412:>6}s T3324={T3324}s")
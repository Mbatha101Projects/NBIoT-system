#!/usr/bin/env python3
"""
Seed search, re-run for the updated 30-state production system (10
log-spaced traffic buckets, flat 0.45 delay weight -- see environment.py
and changelog for the investigations behind both changes).

Saves full results to results/seed_search_30state.csv, not just the
terminal -- addressing the earlier concern that test output wasn't
being kept anywhere checkable.

Run from your qlearning/ folder: python3 seed_search.py
"""
import sys
sys.path.insert(0, ".")
import os
import csv
import numpy as np
from train import train
from smp_engine import compute_smp
from environment import (
    ACTIONS, CE_MODES, TRAFFIC_BUCKETS, bucket_to_representative_interval,
    normalize_lifetime, normalize_delay, DELAY_WEIGHT_BY_BUCKET,
)

SEEDS = list(range(1, 21))  # 20 seeds: 1 through 20
T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
CORRECTNESS_GAP_THRESHOLD = 0.02
N_STATES = len(CE_MODES) * len(TRAFFIC_BUCKETS)


def true_optimal_for_all_states():
    """Computed ONCE -- doesn't depend on the seed, only on the (unmodified,
    production) environment settings."""
    optimal = {}
    for ce in CE_MODES:
        for bucket in range(len(TRAFFIC_BUCKETS)):
            interval = bucket_to_representative_interval(bucket)
            w_dl = 1.0 * DELAY_WEIGHT_BY_BUCKET[bucket]
            rewards = {}
            for T3412 in T3412_vals:
                r = compute_smp(T3412, 20, interval, ce)
                nlt = normalize_lifetime(r.lifetime_yr)
                ndl = normalize_delay(r.delay_seconds)
                rewards[T3412] = 0.7 * nlt - w_dl * ndl
            optimal[(ce, bucket)] = (max(rewards, key=rewards.get), rewards)
    return optimal


def audit(agent, optimal):
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
            if gap <= CORRECTNESS_GAP_THRESHOLD:
                n_correct += 1
    return n_correct


optimal = true_optimal_for_all_states()

os.makedirs("results", exist_ok=True)
csv_path = "results/seed_search_30state.csv"

print(f"Testing {len(SEEDS)} seeds on the UPDATED 30-state production system...\n")
results = []
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "n_correct", "n_states"])
    for seed in SEEDS:
        agent = train(n_episodes=60000, seed=seed, verbose=False)
        score = audit(agent, optimal)
        results.append((seed, score))
        writer.writerow([seed, score, N_STATES])
        print(f"  seed={seed:>3}: {score}/{N_STATES}")

results_sorted = sorted(results, key=lambda x: -x[1])
scores_only = [s for _, s in results]

print("\n" + "="*70)
print("RANKED (best to worst) -- for picking a deployment candidate")
print("="*70)
for rank, (seed, score) in enumerate(results_sorted, 1):
    print(f"  #{rank:>2}: seed={seed:>3}  {score}/{N_STATES}")

print("\n" + "="*70)
print("AGGREGATE STATISTICS -- for honest reporting")
print("="*70)
print(f"  Mean:  {np.mean(scores_only):.2f}/{N_STATES}")
print(f"  Range: {min(scores_only)}-{max(scores_only)}/{N_STATES}")
print(f"  Std:   {np.std(scores_only):.2f}")
print(f"\n  Best seed found: {results_sorted[0][0]} ({results_sorted[0][1]}/{N_STATES})")
print(f"\n[SAVED] Full results written to {csv_path}")
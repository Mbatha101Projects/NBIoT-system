#!/usr/bin/env python3

import sys
sys.path.insert(0, ".")
import numpy as np
import environment
from train import train
from smp_engine import compute_smp
from environment import (
    ACTIONS, CE_MODES, TRAFFIC_BUCKETS, bucket_to_representative_interval,
    normalize_lifetime, normalize_delay,
)

CANDIDATES = {
    "original": {0: 0.60, 1: 0.45, 2: 0.30, 3: 0.15},
    "flatter":  {0: 0.40, 1: 0.35, 2: 0.30, 3: 0.25},
    "steeper":  {0: 0.80, 1: 0.55, 2: 0.30, 3: 0.05},
    "uniform":  {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35},
    "inverted": {0: 0.15, 1: 0.30, 2: 0.45, 3: 0.60},
    "wider":        {0: 0.70, 1: 0.50, 2: 0.30, 3: 0.10},
    "narrower":     {0: 0.50, 1: 0.40, 2: 0.30, 3: 0.20},
    "uniform_low":  {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
    "uniform_high": {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45},
}
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]

T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
bucket_names = [b[2] for b in TRAFFIC_BUCKETS]
ce_names = {0: "CE-A", 1: "CE-B", 2: "CE-C"}
CORRECTNESS_GAP_THRESHOLD = 0.02


def true_optimal_for_weights(weights):
    optimal = {}
    for ce in CE_MODES:
        for bucket in range(len(TRAFFIC_BUCKETS)):
            interval = bucket_to_representative_interval(bucket)
            w_dl = 1.0 * weights[bucket]
            rewards = {}
            for T3412 in T3412_vals:
                r = compute_smp(T3412, 20, interval, ce)
                nlt = normalize_lifetime(r.lifetime_yr)
                ndl = normalize_delay(r.delay_seconds)
                rewards[T3412] = 0.7 * nlt - w_dl * ndl
            optimal[(ce, bucket)] = (max(rewards, key=rewards.get), rewards)
    return optimal


def audit_against(agent, optimal):
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


all_results = {}
for name, weights in CANDIDATES.items():
    print(f"\n=== '{name}' weights: {weights} ===")
    optimal = true_optimal_for_weights(weights)  # same for every seed, computed once
    environment.DELAY_WEIGHT_BY_BUCKET = weights

    per_seed_scores = []
    for seed in SEEDS:
        agent = train(n_episodes=60000, seed=seed, verbose=False)
        n_correct = audit_against(agent, optimal)
        per_seed_scores.append(n_correct)
        print(f"  seed={seed}: {n_correct}/12")

    all_results[name] = per_seed_scores
    mean = sum(per_seed_scores) / len(per_seed_scores)
    print(f"  -> mean={mean:.1f}/12  range={min(per_seed_scores)}-{max(per_seed_scores)}  "
          f"spread={max(per_seed_scores)-min(per_seed_scores)}")

print("\n" + "="*70)
print("SUMMARY -- mean and consistency across 5 seeds per weight scheme")
print("="*70)
print(f"{'Scheme':<10}{'Per-seed scores':<25}{'Mean':<8}{'Range':<10}{'Spread'}")
for name, scores in all_results.items():
    mean = sum(scores)/len(scores)
    print(f"{name:<10}{str(scores):<25}{mean:<8.1f}{f'{min(scores)}-{max(scores)}':<10}{max(scores)-min(scores)}")
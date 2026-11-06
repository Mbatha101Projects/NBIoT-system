#!/usr/bin/env python3
"""
Point 4 follow-up: tests whether the odd result from the 30-state
log-spaced scheme (bucket 0 and bucket 2 landing on identical actions,
despite bucket 1 differing) was caused by insufficient training for the
larger state space, rather than a real property of the traffic pattern.

Scales training episodes from 60,000 to 150,000 -- proportional to the
2.5x increase in state count (30 states vs the original 12) -- and
tests across 5 seeds instead of 1, matching the rigor already applied
to the reward-weight and initialisation investigations.

Run from your qlearning/ folder: python3 bucket_scaling_test.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import environment
from train import train
from smp_engine import compute_smp
from environment import (
    ACTIONS, CE_MODES, normalize_lifetime, normalize_delay,
)

# ── Same log-spaced 10-bucket scheme as before ───────────────────────────
N_BUCKETS = 10
LO, HI = 60, 86400
_raw = [LO * (HI/LO)**(i/N_BUCKETS) for i in range(N_BUCKETS+1)]
_boundaries = [0] + _raw[1:]
NEW_TRAFFIC_BUCKETS = [
    (_boundaries[i], _boundaries[i+1], f"bucket_{i}")
    for i in range(N_BUCKETS)
]
NEW_DELAY_WEIGHTS = {i: 0.45 for i in range(N_BUCKETS)}
environment.TRAFFIC_BUCKETS = NEW_TRAFFIC_BUCKETS
environment.DELAY_WEIGHT_BY_BUCKET = NEW_DELAY_WEIGHTS

T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
CORRECTNESS_GAP_THRESHOLD = 0.02

# Episode scaling: 60,000 was proven sufficient for 12 states.
# 30 states = 2.5x more -> scale proportionally.
ORIGINAL_STATES = 12
NEW_STATES = len(CE_MODES) * N_BUCKETS
SCALE_FACTOR = NEW_STATES / ORIGINAL_STATES
N_EPISODES_SCALED = int(60000 * SCALE_FACTOR)

print(f"State count: {NEW_STATES} (was {ORIGINAL_STATES}) -> scale factor {SCALE_FACTOR:.2f}x")
print(f"Episodes: {N_EPISODES_SCALED} (was 60,000)\n")


def true_optimal_for_all_states():
    """Brute-force true-optimal T3412 for every one of the 30 states,
    using the actual log-spaced buckets' representative intervals."""
    optimal = {}
    for ce in CE_MODES:
        for bucket in range(N_BUCKETS):
            lo, hi, _name = NEW_TRAFFIC_BUCKETS[bucket]
            interval = hi/2 if lo == 0 else (lo*hi)**0.5  # same geometric-mid logic as environment.py
            w_dl = 1.0 * NEW_DELAY_WEIGHTS[bucket]
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
        for bucket in range(N_BUCKETS):
            state_idx = ce * N_BUCKETS + bucket
            action_idx = int(np.argmax(agent.q_table[state_idx]))
            learned_T3412 = ACTIONS[action_idx][0]
            true_best_T3412, rewards = optimal[(ce, bucket)]
            true_best_reward = rewards[true_best_T3412]
            learned_reward = rewards.get(learned_T3412, true_best_reward)
            gap = true_best_reward - learned_reward
            if gap <= CORRECTNESS_GAP_THRESHOLD:
                n_correct += 1
    return n_correct


optimal = true_optimal_for_all_states()  # computed once, same for every seed

print("="*70)
print(f"BASELINE: original 60,000 episodes (as actually run earlier today)")
print("="*70)
baseline_scores = []
for seed in [42, 1, 2, 3, 4]:
    agent = train(n_episodes=60000, seed=seed, verbose=False)
    score = audit(agent, optimal)
    baseline_scores.append(score)
    print(f"  seed={seed}: {score}/{NEW_STATES}")
print(f"  -> mean={sum(baseline_scores)/5:.1f}/{NEW_STATES}  "
      f"range={min(baseline_scores)}-{max(baseline_scores)}")

print(f"\n{'='*70}")
print(f"SCALED: {N_EPISODES_SCALED} episodes (proportional to 30 states)")
print("="*70)
scaled_scores = []
for seed in [42, 1, 2, 3, 4]:
    agent = train(n_episodes=N_EPISODES_SCALED, seed=seed, verbose=False)
    score = audit(agent, optimal)
    scaled_scores.append(score)
    print(f"  seed={seed}: {score}/{NEW_STATES}")
print(f"  -> mean={sum(scaled_scores)/5:.1f}/{NEW_STATES}  "
      f"range={min(scaled_scores)}-{max(scaled_scores)}")

print(f"\n{'='*70}")
print("VERDICT")
print("="*70)
print(f"Baseline (60,000 ep): mean={sum(baseline_scores)/5:.1f}, "
      f"spread={max(baseline_scores)-min(baseline_scores)}")
print(f"Scaled ({N_EPISODES_SCALED} ep):  mean={sum(scaled_scores)/5:.1f}, "
      f"spread={max(scaled_scores)-min(scaled_scores)}")

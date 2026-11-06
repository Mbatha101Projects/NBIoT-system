#!/usr/bin/env python3
"""
Stochastic scenario generators for Point 1 of the reviewer feedback.

Generates randomized traffic-interval sequences (Poisson, Weibull,
bursty ON-OFF) and Markov-chain coverage transitions, to replace/extend
the 6 handcrafted scenarios with hundreds-to-thousands of randomly
generated ones.

Each generator produces a list of (ce_mode, traffic_interval_seconds)
pairs -- one per step -- matching the format the existing evaluation
harness (evaluate.py / scenarios.py) already consumes.
"""
import numpy as np


# ═════════════════════════════════════════════════════════════════════════
# TRAFFIC INTERVAL GENERATORS
# ═════════════════════════════════════════════════════════════════════════

def poisson_traffic_intervals(n_steps, mean_interval_s, rng):
    """
    Poisson-process traffic: inter-arrival times are exponentially
    distributed (this IS what defines a Poisson process). mean_interval_s
    is the average time between reports.
    """
    return rng.exponential(scale=mean_interval_s, size=n_steps)


def weibull_traffic_intervals(n_steps, mean_interval_s, shape, rng):
    """
    Weibull-distributed intervals. shape<1 = more bursty/clustered than
    Poisson; shape>1 = more regular/predictable than Poisson; shape=1
    reduces to the same exponential distribution as Poisson.
    """
    # numpy's weibull() is the standard (scale=1) form; scale it to hit
    # the target mean via the Weibull mean formula: scale * Gamma(1+1/shape)
    from math import gamma
    scale = mean_interval_s / gamma(1 + 1/shape)
    return scale * rng.weibull(shape, size=n_steps)


def bursty_on_off_intervals(n_steps, on_interval_s, off_interval_s,
                             on_duration_steps, off_duration_steps, rng):
    """
    Alternates between ON periods (frequent short intervals) and OFF
    periods (long gaps) -- e.g. a sensor that reports rapidly during an
    event, then goes quiet for an extended period.
    """
    intervals = []
    state = "ON"
    steps_in_state = 0
    target_duration = on_duration_steps
    for _ in range(n_steps):
        if state == "ON":
            intervals.append(rng.exponential(scale=on_interval_s))
        else:
            intervals.append(rng.exponential(scale=off_interval_s))
        steps_in_state += 1
        if steps_in_state >= target_duration:
            state = "OFF" if state == "ON" else "ON"
            target_duration = off_duration_steps if state == "OFF" else on_duration_steps
            steps_in_state = 0
    return np.array(intervals)


# ═════════════════════════════════════════════════════════════════════════
# COVERAGE TRANSITION GENERATOR (structured Markov chain, not uniform-random)
# ═════════════════════════════════════════════════════════════════════════

def markov_coverage_sequence(n_steps, transition_matrix, initial_ce, rng):
    """
    Generates a coverage-mode sequence using a real transition matrix,
    allowing "sticky" states (e.g. staying in CE-A most of the time,
    rather than uniformly jumping to a random new mode every change) --
    more physically realistic than pure uniform reassignment.
    transition_matrix[i][j] = probability of moving from mode i to mode j.
    """
    sequence = [initial_ce]
    current = initial_ce
    for _ in range(n_steps - 1):
        current = rng.choice(3, p=transition_matrix[current])
        sequence.append(current)
    return sequence


# ═════════════════════════════════════════════════════════════════════════
# VERIFICATION -- check each generator actually behaves as intended
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 100000

    print("=== Poisson traffic check ===")
    intervals = poisson_traffic_intervals(N, mean_interval_s=3600, rng=rng)
    print(f"Target mean: 3600s, empirical mean: {intervals.mean():.1f}s")
    print(f"Std/mean ratio: {intervals.std()/intervals.mean():.3f} (should be ~1.0 for a true Poisson process)")

    print("\n=== Weibull traffic check (bursty, shape=0.5) ===")
    intervals_bursty = weibull_traffic_intervals(N, mean_interval_s=3600, shape=0.5, rng=rng)
    print(f"Target mean: 3600s, empirical mean: {intervals_bursty.mean():.1f}s")
    print(f"Std/mean ratio: {intervals_bursty.std()/intervals_bursty.mean():.3f} (should be >1.0, more variable/bursty than Poisson)")

    print("\n=== Weibull traffic check (regular, shape=2.0) ===")
    intervals_regular = weibull_traffic_intervals(N, mean_interval_s=3600, shape=2.0, rng=rng)
    print(f"Target mean: 3600s, empirical mean: {intervals_regular.mean():.1f}s")
    print(f"Std/mean ratio: {intervals_regular.std()/intervals_regular.mean():.3f} (should be <1.0, more regular than Poisson)")

    print("\n=== Bursty ON-OFF check ===")
    burst = bursty_on_off_intervals(1000, on_interval_s=60, off_interval_s=7200,
                                     on_duration_steps=5, off_duration_steps=20, rng=rng)
    print(f"First 30 intervals (should show short-then-long pattern):")
    print("  ", [round(x,0) for x in burst[:30]])

    print("\n=== Markov coverage check (sticky CE-A) ===")
    sticky_matrix = np.array([
        [0.90, 0.07, 0.03],  # from CE-A: 90% stay, small chance of moving
        [0.15, 0.70, 0.15],  # from CE-B
        [0.05, 0.15, 0.80],  # from CE-C
    ])
    seq = markov_coverage_sequence(10000, sticky_matrix, initial_ce=0, rng=rng)
    seq = np.array(seq)
    print(f"Time spent in CE-A: {(seq==0).mean()*100:.1f}%")
    print(f"Time spent in CE-B: {(seq==1).mean()*100:.1f}%")
    print(f"Time spent in CE-C: {(seq==2).mean()*100:.1f}%")
    # count actual switches vs total steps -- should be much less than
    # the ~67% switch rate a uniform-random reassignment would produce
    switches = (seq[1:] != seq[:-1]).sum()
    print(f"Switch rate: {switches/len(seq)*100:.1f}% (should be well under 33%, confirming 'stickiness')")

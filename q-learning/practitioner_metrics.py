#!/usr/bin/env python3
"""
Point 14 (reviewer feedback): practitioner-friendly metrics.

Reports lifetime improvement %, delay change %, reconfiguration counts,
and cumulative battery-life gain -- not just average reward -- using
your existing, already-tested evaluate.py functions directly, on your
real, current trained agent.

Run from your qlearning/ folder: python3 practitioner_metrics.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from agent import QLearningAgent
from evaluate import run_adaptive, run_static, find_best_static_action, summarize
from scenarios import SCENARIOS
from environment import CE_MODES, TRAFFIC_BUCKETS

N_STATES = len(CE_MODES) * len(TRAFFIC_BUCKETS)  # dynamic -- adapts automatically
                                                    # if the state space changes again


def reconfiguration_count(results):
    """Counts actual step-to-step changes in (T3412, T3324) -- not just
    the number of distinct actions used, but how many times the device
    would genuinely need to be reconfigured across the scenario."""
    changes = 0
    for i in range(1, len(results)):
        prev = (results[i-1].T3412, results[i-1].T3324)
        curr = (results[i].T3412, results[i].T3324)
        if prev != curr:
            changes += 1
    return changes


agent = QLearningAgent(n_states=N_STATES, n_actions=42)
agent.load("results/q_table.npy")

print(f"{'Scenario':<28}{'Lifetime vs naive':<26}{'Lifetime vs best':<26}"
      f"{'Delay vs naive':<24}{'Reconfigs':<12}")

all_lifetime_vs_naive = []
all_lifetime_vs_best = []
all_delay_vs_naive = []
all_reconfigs = []

for key, scenario in SCENARIOS.items():
    adaptive = run_adaptive(agent, scenario)
    best_T3412, best_T3324 = find_best_static_action(scenario)
    best_static = run_static(best_T3412, best_T3324, scenario)
    naive_static = run_static(3600, 20, scenario)

    a_sum = summarize(adaptive)
    b_sum = summarize(best_static)
    n_sum = summarize(naive_static)

    lt_vs_naive_pct = (a_sum["avg_lifetime_yr"] - n_sum["avg_lifetime_yr"]) / n_sum["avg_lifetime_yr"] * 100
    lt_vs_best_pct = (a_sum["avg_lifetime_yr"] - b_sum["avg_lifetime_yr"]) / b_sum["avg_lifetime_yr"] * 100
    dl_vs_naive_pct = (a_sum["avg_delay_s"] - n_sum["avg_delay_s"]) / n_sum["avg_delay_s"] * 100
    reconfigs = reconfiguration_count(adaptive)

    all_lifetime_vs_naive.append(lt_vs_naive_pct)
    all_lifetime_vs_best.append(lt_vs_best_pct)
    all_delay_vs_naive.append(dl_vs_naive_pct)
    all_reconfigs.append(reconfigs)

    # Absolute numbers shown alongside every percentage -- a percentage
    # alone can't be sanity-checked; the raw values make it self-explanatory.
    lt_naive_str = f"{lt_vs_naive_pct:+.1f}% ({a_sum['avg_lifetime_yr']:.2f}v{n_sum['avg_lifetime_yr']:.2f}yr)"
    lt_best_str = f"{lt_vs_best_pct:+.1f}% ({a_sum['avg_lifetime_yr']:.2f}v{b_sum['avg_lifetime_yr']:.2f}yr)"
    dl_naive_str = f"{dl_vs_naive_pct:+.1f}% ({a_sum['avg_delay_s']:.0f}v{n_sum['avg_delay_s']:.0f}s)"

    print(f"{scenario.name:<28}{lt_naive_str:<26}{lt_best_str:<26}{dl_naive_str:<24}{reconfigs:>10}")

print("\n" + "="*70)
print("OVERALL (mean across all scenarios)")
print("="*70)
print(f"  Lifetime improvement vs naive static: {np.mean(all_lifetime_vs_naive):+.1f}%")
print(f"  Lifetime vs best-possible static:     {np.mean(all_lifetime_vs_best):+.1f}%")
print(f"  Delay change vs naive static:         {np.mean(all_delay_vs_naive):+.1f}%")
print(f"  Average reconfigurations per scenario: {np.mean(all_reconfigs):.1f}")
print()
print("Note on 'signalling overhead': each reconfiguration corresponds to a real")
print("network exchange (TAU/RRC reconfiguration) to update the device's timers.")
print("The reconfiguration count above is reported directly as the practitioner-")
print("facing proxy for this cost, rather than assuming an unverified per-message")
print("energy/byte figure not established elsewhere in this project.")
"""


PURPOSE
───────
Loads the trained Q-table (results/q_table.npy) and runs it, in greedy
(no-exploration) mode, through every named scenario in scenarios.py.
For each scenario, also computes the BEST POSSIBLE fixed (T3412,T3324)
configuration — searched across the full action grid — as the static
baseline. This is the file that directly produces the RQ4/Objective 4
numbers: "how does the adaptive model compare to static benchmark
configurations in terms of energy consumption [lifetime] and latency
[delay]."



A second, more naive static baseline (T3412=3600s, T3324=20s — a
commonly-used mid-range default seen throughout this project's own
NS-3 sweep defaults) is ALSO reported, to show how much a typical,
non-optimized static choice underperforms both the adaptive agent AND
the best-possible static choice.
"""

import os
import csv
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict

from smp_engine import compute_smp
from environment import (
    ACTIONS, N_ACTIONS, traffic_to_bucket, CE_MODES, TRAFFIC_BUCKETS,
    normalize_lifetime, normalize_delay, DELAY_WEIGHT_BY_BUCKET,
)
from agent import QLearningAgent
from scenarios import SCENARIOS, Scenario, get_scenario

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

W_LIFETIME = 0.7   # must match environment.py's NbIotEnv default
W_DELAY = 1.0       # must match environment.py's NbIotEnv default

NAIVE_STATIC_ACTION = (3600, 20)  # a common, non-optimized mid-range choice


@dataclass
class StepResult:
    ce_mode: int
    interval_s: float
    T3412: int
    T3324: int
    lifetime_yr: float
    delay_s: float
    reward: float
    tau_fired: bool


def evaluate_action_on_condition(T3412: int, T3324: int, ce_mode: int,
                                 interval_s: float) -> StepResult:
    """Compute the TRUE outcome of a specific action under a specific,
    exact condition (not a bucket abstraction) — the core evaluation
    primitive used by both the adaptive and static evaluation paths."""
    result = compute_smp(T3412, T3324, interval_s, ce_mode)
    bucket = traffic_to_bucket(interval_s)
    norm_lt = normalize_lifetime(result.lifetime_yr)
    norm_dl = normalize_delay(result.delay_seconds)
    w_delay_effective = W_DELAY * DELAY_WEIGHT_BY_BUCKET[bucket]
    reward = W_LIFETIME * norm_lt - w_delay_effective * norm_dl
    return StepResult(
        ce_mode=ce_mode, interval_s=interval_s, T3412=T3412, T3324=T3324,
        lifetime_yr=result.lifetime_yr, delay_s=result.delay_seconds,
        reward=reward, tau_fired=result.tau_fired,
    )


def run_adaptive(agent: QLearningAgent, scenario: Scenario) -> List[StepResult]:
    """Step through the scenario using the trained agent's GREEDY policy
    (no exploration — this is evaluation, not training)."""
    results = []
    for ce_mode, interval_s in scenario.flatten():
        bucket = traffic_to_bucket(interval_s)
        state = ce_mode * len(TRAFFIC_BUCKETS) + bucket;  # must match environment.py's _state_index
        action_idx = agent.choose_action(state, greedy=True)
        T3412, T3324 = ACTIONS[action_idx]
        results.append(evaluate_action_on_condition(T3412, T3324, ce_mode, interval_s))
    return results


def run_static(T3412: int, T3324: int, scenario: Scenario) -> List[StepResult]:
    """Apply ONE FIXED action across the entire scenario — the static
    baseline's defining constraint: it cannot change per step."""
    results = []
    for ce_mode, interval_s in scenario.flatten():
        results.append(evaluate_action_on_condition(T3412, T3324, ce_mode, interval_s))
    return results


def find_best_static_action(scenario: Scenario) -> tuple:
    """Search the FULL action grid for the single (T3412,T3324) pair that
    performs best ON AVERAGE across the scenario's entire condition
    sequence. This is the fairest possible static baseline — see module
    docstring for why."""
    best_action = None
    best_avg_reward = -float("inf")
    for T3412, T3324 in ACTIONS:
        results = run_static(T3412, T3324, scenario)
        avg_reward = np.mean([r.reward for r in results])
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_action = (T3412, T3324)
    return best_action


def summarize(results: List[StepResult]) -> Dict:
    return {
        "avg_lifetime_yr": float(np.mean([r.lifetime_yr for r in results])),
        "avg_delay_s": float(np.mean([r.delay_s for r in results])),
        "avg_reward": float(np.mean([r.reward for r in results])),
        "total_reward": float(np.sum([r.reward for r in results])),
        "tau_fired_pct": 100.0 * np.mean([r.tau_fired for r in results]),
        "n_steps": len(results),
    }


def evaluate_all_scenarios(agent: QLearningAgent, verbose: bool = True) -> List[Dict]:
    """Run every registered scenario through: the adaptive agent, the
    best-possible static baseline, and a naive static baseline. Returns
    one summary row per (scenario, method) combination."""
    rows = []

    for scenario_key, scenario in SCENARIOS.items():
        if verbose:
            print(f"\n[SCENARIO] {scenario.name}  ({scenario.total_steps()} steps)")

        # ── Adaptive agent ──────────────────────────────────────────────────
        adaptive_results = run_adaptive(agent, scenario)
        adaptive_summary = summarize(adaptive_results)
        adaptive_actions_used = sorted(set((r.T3412, r.T3324) for r in adaptive_results))

        # ── Best possible static baseline (searched, not guessed) ──────────
        best_T3412, best_T3324 = find_best_static_action(scenario)
        best_static_results = run_static(best_T3412, best_T3324, scenario)
        best_static_summary = summarize(best_static_results)

        # ── Naive static baseline (a common, non-optimized default) ────────
        naive_results = run_static(*NAIVE_STATIC_ACTION, scenario)
        naive_summary = summarize(naive_results)

        for method, summary, action_desc in [
            ("adaptive", adaptive_summary, f"varies ({len(adaptive_actions_used)} distinct actions)"),
            ("best_static", best_static_summary, f"T3412={best_T3412},T3324={best_T3324}"),
            ("naive_static", naive_summary, f"T3412={NAIVE_STATIC_ACTION[0]},T3324={NAIVE_STATIC_ACTION[1]}"),
        ]:
            row = {"scenario": scenario_key, "method": method, "action": action_desc}
            row.update(summary)
            rows.append(row)

        if verbose:
            print(f"  adaptive:     avg_lifetime={adaptive_summary['avg_lifetime_yr']:8.2f}yr  "
                  f"avg_delay={adaptive_summary['avg_delay_s']:9.1f}s  "
                  f"avg_reward={adaptive_summary['avg_reward']:+.4f}  "
                  f"actions_used={len(adaptive_actions_used)}")
            print(f"  best_static:  avg_lifetime={best_static_summary['avg_lifetime_yr']:8.2f}yr  "
                  f"avg_delay={best_static_summary['avg_delay_s']:9.1f}s  "
                  f"avg_reward={best_static_summary['avg_reward']:+.4f}  "
                  f"(T3412={best_T3412},T3324={best_T3324})")
            print(f"  naive_static: avg_lifetime={naive_summary['avg_lifetime_yr']:8.2f}yr  "
                  f"avg_delay={naive_summary['avg_delay_s']:9.1f}s  "
                  f"avg_reward={naive_summary['avg_reward']:+.4f}")

            gap_vs_best = adaptive_summary['avg_reward'] - best_static_summary['avg_reward']
            gap_vs_naive = adaptive_summary['avg_reward'] - naive_summary['avg_reward']
            print(f"  -> adaptive vs best_static reward gap:  {gap_vs_best:+.4f}")
            print(f"  -> adaptive vs naive_static reward gap: {gap_vs_naive:+.4f}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained NB-IoT adaptive agent")
    parser.add_argument("--qtable", type=str,
                        default=os.path.join(RESULTS_DIR, "q_table.npy"))
    args = parser.parse_args()

    if not os.path.exists(args.qtable):
        raise FileNotFoundError(
            f"Q-table not found at {args.qtable} — run train.py first."
        )

    agent = QLearningAgent(n_states=len(CE_MODES)*len(TRAFFIC_BUCKETS), n_actions=N_ACTIONS)
    agent.load(args.qtable)
    print(f"[LOAD] Q-table loaded from {args.qtable}")

    rows = evaluate_all_scenarios(agent)

    out_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SAVED] {out_path}")

    # ── Overall summary across all scenarios ──────────────────────────────
    print("\n" + "="*70)
    print("OVERALL SUMMARY (averaged across all scenarios)")
    print("="*70)
    for method in ["adaptive", "best_static", "naive_static"]:
        method_rows = [r for r in rows if r["method"] == method]
        avg_lt = np.mean([r["avg_lifetime_yr"] for r in method_rows])
        avg_dl = np.mean([r["avg_delay_s"] for r in method_rows])
        avg_rw = np.mean([r["avg_reward"] for r in method_rows])
        print(f"  {method:14s}  avg_lifetime={avg_lt:8.2f}yr  "
              f"avg_delay={avg_dl:9.1f}s  avg_reward={avg_rw:+.4f}")
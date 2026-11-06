
import sys
sys.path.insert(0, ".")
import csv
import os
import argparse
import numpy as np
from environment import CE_MODES, TRAFFIC_BUCKETS
from agent import QLearningAgent
from evaluate import run_adaptive, run_static, find_best_static_action, summarize
from scenarios import Scenario, ScenarioStep
from generators import (
    poisson_traffic_intervals, weibull_traffic_intervals,
    bursty_on_off_intervals, markov_coverage_sequence,
)

N_SCENARIOS_PER_TYPE = 1000   
STEPS_PER_SCENARIO = 20      

STICKY_COVERAGE_MATRIX = np.array([
    [0.90, 0.07, 0.03],
    [0.15, 0.70, 0.15],
    [0.05, 0.15, 0.80],
])


def held_coverage_sequence(n_steps, rng, min_hold=2, max_hold=8):
    sequence = []
    current_ce = int(rng.integers(0, 3))
    while len(sequence) < n_steps:
        hold = int(rng.integers(min_hold, max_hold + 1))
        sequence.extend([current_ce] * hold)
        current_ce = int(rng.integers(0, 3))
    return sequence[:n_steps]


def held_traffic_regime(n_steps, interval_generator_fn, rng, min_hold=2, max_hold=8):
    intervals = []
    while len(intervals) < n_steps:
        hold = int(rng.integers(min_hold, max_hold + 1))
        phase_mean = float(rng.uniform(300, 43200))  # phase's typical reporting rate
        phase_intervals = interval_generator_fn(hold, phase_mean, rng)
        intervals.extend(phase_intervals)
    return intervals[:n_steps]


def build_random_scenario(name, ce_sequence, intervals):
    steps = [
        ScenarioStep(ce_mode=int(ce), traffic_interval_s=float(iv), duration_steps=1)
        for ce, iv in zip(ce_sequence, intervals)
    ]
    return Scenario(name=name, description="Randomly generated", steps=steps)


def run_batch(agent, scenario_generator_fn, label, n_scenarios, rng, csv_writer, step_csv_writer):
    adaptive_rewards, best_static_rewards, naive_static_rewards = [], [], []
    for i in range(n_scenarios):
        scenario = scenario_generator_fn(rng)
        adaptive_results = run_adaptive(agent, scenario)
        best_T3412, best_T3324 = find_best_static_action(scenario)
        best_static_results = run_static(best_T3412, best_T3324, scenario)
        naive_static_results = run_static(3600, 20, scenario)

        adaptive_summary = summarize(adaptive_results)
        best_static_summary = summarize(best_static_results)
        naive_static_summary = summarize(naive_static_results)

        adaptive_rewards.append(adaptive_summary["avg_reward"])
        best_static_rewards.append(best_static_summary["avg_reward"])
        naive_static_rewards.append(naive_static_summary["avg_reward"])

        csv_writer.writerow({
            "generator_type": label,
            "scenario_index": i,
            "adaptive_reward": adaptive_summary["avg_reward"],
            "best_static_reward": best_static_summary["avg_reward"],
            "naive_static_reward": naive_static_summary["avg_reward"],
            "best_static_T3412": best_T3412,
            "best_static_T3324": best_T3324,
            "adaptive_beats_best_static": adaptive_summary["avg_reward"] > best_static_summary["avg_reward"],
        })

        # full per-step trace -- what conditions this specific scenario
        # actually went through, answering "can I see exactly what ran"
        for step_idx, (ce_mode, interval_s) in enumerate(scenario.flatten()):
            step_csv_writer.writerow({
                "generator_type": label,
                "scenario_index": i,
                "step_index": step_idx,
                "ce_mode": ce_mode,
                "traffic_interval_s": round(interval_s, 2),
            })

    gaps = [a - b for a, b in zip(adaptive_rewards, best_static_rewards)]
    mean_gap = np.mean(gaps)
    pct_of_best = (np.mean(adaptive_rewards) / np.mean(best_static_rewards)) * 100

    beats_best = sum(g > 0 for g in gaps)
    losses = [g for g in gaps if g < 0]
    mean_loss_when_losing = np.mean(losses) if losses else 0.0

    print(f"\n=== {label} (n={n_scenarios}) ===")
    print(f"  Adaptive:      mean={np.mean(adaptive_rewards):+.4f}  std={np.std(adaptive_rewards):.4f}  "
          f"worst={np.min(adaptive_rewards):+.4f}")
    print(f"  Best static:   mean={np.mean(best_static_rewards):+.4f}  std={np.std(best_static_rewards):.4f}")
    print(f"  Naive static:  mean={np.mean(naive_static_rewards):+.4f}  std={np.std(naive_static_rewards):.4f}")
    print()
    print(f"  How close adaptive gets to the hindsight-optimal baseline:")
    print(f"    Adaptive achieves {pct_of_best:.1f}% of best-static's average reward")
    print(f"    Mean gap (adaptive - best_static): {mean_gap:+.4f}")
    print()
    print(f"  Supporting detail: Binary win/loss count:")
    print(f"    Adaptive beats best-static in {beats_best}/{n_scenarios} scenarios ({beats_best/n_scenarios*100:.1f}%)")
    print(f"    When adaptive DOES lose, average loss size: {mean_loss_when_losing:+.4f} "
          f"(small = losses are narrow, not severe)")
    return adaptive_rewards, best_static_rewards, naive_static_rewards


def poisson_scenario(rng):
    ce_seq = held_coverage_sequence(STEPS_PER_SCENARIO, rng)
    intervals = held_traffic_regime(STEPS_PER_SCENARIO,
        lambda n, mean, r: poisson_traffic_intervals(n, mean, r), rng)
    return build_random_scenario("random_poisson", ce_seq, intervals)


def weibull_bursty_scenario(rng):
    ce_seq = held_coverage_sequence(STEPS_PER_SCENARIO, rng)
    intervals = held_traffic_regime(STEPS_PER_SCENARIO,
        lambda n, mean, r: weibull_traffic_intervals(n, mean, 0.5, r), rng)
    return build_random_scenario("random_weibull_bursty", ce_seq, intervals)


def onoff_scenario(rng):
    ce_seq = markov_coverage_sequence(STEPS_PER_SCENARIO, STICKY_COVERAGE_MATRIX,
                                       initial_ce=int(rng.integers(0,3)), rng=rng)
    intervals = bursty_on_off_intervals(STEPS_PER_SCENARIO, on_interval_s=300, off_interval_s=7200,
                                         on_duration_steps=4, off_duration_steps=8, rng=rng)
    return build_random_scenario("random_onoff_sticky_coverage", ce_seq, intervals)


def verify_table_from_csv(summary_path="results/random_scenarios_results.csv"):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"{summary_path} not found -- run this script without --verify first "
            f"to generate it.")

    rows_by_type = {}
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_type = row["generator_type"]
            rows_by_type.setdefault(gen_type, []).append(row)

    print("=" * 90)
    print("VERIFICATION: Table Headline numbers, recomputed directly from random_scenarios.csv")
    print("=" * 90)
    print(f"{'Traffic model':<50}{'% of best-static':<18}{'Mean gap':<12}{'Win rate'}")
    for gen_type, rows in rows_by_type.items():
        adaptive = np.array([float(r["adaptive_reward"]) for r in rows])
        best_static = np.array([float(r["best_static_reward"]) for r in rows])
        pct = adaptive.mean() / best_static.mean() * 100
        gap = adaptive.mean() - best_static.mean()
        win_rate = np.mean([r["adaptive_beats_best_static"] == "True" for r in rows]) * 100
        print(f"{gen_type:<50}{pct:<18.1f}{gap:<12.4f}{win_rate:.1f}%")
    print(f"\n(n={len(rows)} scenarios per traffic model, {sum(len(r) for r in rows_by_type.values())} total)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stochastic scenario results, or verify the headline "
                     "table numbers from an already-generated CSV.")
    parser.add_argument("--verify", action="store_true",
                         help="Skip generation entirely -- just recompute and print "
                              "Table IV's numbers from the existing results CSV.")
    args = parser.parse_args()

    if args.verify:
        verify_table_from_csv()
        sys.exit(0)

    rng = np.random.default_rng(42)

    agent = QLearningAgent(n_states=len(CE_MODES)*len(TRAFFIC_BUCKETS), n_actions=42)
    agent.load("results/q_table.npy")

    os.makedirs("results", exist_ok=True)
    summary_path = "results/random_scenarios_results.csv"
    steps_path = "results/random_scenarios_steps.csv"

    with open(summary_path, "w", newline="") as f_summary, open(steps_path, "w", newline="") as f_steps:
        summary_fields = ["generator_type", "scenario_index", "adaptive_reward", "best_static_reward",
                           "naive_static_reward", "best_static_T3412", "best_static_T3324", "adaptive_beats_best_static"]
        step_fields = ["generator_type", "scenario_index", "step_index", "ce_mode", "traffic_interval_s"]
        summary_writer = csv.DictWriter(f_summary, fieldnames=summary_fields)
        step_writer = csv.DictWriter(f_steps, fieldnames=step_fields)
        summary_writer.writeheader()
        step_writer.writeheader()

        run_batch(agent, poisson_scenario, "Poisson traffic, uniform-random coverage", N_SCENARIOS_PER_TYPE, rng, summary_writer, step_writer)
        run_batch(agent, weibull_bursty_scenario, "Bursty Weibull traffic, uniform-random coverage", N_SCENARIOS_PER_TYPE, rng, summary_writer, step_writer)
        run_batch(agent, onoff_scenario, "ON-OFF bursty traffic, sticky Markov coverage", N_SCENARIOS_PER_TYPE, rng, summary_writer, step_writer)

    print(f"\n[SAVED] Per-scenario summary: {summary_path}")
    print(f"[SAVED] Full per-step trace (every condition each scenario went through): {steps_path}")
    print(f"[SAVED] Total scenarios: {N_SCENARIOS_PER_TYPE * 3}")
    print()
    verify_table_from_csv(summary_path)
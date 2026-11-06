

import os
import argparse

from smp_engine import compute_smp
from environment import (
    ACTIONS, N_ACTIONS, CE_MODES, TRAFFIC_BUCKETS,
    traffic_to_bucket,
)
from agent import QLearningAgent
from scenarios import SCENARIOS, get_scenario, list_scenarios

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CE_LABELS = {0: "CE-A (Good coverage)", 1: "CE-B (Medium coverage)", 2: "CE-C (Poor coverage)"}


def load_trained_agent() -> QLearningAgent:
    qtable_path = os.path.join(RESULTS_DIR, "q_table.npy")
    if not os.path.exists(qtable_path):
        raise FileNotFoundError(
            f"No trained agent found at {qtable_path} — run train.py first."
        )
    agent = QLearningAgent(n_states=len(CE_MODES)*len(TRAFFIC_BUCKETS), n_actions=N_ACTIONS)
    agent.load(qtable_path)
    return agent


def query_agent(agent: QLearningAgent, ce_mode: int, interval_s: float) -> dict:
    """
    Ask the trained agent what it recommends for a specific condition,
    then compute the TRUE resulting lifetime/delay using the exact
    interval given (not a bucket abstraction) — same approach as
    evaluate.py, for consistency.
    """
    bucket = traffic_to_bucket(interval_s)
    state = ce_mode * len(TRAFFIC_BUCKETS) + bucket
    action_idx = agent.choose_action(state, greedy=True)
    T3412, T3324 = ACTIONS[action_idx]

    result = compute_smp(T3412, T3324, interval_s, ce_mode)

    return {
        "ce_mode": ce_mode, "interval_s": interval_s, "bucket": bucket,
        "T3412": T3412, "T3324": T3324,
        "lifetime_yr": result.lifetime_yr, "delay_s": result.delay_seconds,
        "tau_fired": result.tau_fired,
    }


def format_result_line(r: dict, changed: bool = False, step_num: int = None) -> str:
    bucket_name = TRAFFIC_BUCKETS[r["bucket"]][2]
    prefix = f"step {step_num:2d}  " if step_num is not None else "  "
    marker = "  <<< ACTION CHANGED" if changed else ""
    return (f"{prefix}CE={r['ce_mode']} interval={r['interval_s']:6.0f}s "
            f"({bucket_name:14s})  ->  T3412={r['T3412']:6d}s  T3324={r['T3324']:2d}s  "
            f"|  lifetime={r['lifetime_yr']:7.2f}yr  delay={r['delay_s']:8.1f}s{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYBACK MODE
# ─────────────────────────────────────────────────────────────────────────────

def playback_scenario(agent: QLearningAgent, scenario_name: str) -> None:
    scenario = get_scenario(scenario_name)
    print(f"\n{'='*78}")
    print(f"PLAYBACK: {scenario.name}")
    print(f"{'='*78}")
    print(f"{scenario.description}\n")

    prev_action = None
    n_changes = 0

    for i, (ce_mode, interval_s) in enumerate(scenario.flatten()):
        r = query_agent(agent, ce_mode, interval_s)
        current_action = (r["T3412"], r["T3324"])
        changed = prev_action is not None and current_action != prev_action
        if changed:
            n_changes += 1
        print(format_result_line(r, changed=changed, step_num=i))
        prev_action = current_action

    print(f"\n{'-'*78}")
    print(f"Total steps: {scenario.total_steps()}   Action changes observed: {n_changes}")
    if n_changes > 0:
        print(f"-> The agent visibly changed its recommendation {n_changes} time(s) "
              f"as conditions changed within this single scenario.")
    else:
        print(f"-> The agent's recommendation stayed constant throughout — this "
              f"scenario's conditions did not change enough to warrant a "
              f"different timer choice (a legitimate, correct outcome; see "
              f"'mixed_stress_test' or 'agricultural_irrigation' for scenarios "
              f"with visible transitions).")


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

def interactive_loop(agent: QLearningAgent) -> None:
    print(f"\n{'='*78}")
    print("INTERACTIVE MODE — query the trained agent live")
    print(f"{'='*78}")
    print("At each prompt, enter a coverage mode and a traffic interval to see")
    print("what the trained agent recommends. Type 'q' at any prompt to quit.\n")
    print("Coverage modes:  0 = CE-A (Good)   1 = CE-B (Medium)   2 = CE-C (Poor)")
    print("Traffic interval: any value in seconds (e.g. 300 for very frequent,")
    print("                   86400 for once a day)\n")

    history = []

    while True:
        ce_in = input("Coverage mode [0/1/2] (or 'q' to quit): ").strip()
        if ce_in.lower() == "q":
            break
        try:
            ce_mode = int(ce_in)
            if ce_mode not in CE_MODES:
                print("  Please enter 0, 1, or 2.\n")
                continue
        except ValueError:
            print("  Please enter a number (0, 1, or 2).\n")
            continue

        interval_in = input("Traffic interval in seconds (or 'q' to quit): ").strip()
        if interval_in.lower() == "q":
            break
        try:
            interval_s = float(interval_in)
            if interval_s <= 0:
                print("  Interval must be positive.\n")
                continue
        except ValueError:
            print("  Please enter a number.\n")
            continue

        r = query_agent(agent, ce_mode, interval_s)
        changed = len(history) > 0 and (r["T3412"], r["T3324"]) != history[-1]
        print("  " + format_result_line(r, changed=changed))
        if r["tau_fired"]:
            print("  (Note: TAU would fire under this configuration — traffic "
                  "interval exceeds T3412)")
        print()
        history.append((r["T3412"], r["T3324"]))

    if len(history) > 1:
        distinct = len(set(history))
        print(f"\nSession summary: {len(history)} queries, "
              f"{distinct} distinct recommendation(s) given.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate the trained NB-IoT adaptive agent's dynamic behaviour"
    )
    parser.add_argument("--playback", type=str, default=None,
                        help="Play back a named scenario (see --list for options)")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive query mode")
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios and exit")
    parser.add_argument("--all", action="store_true",
                        help="Play back every registered scenario in sequence")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        list_scenarios()
        raise SystemExit(0)

    agent = load_trained_agent()
    print(f"[LOAD] Trained agent loaded from {os.path.join(RESULTS_DIR, 'q_table.npy')}")

    if args.all:
        for key in SCENARIOS:
            playback_scenario(agent, key)
    elif args.playback:
        playback_scenario(agent, args.playback)
    elif args.interactive:
        interactive_loop(agent)
    else:
        print("\nNo mode selected. Use one of:")
        print("  python3 demo.py --list                       # see available scenarios")
        print("  python3 demo.py --playback agricultural_irrigation")
        print("  python3 demo.py --all                          # play back every scenario")
        print("  python3 demo.py --interactive                  # explore live, type your own conditions")
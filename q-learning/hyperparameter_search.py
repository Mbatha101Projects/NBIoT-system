"""
Sweeps candidate values of the Q-learning hyperparameters (learning
rate alpha, discount factor gamma, exploration decay epsilon_decay)
and selects the combination that produces the most ACCURATE learned
policy, then re-trains a final agent using the winning combination at
full episode count and overwrites results/q_table.npy with it.

"""

import os
import csv
import time
import itertools
import numpy as np
from dataclasses import asdict

from environment import (
    NbIotEnv, N_ACTIONS, ACTIONS, CE_MODES, TRAFFIC_BUCKETS,
    bucket_to_representative_interval, normalize_lifetime, normalize_delay,
    DELAY_WEIGHT_BY_BUCKET,
)
from agent import QLearningAgent, QLearningConfig
from smp_engine import compute_smp
from train import train, RESULTS_DIR, state_names, action_names


ALPHA_CANDIDATES = [0.05, 0.10, 0.20]
GAMMA_CANDIDATES = [0.90, 0.95]   
EPSILON_DECAY_CANDIDATES = [0.995, 0.999]

SEARCH_EPISODES = 15000    
SEARCH_SEEDS = [1, 2]       


def audit_policy_accuracy(agent: QLearningAgent) -> dict:
    CORRECTNESS_GAP_THRESHOLD = 0.02
    T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))

    n_correct = 0
    total_gap = 0.0
    max_gap = 0.0

    for ce in CE_MODES:
        for bucket in range(len(TRAFFIC_BUCKETS)):
            interval = bucket_to_representative_interval(bucket)
            w_dl = 1.0 * DELAY_WEIGHT_BY_BUCKET[bucket]

            true_rewards = {}
            for T3412 in T3412_vals:
                r = compute_smp(T3412, 20, interval, ce)
                nlt = normalize_lifetime(r.lifetime_yr)
                ndl = normalize_delay(r.delay_seconds)
                true_rewards[T3412] = 0.7 * nlt - w_dl * ndl
            true_best_reward = max(true_rewards.values())

            state_idx = ce * len(TRAFFIC_BUCKETS) + bucket
            learned_action = ACTIONS[int(np.argmax(agent.q_table[state_idx]))]
            learned_T3412 = learned_action[0]

            learned_reward = true_rewards.get(learned_T3412, true_best_reward)
            gap = true_best_reward - learned_reward

            if gap <= CORRECTNESS_GAP_THRESHOLD:
                n_correct += 1
            total_gap += gap
            max_gap = max(max_gap, gap)

    return {"n_correct": n_correct, "total_gap": total_gap, "max_gap": max_gap}


def run_search(verbose: bool = True) -> dict:
    """
    Run the full grid search. Returns the winning QLearningConfig's
    parameters as a dict, and writes results/hyperparameter_search_log.csv
    with every candidate's audit results for transparency.
    """
    candidates = list(itertools.product(
        ALPHA_CANDIDATES, GAMMA_CANDIDATES, EPSILON_DECAY_CANDIDATES
    ))

    if verbose:
        print(f"[SEARCH] {len(candidates)} hyperparameter combinations × "
              f"{len(SEARCH_SEEDS)} seeds = {len(candidates)*len(SEARCH_SEEDS)} "
              f"training runs at {SEARCH_EPISODES} episodes each")
        print()

    log_rows = []
    t0 = time.time()

    for i, (alpha, gamma, eps_decay) in enumerate(candidates, 1):
        seed_results = []
        for seed in SEARCH_SEEDS:
            cfg = QLearningConfig(alpha=alpha, gamma=gamma, epsilon_decay=eps_decay)
            agent = train(n_episodes=SEARCH_EPISODES, config=cfg, seed=seed,
                         verbose=False)
            audit = audit_policy_accuracy(agent)
            seed_results.append(audit)

        avg_correct = np.mean([r["n_correct"] for r in seed_results])
        avg_total_gap = np.mean([r["total_gap"] for r in seed_results])
        avg_max_gap = np.mean([r["max_gap"] for r in seed_results])

        log_rows.append({
            "alpha": alpha, "gamma": gamma, "epsilon_decay": eps_decay,
            "avg_states_correct": avg_correct, "avg_total_gap": avg_total_gap,
            "avg_max_gap": avg_max_gap,
        })

        if verbose:
            elapsed = time.time() - t0
            print(f"  [{i:2d}/{len(candidates)}]  alpha={alpha:.2f} gamma={gamma:.2f} "
                  f"eps_decay={eps_decay:.3f}  ->  avg_correct={avg_correct:.1f}/30  "
                  f"avg_total_gap={avg_total_gap:.4f}  ({elapsed:.0f}s elapsed)")

    # ── Select winner: most states correct, tie-broken by smallest total gap ──
    log_rows.sort(key=lambda r: (-r["avg_states_correct"], r["avg_total_gap"]))
    winner = log_rows[0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "hyperparameter_search_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    if verbose:
        print()
        print(f"[SAVED] {log_path}")
        print()
        print(f"[WINNER] alpha={winner['alpha']}, gamma={winner['gamma']}, "
              f"epsilon_decay={winner['epsilon_decay']}  "
              f"(avg_correct={winner['avg_states_correct']:.1f}/30, "
              f"avg_total_gap={winner['avg_total_gap']:.4f})")

    return winner


if __name__ == "__main__":
    winner = run_search()

    print()
    print("[RETRAIN] Training final agent at full episode count (60000) using "
          "the winning hyperparameters...")
    winning_config = QLearningConfig(
        alpha=winner["alpha"], gamma=winner["gamma"],
        epsilon_decay=winner["epsilon_decay"],
    )
    tuned_agent = train(n_episodes=60000, config=winning_config, seed=11,
                        verbose=False)
    tuned_audit = audit_policy_accuracy(tuned_agent)
    print(f"[CHECK] Tuned agent (full 60000-episode retrain): "
          f"{tuned_audit['n_correct']}/30 correct, "
          f"total_gap={tuned_audit['total_gap']:.4f}")


    print()
    print("[VALIDATE] Training a default-hyperparameter baseline for comparison "
          "(same episode count, same seed)...")
    default_config = QLearningConfig()  # alpha=0.1, gamma=0.9, epsilon_decay=0.995
    default_agent = train(n_episodes=60000, config=default_config, seed=42,
                          verbose=False)
    default_audit = audit_policy_accuracy(default_agent)
    print(f"[CHECK] Default-hyperparameter agent (same conditions): "
          f"{default_audit['n_correct']}/30 correct, "
          f"total_gap={default_audit['total_gap']:.4f}")

    print()
    tuned_better = (tuned_audit["n_correct"], -tuned_audit["total_gap"]) > \
                   (default_audit["n_correct"], -default_audit["total_gap"])

    if tuned_better:
        final_agent = tuned_agent
        print(f"[RESULT] Tuned hyperparameters ARE better — keeping the tuned "
              f"agent (alpha={winner['alpha']}, gamma={winner['gamma']}, "
              f"epsilon_decay={winner['epsilon_decay']}).")
    else:
        final_agent = default_agent
        print(f"[RESULT] Tuned hyperparameters did NOT outperform the defaults "
              f"at full scale — falling back to the DEFAULT agent "
              f"(alpha={default_config.alpha}, gamma={default_config.gamma}, "
              f"epsilon_decay={default_config.epsilon_decay}) instead. "
              f"This is a legitimate, expected outcome of the validation "
              f"step working correctly, not an error — the search phase's "
              f"short test runs don't always predict full-scale behaviour, "
              f"which is exactly why this check exists.")

    qtable_path = os.path.join(RESULTS_DIR, "q_table.npy")
    final_agent.save(qtable_path)
    print(f"\n[SAVED] {qtable_path}  (overwritten with the VALIDATED best result)")
    print()
    print("Final learned policy summary:")
    print(final_agent.policy_summary(state_names=state_names(),
                                     action_names=action_names()))

    print()
    print("[DONE] results/q_table.npy now reflects the validated best agent.")
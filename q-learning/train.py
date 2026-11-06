

import os
import csv
import argparse
import numpy as np

from environment import NbIotEnv, N_ACTIONS, TRAFFIC_BUCKETS, CE_MODES
from agent import QLearningAgent, QLearningConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def state_names() -> list:
    """Human-readable labels for each state index, matching
    environment.py's _state_index() ordering (ce_mode outer, bucket inner),
    used for policy_summary() output and any future plotting."""
    names = []
    for ce in CE_MODES:
        ce_label = ["CE-A", "CE-B", "CE-C"][ce]
        for _lo, _hi, bucket_name in TRAFFIC_BUCKETS:
            names.append(f"{ce_label}/{bucket_name}")
    return names


def action_names() -> list:
    """Human-readable labels for each action index."""
    from environment import ACTIONS
    return [f"T3412={t3412}s,T3324={t3324}s" for t3412, t3324 in ACTIONS]


def train(n_episodes: int = 20000, episode_length: int = 20,
         w_lifetime: float = 0.7, w_delay: float = 1.0,
         config: QLearningConfig = None, seed: int = 11,
         verbose: bool = True, log_every: int = 100, save_outputs: bool = False) -> QLearningAgent:
    """
    Run a full training campaign.

    w_lifetime, w_delay: see environment.py's NbIotEnv docstring — w_delay
    is a SCALING FACTOR applied on top of the per-traffic-bucket
    DELAY_WEIGHT_BY_BUCKET table, not a fixed weight by itself.

    “Writes q_table.npy and training_log.csv to RESULTS_DIR only when save_outputs=True
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = NbIotEnv(w_lifetime=w_lifetime, w_delay=w_delay,
                   episode_length=episode_length, seed=seed)
    agent = QLearningAgent(n_states=env.n_states, n_actions=N_ACTIONS,
                           config=config or QLearningConfig(), seed=seed)

    log_rows = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        tau_fired_count = 0

        for _ in range(episode_length):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            if info.tau_fired:
                tau_fired_count += 1
            state = next_state
            if done:
                break

        agent.end_episode()

        log_rows.append({
            "episode": episode,
            "total_reward": episode_reward,
            "avg_reward_per_step": episode_reward / episode_length,
            "epsilon": agent.epsilon,
            "tau_fired_count": tau_fired_count,
            "min_visit_count": int(agent.visit_counts.min()),
            "table_coverage_pct": 100.0 * np.count_nonzero(agent.visit_counts) / agent.visit_counts.size,
        })

        if verbose and (episode % log_every == 0 or episode == n_episodes):
            recent = log_rows[-log_every:] if len(log_rows) >= log_every else log_rows
            avg_recent = np.mean([r["total_reward"] for r in recent])
            print(f"  episode {episode:5d}/{n_episodes}  "
                  f"avg_reward(last{len(recent)})={avg_recent:+.4f}  "
                  f"epsilon={agent.epsilon:.4f}  "
                  f"min_visits={log_rows[-1]['min_visit_count']:3d}  "
                  f"coverage={log_rows[-1]['table_coverage_pct']:.1f}%")

        # ── Save outputs (only when asked to, so experiments can't overwrite the production table)
    if save_outputs:
        q_table_path = os.path.join(RESULTS_DIR, "q_table.npy")
        agent.save(q_table_path)

        log_path = os.path.join(RESULTS_DIR, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

        if verbose:
            print()
            print(f"[SAVED] Q-table: {q_table_path}")
            print(f"[SAVED] Training log: {log_path}")

    if verbose:
        print()
        print("Learned policy summary:")
        print(agent.policy_summary(state_names=state_names(), action_names=action_names()))

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NB-IoT adaptive Q-learning agent")
    parser.add_argument("--episodes", type=int, default=60000)
    parser.add_argument("--episode-length", type=int, default=20)
    parser.add_argument("--w-lifetime", type=float, default=0.7)
    parser.add_argument("--w-delay", type=float, default=1.0,
                        help="Scaling factor on top of the per-traffic-bucket "
                             "delay weight table in environment.py — see "
                             "NbIotEnv docstring for details.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    args = parser.parse_args()

    cfg = QLearningConfig(alpha=args.alpha, gamma=args.gamma,
                          epsilon_decay=args.epsilon_decay)

    print(f"[TRAIN] Starting training: {args.episodes} episodes, "
          f"w_lifetime={args.w_lifetime}, w_delay={args.w_delay}")
    print(f"[TRAIN] Hyperparameters: alpha={cfg.alpha}, gamma={cfg.gamma}, "
          f"epsilon_decay={cfg.epsilon_decay}")
    print()

    train(n_episodes=args.episodes, episode_length=args.episode_length,
          w_lifetime=args.w_lifetime, w_delay=args.w_delay,
          config=cfg, seed=args.seed, save_outputs=True)
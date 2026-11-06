#!/usr/bin/env python3
"""
Regenerates Figure 8 (RL method comparison) from real, freshly-run training
for all six methods, including Double Q-learning itself (implemented here
matching the project's actual two-table update rule) -- nothing in this
script's output is reused from earlier summary numbers; every score comes
from a real training run executed when you run this script. Saves every
individual seed's result to CSV so the figure is independently checkable,
matching the verifiability pattern used throughout this project (see
monte_carlo_uncertainty.py).

Uses each method's own best-found settings from the earlier tuning
investigation (see System_Changelog.pdf, Change 9) rather than re-running
that full search -- this script focuses on producing a clean, final,
publication-ready comparison and figure.

Run from your qlearning/ folder: python3 rl_comparison_figure.py
"""
import sys
sys.path.insert(0, ".")
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from environment import NbIotEnv, ACTIONS, N_ACTIONS, CE_MODES, TRAFFIC_BUCKETS
from smp_engine import compute_smp
from environment import bucket_to_representative_interval, normalize_lifetime, normalize_delay, DELAY_WEIGHT_BY_BUCKET

N_STATES = len(CE_MODES) * len(TRAFFIC_BUCKETS)
GAMMA = 0.9
EPSILON_START, EPSILON_MIN, EPSILON_DECAY = 1.0, 0.05, 0.995
EPISODE_LENGTH = 20
N_EPISODES = 60000
T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
CORRECTNESS_GAP_THRESHOLD = 0.02
SEEDS = list(range(1, 11))  # 10 seeds per method

# Each method's own best-found settings, established in the earlier
# individual-tuning investigation -- reused here rather than re-searched
METHOD_SETTINGS = {
    "Plain Q-learning": {"alpha": 0.05},
    "SARSA": {"alpha": 0.05},
    "UCB": {"c": 0.5},
    "Monte Carlo": {"alpha": 0.05},
    "Double Q-learning\n(production)": {"alpha": 0.1},  # actual production default
    "Contextual\nBandit": {"alpha": 0.1},  # no tuning needed -- already perfect
}


def epsilon_greedy(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, N_ACTIONS)
    return int(np.argmax(q_table[state]))


def ucb_select(q_table, state, action_counts, state_visits, c):
    untried = np.where(action_counts[state] == 0)[0]
    if len(untried) > 0:
        return int(untried[0])
    bonus = c * np.sqrt(np.log(state_visits[state]) / action_counts[state])
    return int(np.argmax(q_table[state] + bonus))


def make_env(seed):
    return NbIotEnv(w_lifetime=0.7, w_delay=1.0, episode_length=EPISODE_LENGTH, seed=seed)


def train_plain_qlearning(seed, alpha):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(N_EPISODES):
        state = env.reset()
        for step in range(EPISODE_LENGTH):
            action = epsilon_greedy(q_table, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            current_q = q_table[state, action]
            target = reward if done else reward + GAMMA * np.max(q_table[next_state])
            q_table[state, action] = current_q + alpha * (target - current_q)
            state = next_state
            if done: break
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    return q_table


def train_sarsa(seed, alpha):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(N_EPISODES):
        state = env.reset()
        action = epsilon_greedy(q_table, state, epsilon, rng)
        for step in range(EPISODE_LENGTH):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(q_table, next_state, epsilon, rng)
            current_q = q_table[state, action]
            target = reward if done else reward + GAMMA * q_table[next_state, next_action]
            q_table[state, action] = current_q + alpha * (target - current_q)
            state, action = next_state, next_action
            if done: break
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    return q_table


def train_bandit(seed, alpha):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(N_EPISODES):
        state = env.reset()
        for step in range(EPISODE_LENGTH):
            action = epsilon_greedy(q_table, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            current_q = q_table[state, action]
            q_table[state, action] = current_q + alpha * (reward - current_q)
            state = next_state
            if done: break
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    return q_table


def train_ucb(seed, c):
    q_table = np.zeros((N_STATES, N_ACTIONS))
    action_counts = np.zeros((N_STATES, N_ACTIONS))
    state_visits = np.zeros(N_STATES)
    env = make_env(seed)
    alpha = 0.1
    for episode in range(N_EPISODES):
        state = env.reset()
        for step in range(EPISODE_LENGTH):
            state_visits[state] += 1
            action = ucb_select(q_table, state, action_counts, state_visits, c)
            action_counts[state, action] += 1
            next_state, reward, done, _ = env.step(action)
            current_q = q_table[state, action]
            target = reward if done else reward + GAMMA * np.max(q_table[next_state])
            q_table[state, action] = current_q + alpha * (target - current_q)
            state = next_state
            if done: break
    return q_table


def train_monte_carlo(seed, alpha):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(N_EPISODES):
        state = env.reset()
        trajectory = []
        for step in range(EPISODE_LENGTH):
            action = epsilon_greedy(q_table, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            if done: break
        G = 0.0
        for s, a, r in reversed(trajectory):
            G = r + GAMMA * G
            q_table[s, a] = q_table[s, a] + alpha * (G - q_table[s, a])
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    return q_table


def train_double_q(seed, alpha):
    """The project's actual production algorithm: two independent Q-tables,
    decoupling action selection from value estimation on each update."""
    rng = np.random.default_rng(seed)
    q_a = np.zeros((N_STATES, N_ACTIONS))
    q_b = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(N_EPISODES):
        state = env.reset()
        for step in range(EPISODE_LENGTH):
            combined = q_a[state] + q_b[state]
            if rng.random() < epsilon:
                action = rng.integers(0, N_ACTIONS)
            else:
                action = int(np.argmax(combined))
            next_state, reward, done, _ = env.step(action)
            if rng.random() < 0.5:
                best_next = int(np.argmax(q_a[next_state]))
                target = reward if done else reward + GAMMA * q_b[next_state, best_next]
                q_a[state, action] += alpha * (target - q_a[state, action])
            else:
                best_next = int(np.argmax(q_b[next_state]))
                target = reward if done else reward + GAMMA * q_a[next_state, best_next]
                q_b[state, action] += alpha * (target - q_b[state, action])
            state = next_state
            if done: break
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    return q_a + q_b  # combined table, matching how the production agent's q_table property works


def true_optimal_for_all_states():
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


def audit(q_table, optimal):
    n_correct = 0
    for ce in CE_MODES:
        for bucket in range(len(TRAFFIC_BUCKETS)):
            state_idx = ce * len(TRAFFIC_BUCKETS) + bucket
            action_idx = int(np.argmax(q_table[state_idx]))
            learned_T3412 = ACTIONS[action_idx][0]
            true_best_T3412, rewards = optimal[(ce, bucket)]
            true_best_reward = rewards[true_best_T3412]
            learned_reward = rewards.get(learned_T3412, true_best_reward)
            if true_best_reward - learned_reward <= CORRECTNESS_GAP_THRESHOLD:
                n_correct += 1
    return n_correct


TRAIN_FN = {
    "Plain Q-learning": lambda seed, s: train_plain_qlearning(seed, s["alpha"]),
    "SARSA": lambda seed, s: train_sarsa(seed, s["alpha"]),
    "UCB": lambda seed, s: train_ucb(seed, s["c"]),
    "Monte Carlo": lambda seed, s: train_monte_carlo(seed, s["alpha"]),
    "Double Q-learning\n(production)": lambda seed, s: train_double_q(seed, s["alpha"]),
    "Contextual\nBandit": lambda seed, s: train_bandit(seed, s["alpha"]),
}

optimal = true_optimal_for_all_states()
os.makedirs("results", exist_ok=True)

all_results = {}
raw_rows = []

for method_name, train_fn in TRAIN_FN.items():
    print(f"Training {method_name.strip()}...")
    scores = []
    settings = METHOD_SETTINGS[method_name]
    for seed in SEEDS:
        q = train_fn(seed, settings)
        score = audit(q, optimal)
        scores.append(score)
        raw_rows.append({"method": method_name.strip(), "seed": seed, "states_correct": score})
        print(f"  seed={seed}: {score}/{N_STATES}")
    all_results[method_name] = scores

# Save every individual seed's result -- real, checkable evidence
raw_path = "results/rl_comparison_raw.csv"
with open(raw_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "seed", "states_correct"])
    writer.writeheader()
    writer.writerows(raw_rows)
print(f"\n[SAVED] Every individual seed's result: {raw_path}")

# ── Figure: box + individual points, deliberately different style from ──
# ── the heatmap (Figure 7) and the bar chart used elsewhere ─────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
method_order = ["Plain Q-learning", "SARSA", "UCB", "Monte Carlo",
                 "Double Q-learning\n(production)", "Contextual\nBandit"]
data = [all_results[m] for m in method_order]

bp = ax.boxplot(data, patch_artist=True, widths=0.5, showmeans=True,
                 meanprops={"marker": "D", "markerfacecolor": "white",
                            "markeredgecolor": "black", "markersize": 6},
                 medianprops={"color": "black", "linewidth": 1.5},
                 boxprops={"facecolor": "#e8ecf3", "edgecolor": "#2c3e50"},
                 whiskerprops={"color": "#2c3e50"}, capprops={"color": "#2c3e50"})

# Overlay individual seed results as jittered points -- shows the real
# spread directly, not just summary statistics
rng = np.random.default_rng(0)
for i, scores in enumerate(data, start=1):
    jitter = rng.uniform(-0.12, 0.12, size=len(scores))
    ax.scatter(np.full(len(scores), i) + jitter, scores,
               color="#c0392b", alpha=0.6, s=22, zorder=3, edgecolors="none")

ax.set_xticks(range(1, len(method_order) + 1))
ax.set_xticklabels(method_order, fontsize=8.5)
ax.set_ylabel(f"States matching true-optimal action (out of {N_STATES})")
ax.set_title("RL Method Comparison: Full Seed-by-Seed Distribution\n"
              "(box = median/quartiles, diamond = mean, red dots = individual seeds)")
ax.set_ylim(0, N_STATES + 2)
ax.axhline(N_STATES, color="#8d0bf0", linestyle="--", alpha=0.4, linewidth=1)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
fig_path = "results/figure8_rl_comparison.png"
plt.savefig(fig_path, dpi=150)
print(f"[SAVED] Figure: {fig_path}")
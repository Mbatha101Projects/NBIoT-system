#!/usr/bin/env python3
"""
Point 3 (reviewer feedback), individually-tuned version: each baseline
gets its OWN hyperparameter search (not just Double Q-learning's
settings reused everywhere) before final validation -- testing what
each method's BEST achievable performance is, not just a fair-but-
possibly-suboptimal comparison.

UCB's exploration constant search is directly informed by literature
finding that UCB1 tends to need smaller exploration constants when the
reward function is normalized (as this project's is) -- see Wiering &
van Otterlo, "Comparing Exploration Strategies for Q-learning in Random
Stochastic Mazes".

Contextual bandit is excluded from re-tuning -- it already achieved a
perfect 12/12 across all 10 seeds with the original settings, so there
is no further room to improve it.

Run from your qlearning/ folder: python3 rl_baselines_tuned.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from environment import NbIotEnv, ACTIONS, N_ACTIONS, CE_MODES, TRAFFIC_BUCKETS
from smp_engine import compute_smp
from environment import bucket_to_representative_interval, normalize_lifetime, normalize_delay, DELAY_WEIGHT_BY_BUCKET

N_STATES = len(CE_MODES) * len(TRAFFIC_BUCKETS)
GAMMA = 0.9
EPSILON_START, EPSILON_MIN, EPSILON_DECAY = 1.0, 0.05, 0.995
EPISODE_LENGTH = 20
T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
CORRECTNESS_GAP_THRESHOLD = 0.02

PROXY_EPISODES = 20000   # cheap search phase
PROXY_SEEDS = [1, 2, 3]
FULL_EPISODES = 60000    # final validation phase, matching everything else tested
FULL_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]

ALPHA_CANDIDATES = [0.05, 0.1, 0.2]
UCB_C_CANDIDATES = [0.5, 1.0, 2.0, 4.0]  # includes smaller values, informed by the
                                          # normalized-reward finding in the literature


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
            gap = true_best_reward - learned_reward
            if gap <= CORRECTNESS_GAP_THRESHOLD:
                n_correct += 1
    return n_correct


# ── Training functions, parameterised by alpha (and c for UCB) ──────────
def train_plain_qlearning(seed, n_episodes, alpha):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(1, n_episodes + 1):
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


def train_sarsa(seed, n_episodes, alpha):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(1, n_episodes + 1):
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


def train_ucb(seed, n_episodes, c):
    q_table = np.zeros((N_STATES, N_ACTIONS))
    action_counts = np.zeros((N_STATES, N_ACTIONS))
    state_visits = np.zeros(N_STATES)
    env = make_env(seed)
    alpha = 0.1  # kept fixed here -- only c is being tuned for UCB
    for episode in range(1, n_episodes + 1):
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


def train_monte_carlo(seed, n_episodes, alpha):
    rng = np.random.default_rng(seed)
    q_table = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON_START
    env = make_env(seed)
    for episode in range(1, n_episodes + 1):
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


optimal = true_optimal_for_all_states()


def search_and_validate(name, train_fn, candidates, param_name):
    print(f"\n{'='*70}\n{name} -- proxy search over {param_name}\n{'='*70}")
    candidate_scores = {}
    for cand in candidates:
        scores = []
        for seed in PROXY_SEEDS:
            q = train_fn(seed, PROXY_EPISODES, cand)
            scores.append(audit(q, optimal))
        mean_score = np.mean(scores)
        candidate_scores[cand] = mean_score
        print(f"  {param_name}={cand}: proxy mean={mean_score:.2f}/12 (seeds {PROXY_SEEDS}, {PROXY_EPISODES} episodes)")

    best_cand = max(candidate_scores, key=candidate_scores.get)
    print(f"  -> best {param_name} found: {best_cand}")

    print(f"\n  Validating {param_name}={best_cand} at full scale ({FULL_EPISODES} episodes, {len(FULL_SEEDS)} seeds)...")
    full_scores = []
    for seed in FULL_SEEDS:
        q = train_fn(seed, FULL_EPISODES, best_cand)
        score = audit(q, optimal)
        full_scores.append(score)
        print(f"    seed={seed}: {score}/12")
    print(f"  -> FINAL (tuned): mean={np.mean(full_scores):.2f}/12  range={min(full_scores)}-{max(full_scores)}")
    return best_cand, full_scores


results = {}
results["Plain Q-learning"] = search_and_validate("Plain Q-learning", train_plain_qlearning, ALPHA_CANDIDATES, "alpha")
results["SARSA"] = search_and_validate("SARSA", train_sarsa, ALPHA_CANDIDATES, "alpha")
results["UCB"] = search_and_validate("UCB", train_ucb, UCB_C_CANDIDATES, "c")
results["Monte Carlo"] = search_and_validate("Monte Carlo", train_monte_carlo, ALPHA_CANDIDATES, "alpha")

print(f"\n{'='*70}")
print("FINAL COMPARISON -- each method individually tuned")
print(f"{'='*70}")
print(f"{'Method':<22}{'Best param':<14}{'Mean':<10}{'Range'}")
print(f"{'Double Q-learning':<22}{'(established)':<14}{'8.20':<10}4-11")
print(f"{'Contextual bandit':<22}{'(no tuning needed)':<14}{'12.00':<10}12-12")
for name, (best_param, scores) in results.items():
    print(f"{name:<22}{str(best_param):<14}{np.mean(scores):<10.2f}{min(scores)}-{max(scores)}")

import sys
sys.path.insert(0, ".")
import csv
import os
import numpy as np
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
SEEDS = list(range(1, 21))  # 20 seeds, matching the project's established sample size


def epsilon_greedy(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, N_ACTIONS)
    return int(np.argmax(q_table[state]))


def make_env(seed):
    return NbIotEnv(w_lifetime=0.7, w_delay=1.0, episode_length=EPISODE_LENGTH, seed=seed)


def train_plain_qlearning(seed, alpha=0.1):
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


def train_double_q(seed, alpha=0.1):
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
    return q_a + q_b


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


def audit_per_state(q_table, optimal):
    """Returns a list of 30 booleans -- True if that specific state's
    learned action matches the true optimum, in state-index order."""
    results = []
    for ce in CE_MODES:
        for bucket in range(len(TRAFFIC_BUCKETS)):
            state_idx = ce * len(TRAFFIC_BUCKETS) + bucket
            action_idx = int(np.argmax(q_table[state_idx]))
            learned_T3412 = ACTIONS[action_idx][0]
            true_best_T3412, rewards = optimal[(ce, bucket)]
            true_best_reward = rewards[true_best_T3412]
            learned_reward = rewards.get(learned_T3412, true_best_reward)
            ok = (true_best_reward - learned_reward) <= CORRECTNESS_GAP_THRESHOLD
            results.append(ok)
    return results


optimal = true_optimal_for_all_states()
bucket_names = [b[2] for b in TRAFFIC_BUCKETS]
ce_names = ["CE-A", "CE-B", "CE-C"]
state_labels = [f"{ce_names[ce]}/{bucket_names[b]}" for ce in CE_MODES for b in range(len(TRAFFIC_BUCKETS))]

# per_state_wrong_count[method][state_idx] = how many of the 20 seeds got this state wrong
per_state_wrong_count = {"Plain Q-learning": [0]*N_STATES, "Double Q-learning": [0]*N_STATES}
raw_rows = []

for method_name, train_fn in [("Plain Q-learning", train_plain_qlearning), ("Double Q-learning", train_double_q)]:
    print(f"\nTraining {method_name} across {len(SEEDS)} seeds...")
    for seed in SEEDS:
        q = train_fn(seed)
        per_state_correct = audit_per_state(q, optimal)
        n_correct = sum(per_state_correct)
        print(f"  seed={seed}: {n_correct}/{N_STATES} correct")
        for state_idx, correct in enumerate(per_state_correct):
            raw_rows.append({
                "method": method_name, "seed": seed, "state_index": state_idx,
                "state_label": state_labels[state_idx], "correct": correct,
            })
            if not correct:
                per_state_wrong_count[method_name][state_idx] += 1

os.makedirs("results", exist_ok=True)
raw_path = "results/maximization_bias_per_state_raw.csv"
with open(raw_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "seed", "state_index", "state_label", "correct"])
    writer.writeheader()
    writer.writerows(raw_rows)
print(f"\n[SAVED] Full per-state, per-seed results: {raw_path}")

print("\n" + "="*80)
print("VERIFICATION RESULT")
print("="*80)
print(f"{'State':<20}{'Plain Q wrong (out of 20)':<30}{'Double Q wrong (out of 20)'}")
for i, label in enumerate(state_labels):
    pq_wrong = per_state_wrong_count["Plain Q-learning"][i]
    dq_wrong = per_state_wrong_count["Double Q-learning"][i]
    marker = "  <-- most consistently wrong under Plain Q" if pq_wrong >= 15 else ""
    print(f"{label:<20}{pq_wrong:<30}{dq_wrong}{marker}")

max_pq_wrong = max(per_state_wrong_count["Plain Q-learning"])
worst_state_idx = per_state_wrong_count["Plain Q-learning"].index(max_pq_wrong)
worst_state_label = state_labels[worst_state_idx]
dq_wrong_for_that_state = per_state_wrong_count["Double Q-learning"][worst_state_idx]

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"Most consistently wrong state under Plain Q-learning: {worst_state_label}")
print(f"  Wrong in {max_pq_wrong}/{len(SEEDS)} seeds under Plain Q-learning")
print(f"  Wrong in {dq_wrong_for_that_state}/{len(SEEDS)} seeds under Double Q-learning")
if max_pq_wrong >= 15 and dq_wrong_for_that_state <= 5:
    print("  -> CLAIM SUPPORTED: this state shows a genuine, consistent bias under")
    print("     Plain Q-learning that Double Q-learning substantially corrects.")
elif max_pq_wrong >= 15:
    print("  -> PARTIALLY SUPPORTED: a consistently-wrong state exists under Plain")
    print("     Q-learning, but Double Q-learning does not clearly fix it either.")
else:
    print("  -> CLAIM NOT SUPPORTED on the current 30-state system: no single state")
    print("     is wrong consistently enough (>=15/20 seeds) to match the original")
    print("     claim's description. Errors appear more scattered across states.")
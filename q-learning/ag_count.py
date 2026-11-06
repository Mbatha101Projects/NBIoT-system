from train import train
from environment import ACTIONS, CE_MODES, TRAFFIC_BUCKETS, bucket_to_representative_interval, normalize_lifetime, normalize_delay, DELAY_WEIGHT_BY_BUCKET
from smp_engine import compute_smp
import numpy as np

T3412_vals = sorted(set(t3412 for t3412, _t3324 in ACTIONS))
ce_names = {0: "CE-A", 1: "CE-B", 2: "CE-C"}
bucket_names = [b[2] for b in TRAFFIC_BUCKETS]

for i, seed in enumerate([42, 1, 2, 3, 4]):
    agent = train(n_episodes=60000, seed=seed, verbose=False)
    print(f"\n=== Run {i+1} (seed={seed}) ===")
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
            true_best_T3412 = max(true_rewards, key=true_rewards.get)

            state_idx = ce * len(TRAFFIC_BUCKETS) + bucket
            learned_T3412 = ACTIONS[int(np.argmax(agent.q_table[state_idx]))][0]
            gap = true_rewards[true_best_T3412] - true_rewards.get(learned_T3412, true_rewards[true_best_T3412])
            status = "OK" if gap <= 0.02 else "WRONG"
            print(f"  {ce_names[ce]}/{bucket_names[bucket]:14s} learned={learned_T3412:6d} true_best={true_best_T3412:6d} gap={gap:.4f} [{status}]")
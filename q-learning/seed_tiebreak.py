#!/usr/bin/env python3
"""
tiebreak_from_saved.py -- picks the production seed WITHOUT retraining.

Reads the tables that multi_seed_evaluation.py saved in results/seeds/ and,
for every seed that has the top audit score (T3412-only within 0.02, the same
audit as seed_search.py), reports the SUM of reward gaps over all 30 states
(the same tie-break idea as seed_tiebreak.py: 0 = every state exactly right).
Lowest total gap wins. It does not use the evaluation scenarios, so the pick
is not tuned to the numbers you later report.

RUN (q-learning folder, after multi_seed_evaluation.py):
    python tiebreak_from_saved.py
Does not touch results/q_table.npy.
"""
import sys, glob, re
sys.path.insert(0, ".")
import numpy as np
from smp_engine import compute_smp
from environment import (ACTIONS, CE_MODES, TRAFFIC_BUCKETS, bucket_to_representative_interval,
                         normalize_lifetime, normalize_delay)

TOL = 0.02
nb = len(TRAFFIC_BUCKETS)
T3412_vals = sorted(set(t for t, _ in ACTIONS))

ref = {}
for ce in CE_MODES:
    for b in range(nb):
        iv = bucket_to_representative_interval(b)
        ref[(ce, b)] = {t: 0.7 * normalize_lifetime(compute_smp(t, 20, iv, ce).lifetime_yr)
                           - 0.45 * normalize_delay(compute_smp(t, 20, iv, ce).delay_seconds)
                        for t in T3412_vals}

rows = []
for path in sorted(glob.glob("results/seeds/q_table_seed*.npy"),
                   key=lambda p: int(re.search(r"seed(\d+)", p).group(1))):
    seed = int(re.search(r"seed(\d+)", path).group(1))
    q = np.load(path)
    q = (q[0] + q[1]) / 2.0 if q.ndim == 3 else q
    ok, total = 0, 0.0
    for ce in CE_MODES:
        for b in range(nb):
            t = ACTIONS[int(np.argmax(q[ce * nb + b]))][0]
            r = ref[(ce, b)]
            gap = max(r.values()) - r[t]
            total += gap
            ok += gap <= TOL
    rows.append((seed, ok, total))

top = max(r[1] for r in rows)
print(f"{'seed':>5} {'audit':>6} {'total gap':>10}")
for s, ok, tg in sorted(rows, key=lambda r: (-r[1], r[2])):
    print(f"{s:>5} {ok:>4}/30 {tg:>10.5f}" + ("   <-- top audit" if ok == top else ""))
best = min((r for r in rows if r[1] == top), key=lambda r: r[2])
print(f"\nProduction seed = {best[0]}  (audit {best[1]}/30, total gap {best[2]:.5f})")
print(f"Next:  python train.py --seed {best[0]}")
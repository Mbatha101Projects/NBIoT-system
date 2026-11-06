"""
audit_production_policy.py - checks a trained Q-table against the true optimum.

RUN (from your q-learning folder, the one containing smp_engine.py and environment.py):
    python audit_production_policy.py results/q_table.npy

For each of the 30 states it tries all 42 (T3412, T3324) actions with the same
reward the agent trained on, finds the best one, and compares it to the
action the Q-table picks.
  - "exact"  : agent picked the single best action
  - "within" : agent's reward is within TOL of the best (T3324 barely matters, so
               this is the fair test; 0.02 is the tolerance used earlier)
"""
import sys
import numpy as np
from smp_engine import compute_smp
from environment import (ACTIONS, CE_MODES, TRAFFIC_BUCKETS,
                         bucket_to_representative_interval,
                         normalize_lifetime, normalize_delay)

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/q_table.npy"
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02

q = np.load(PATH)
print("Loaded", PATH, "shape", q.shape)
qc = (q[0] + q[1]) / 2.0 if q.ndim == 3 else q   # Double-Q file = 2 tables, average them
nb = len(TRAFFIC_BUCKETS)

exact = within = 0
bad = []
for ce in CE_MODES:
    for b in range(nb):
        iv = bucket_to_representative_interval(b)
        r = np.array([0.7 * normalize_lifetime(compute_smp(t1, t2, iv, ce).lifetime_yr)
                      - 0.45 * normalize_delay(compute_smp(t1, t2, iv, ce).delay_seconds)
                      for (t1, t2) in ACTIONS])
        a = int(np.argmax(qc[ce * nb + b])); best = int(np.argmax(r)); gap = r[best] - r[a]
        exact += (a == best); within += (gap <= TOL)
        if gap > TOL:
            bad.append((f"CE-{'ABC'[ce]}", TRAFFIC_BUCKETS[b][2], ACTIONS[a], ACTIONS[best], round(gap, 4)))

print(f"\nExact best action : {exact}/30")
print(f"Within tolerance {TOL}: {within}/30   <-- this is the '30/30' number")
for x in bad:
    print("  WRONG:", x[0], x[1], "agent", x[2], "best", x[3], "gap", x[4])
print("\nPolicy grid (rows CE-A/B/C, columns traffic buckets 0-9):")
for ce in CE_MODES:
    print(f"CE-{'ABC'[ce]}", [ACTIONS[int(np.argmax(qc[ce * nb + b]))] for b in range(nb)])


# ---------------------------------------------------------------------------
# SAME TEST AS seed_search.py (the "30/30" in the paper): T3412 only,
# true optimum computed at T3324 = 20 s, correct if gap <= 0.02.
# ---------------------------------------------------------------------------
T3412_vals = sorted(set(t for t, _ in ACTIONS))
ok = 0; wrong = []
for ce in CE_MODES:
    for b in range(nb):
        iv = bucket_to_representative_interval(b)
        rw = {}
        for t in T3412_vals:
            x = compute_smp(t, 20, iv, ce)
            rw[t] = 0.7 * normalize_lifetime(x.lifetime_yr) - 0.45 * normalize_delay(x.delay_seconds)
        best = max(rw, key=rw.get)
        learned = ACTIONS[int(np.argmax(qc[ce * nb + b]))][0]
        gap = rw[best] - rw[learned]
        if gap <= 0.02: ok += 1
        else: wrong.append((f"CE-{'ABC'[ce]}", TRAFFIC_BUCKETS[b][2], learned, best, round(gap, 4)))
print(f"\n=== seed_search.py method (T3412 only, gap <= 0.02): {ok}/30 ===")
for w in wrong: print("  WRONG:", w)
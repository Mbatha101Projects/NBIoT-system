from smp_engine import compute_smp
from environment import ACTIONS

def sweep(intervals):
    rows = []
    for ce in (0, 1, 2):
        for (t3412, t3324) in ACTIONS:
            for iv in intervals:
                r = compute_smp(t3412, t3324, iv, ce)
                rows.append((r.delay_seconds, ce, t3412, t3324, iv))
    return rows

for name, ivs in [("7200 s only (the Fig 4.1 / 4.2 grid)", [7200]),
                  ("all intervals", [300, 900, 1800, 3600, 7200, 14400, 43200, 86400])]:
    rows = sweep(ivs)
    lo, hi = min(rows), max(rows)
    print(f"\n{name}  ({len(rows)} points)")
    print(f"  min delay: {lo[0]:.1f} s  at CE={lo[1]} T3412={lo[2]} T3324={lo[3]} interval={lo[4]}")
    print(f"  max delay: {hi[0]:.1f} s  at CE={hi[1]} T3412={hi[2]} T3324={hi[3]} interval={hi[4]}")
    print(f"  ratio max/min: {hi[0]/lo[0]:.0f}x")
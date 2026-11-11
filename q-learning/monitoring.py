"""
monitoring_mechanisms.py  --  CAN THE AGENT SEE ITS CONDITIONS CHEAPLY ENOUGH?

Run next to overhead_analysis.py (same folder as evaluate.py, smp_engine.py ...):

    python monitoring_mechanisms.py
    python monitoring_mechanisms.py --qtable results/q_table.npy

It changes NOTHING in training, constants or the reward.  It re-uses the traces
and populations from overhead_analysis.py and answers the question that the old
"x microwatts always-on" sweep could not: WHERE does the agent get its state
(coverage class + traffic interval) from, and what does each way cost?

THREE WAYS TO OBSERVE THE STATE
-------------------------------
 M1  FREE-RIDE.  Read the state at wake-ups the device ALREADY does.
       * traffic interval  = the host's own report timer / counter (it must
         exist anyway to send the reports)
       * coverage class    = what the modem measures during SYNC (cell search,
         which every wake-up already includes) - no extra radio time.
     Extra cost per step = e_obs, the host energy to read and store two numbers
     (micro-joules).  The state is as fresh as the step, which is exactly what
     the training environment assumes (one observation per step), so there is
     NO staleness penalty to add.
 M2  EXTRA RADIO CHECKS.  Wake the radio every P seconds only to re-measure
     coverage (cell search without RACH).  Cost = e_meas / P, where
     e_meas = P_SYNC x t_meas.  Needed only if the device wants to notice a
     coverage change BETWEEN reports.
 M3  ALWAYS-ON HOST DRAW.  A constant microwatt draw (RTC + MCU sleep + RAM
     retention) that exists whether or not the agent runs.  Only the part that
     the NON-adaptive device would not pay counts; this is the old sweep.

WHAT THE SCRIPT REPORTS (per population)
----------------------------------------
 1. M1: break-even e_obs (micro-joules per step) before the gap to best static
    reaches zero, and the verdict at typical values.
 2. M2: for several t_meas, the SHORTEST check period P that still leaves the
    adaptive device ahead of best static.
 3. M3: break-even microwatts (same as overhead_analysis section D, repeated so
    all three sit in one table).
 4. A verdict table: PASS if the adaptive device is still ahead of best static.

Every cost parameter is an ASSUMPTION swept over a range, not a measurement.
State that in the dissertation; what is reported is the break-even.
"""
import os
import sys
import csv
import argparse
import numpy as np

from smp_engine import P_SYNC_MW, CE_SYNC_SCALE
from environment import ACTIONS, TRAFFIC_BUCKETS, CE_MODES
from agent import QLearningAgent
from evaluate import RESULTS_DIR
import overhead_analysis as oa

E_OBS_GRID_UJ = [0, 1, 10, 100, 1000, 10000]           # host energy per step, micro-joules
T_MEAS_GRID_S = [0.1, 0.5, 1.0, 2.2]                   # radio-on time of one coverage check
P_CHECK_GRID_S = [60, 600, 3600, 6 * 3600, 86400]      # how often the extra check runs
M3_GRID_UW = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
TYPICAL_E_OBS_UJ = 10.0       # ASSUMPTION: a few thousand MCU cycles at tens of mW for ~100 us
TYPICAL_M3_UW = 2.0           # ASSUMPTION: RTC + RAM retention; replace with your MCU datasheet value


def rewards_with_extra(pop, extra_mw):
    """Average adaptive reward per scenario when a per-step extra power (mW)
    is added to the adaptive device.  Same formula as Pop.scen_rewards."""
    life = (oa.BATTERY_MWH / (pop.p + extra_mw)) / 8760.0
    rew = oa.W_LIFETIME * oa._vec_norm_lt(life) - pop.delay_term
    return np.bincount(pop.scen, weights=rew, minlength=pop.n) / pop.counts


def gap_m1(pop, e_obs_uj):
    extra = (e_obs_uj / 1000.0) / pop.iv                 # mJ per step / s per step = mW
    return float((rewards_with_extra(pop, extra) - pop.best).mean())


def gap_m2(pop, t_meas, period):
    ce = np.array([s["ce"] for s in pop.flat])
    e_meas_mj = P_SYNC_MW * t_meas * np.array([CE_SYNC_SCALE[c] for c in ce])
    extra = e_meas_mj / period                           # mW, constant while that coverage holds
    return float((rewards_with_extra(pop, extra) - pop.best).mean())


def gap_m3(pop, uw):
    return float((rewards_with_extra(pop, uw / 1000.0) - pop.best).mean())


def run(agent):
    named, keys = oa.load_named(agent)
    rand, worst, n_bad, n_all = oa.load_random(
        agent, os.path.join(RESULTS_DIR, "random_scenarios_steps.csv"),
        os.path.join(RESULTS_DIR, "random_scenarios_results.csv"))
    pops = [named] + rand
    short = lambda n: n.split(" traffic")[0] if " traffic" in n else n
    rows = [["population", "mechanism", "parameter", "value", "mean_gap_vs_best"]]

    print("=" * 100)
    print("MONITORING MECHANISMS - gap = adaptive - best static (reward points, > 0 = adaptive still ahead)")
    print("=" * 100)
    print("\nBaseline (no monitoring cost):")
    for p in pops:
        print(f"  {short(p.name):<28} gap {gap_m1(p, 0):+.4f}")

    print("\nM1  FREE-RIDE: host energy per step (micro-joules)")
    print(f"  {'Population':<28}" + "".join(f"{v:>9g}" for v in E_OBS_GRID_UJ) + f"{'break-even uJ':>16}")
    be1 = {}
    for p in pops:
        cells = []
        for v in E_OBS_GRID_UJ:
            g = gap_m1(p, v); cells.append(f"{g:+.4f}")
            rows.append([p.name, "M1", "e_obs_uJ", v, g])
        if gap_m1(p, 0) <= 0:
            be = "already <= 0"
        else:
            x = oa.bisect(lambda u, p=p: gap_m1(p, u), 0.0, 1e8, log=False)
            be = "> 1e8 (safe)" if x is None else f"{x:,.0f}"
        be1[p.name] = be
        print(f"  {short(p.name):<28}" + "".join(f"{c:>9}" for c in cells) + f"{be:>16}")

    print("\nM2  EXTRA RADIO CHECKS: gap for check period P (rows) and radio time per check (columns)")
    for p in pops:
        print(f"\n  {short(p.name)}")
        print(f"    {'period P':<12}" + "".join(f"t_meas={t:g}s".rjust(14) for t in T_MEAS_GRID_S))
        for P in P_CHECK_GRID_S:
            cells = []
            for t in T_MEAS_GRID_S:
                g = gap_m2(p, t, P); cells.append(f"{g:+.4f}")
                rows.append([p.name, "M2", f"t_meas={t}s;P_s", P, g])
            print(f"    {oa.fmt_s(P):<12}" + "".join(c.rjust(14) for c in cells))
        line = []
        for t in T_MEAS_GRID_S:
            if gap_m2(p, t, 1e9) <= 0:
                line.append("n/a (already <= 0)")
            else:
                x = oa.bisect(lambda P_, p=p, t=t: gap_m2(p, t, P_),
                              1.0, 1e9, log=True)
                line.append("any period" if x is None else f"P >= {oa.fmt_s(x)}")
        print("    shortest allowed period: " + "; ".join(f"t_meas={t:g}s -> {l}" for t, l in zip(T_MEAS_GRID_S, line)))

    print("\nM3  ALWAYS-ON HOST DRAW (microwatts)")
    print(f"  {'Population':<28}" + "".join(f"{v:>9g}" for v in M3_GRID_UW) + f"{'break-even uW':>16}")
    be3 = {}
    for p in pops:
        cells = []
        for v in M3_GRID_UW:
            g = gap_m3(p, v); cells.append(f"{g:+.4f}")
            rows.append([p.name, "M3", "uW", v, g])
        if gap_m3(p, 0) <= 0:
            be = "already <= 0"
        else:
            x = oa.bisect(lambda u, p=p: gap_m3(p, u), 0.0, 5000.0)
            be = "> 5000" if x is None else f"{x:.2f}"
        be3[p.name] = be
        print(f"  {short(p.name):<28}" + "".join(f"{c:>9}" for c in cells) + f"{be:>16}")

    print("\nVERDICT at the assumed typical values "
          f"(M1 e_obs = {TYPICAL_E_OBS_UJ:g} uJ/step, M3 = {TYPICAL_M3_UW:g} uW):")
    print(f"  {'Population':<28}{'baseline':>10}{'M1':>10}{'M3':>10}{'M1+M3':>10}   verdict")
    for p in pops:
        extra = (TYPICAL_E_OBS_UJ / 1000.0) / p.iv + TYPICAL_M3_UW / 1000.0
        g0, g1, g3 = gap_m1(p, 0), gap_m1(p, TYPICAL_E_OBS_UJ), gap_m3(p, TYPICAL_M3_UW)
        gb = float((rewards_with_extra(p, extra) - p.best).mean())
        print(f"  {short(p.name):<28}{g0:>+10.4f}{g1:>+10.4f}{g3:>+10.4f}{gb:>+10.4f}   "
              f"{'PASS (still ahead)' if gb > 0 else 'FAIL (advantage gone)'}")
    with open(os.path.join(RESULTS_DIR, "monitoring_mechanisms.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\n[SAVED] results/monitoring_mechanisms.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtable", default=os.path.join(RESULTS_DIR, "q_table.npy"))
    a = ap.parse_args()
    if not os.path.exists(a.qtable):
        sys.exit(f"Q-table not found at {a.qtable}")
    ag = QLearningAgent(n_states=len(CE_MODES) * len(TRAFFIC_BUCKETS), n_actions=len(ACTIONS))
    ag.load(a.qtable)
    run(ag)
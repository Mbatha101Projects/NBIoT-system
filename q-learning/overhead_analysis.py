"""
overhead_analysis.py  --  WHAT DOES ADAPTING COST?

Put this file next to evaluate.py / smp_engine.py / environment.py and run:

    python overhead_analysis.py
    python overhead_analysis.py --qtable results/q_table.npy --no-plot

It needs three things that already exist in your project:
    results/q_table.npy                    (the trained policy)
    results/random_scenarios_steps.csv     (the 3,000 random scenarios, step by step)
    results/random_scenarios_results.csv   (their best-static pair and rewards)
and it changes NOTHING: no training, no constants, no reward formula.

WHY THIS EXISTS
---------------
Everything reported so far counts the energy of RUNNING the chosen timers.
It does not count the energy of ADAPTING, which comes in two kinds:

  (1) ONE-OFF cost per timer change.  T3412/T3324 are negotiated with the
      network in the attach / TAU procedure.  A device that wakes only to
      send data does not do a TAU, so changing timers needs an EXTRA TAU
      (E_TAU_BASE_MJ x CE_SYNC_SCALE, the same constant smp_engine already
      uses).  Best-static never changes timers, so it never pays this.

  (2) ALWAYS-ON cost while asleep (the SATNAC question).  Any host-side
      monitoring that draws power during sleep adds to a PSM power of only
      ~10.8 uW, so it is NOT safe to assume it is negligible.  Instead of
      guessing a number we ask: how many uW can the agent afford before
      its advantage over best-static disappears?  (the break-even)

TWO MODELLING CHOICES THAT ARE NOT IN THE LITERATURE (state them as such)
-------------------------------------------------------------------------
  * The reconfiguration TAU costs the same as the periodic TAU (210 mJ x CE
    scale).  This is a proxy, not a measurement.
  * Your steps have no wall-clock length, so a one-off cost cannot be
    spread over a step.  We therefore treat each step as lasting H seconds
    (the "hold time") and sweep H.  The break-even H* is the answer that
    does not depend on guessing.

Free-ride assumption (switchable):  if the traffic interval is longer than
the OLD T3412, a timer-expiry TAU already happens every cycle, so the new
values can ride on it and the switch is free.  Both cases are reported:
    'never'     every switch pays a full extra TAU      (pessimistic)
    'when_due'  switches are free when a TAU is due     (optimistic)
"""

import os
import sys
import csv
import argparse
import numpy as np

from smp_engine import compute_smp, BATTERY_MWH, E_TAU_BASE_MJ, CE_SYNC_SCALE, P_PSM_MW
from environment import (ACTIONS, TRAFFIC_BUCKETS, traffic_to_bucket,
                         normalize_lifetime, normalize_delay,
                         DELAY_WEIGHT_BY_BUCKET, CE_MODES)
from agent import QLearningAgent
from evaluate import (W_LIFETIME, W_DELAY, NAIVE_STATIC_ACTION,
                      evaluate_action_on_condition, find_best_static_action,
                      RESULTS_DIR)
from scenarios import SCENARIOS

H_GRID_S = [3600, 6 * 3600, 86400, 7 * 86400, 30 * 86400]          # 1 h .. 30 d
H_LABELS = ["1 hour", "6 hours", "1 day", "7 days", "30 days"]
MON_GRID_UW = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]          # always-on, microwatts
MODES = ("never", "when_due")
DWELL_GRID = [1, 2, 3, 5, 8]
K_GRID = [1, 5, 20, 100, 1000]                                     # reporting intervals per step

_vec_norm_lt = np.vectorize(normalize_lifetime)


def fmt_s(s):
    if s is None or not np.isfinite(s):
        return "never"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} d"


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRACE: run the policy over a list of (ce, interval) conditions
# ─────────────────────────────────────────────────────────────────────────────

def trace(agent, conditions, min_dwell=1):
    """Returns one dict per step.  A 'switch' is a step whose chosen
    (T3412,T3324) differs from the previous step's.  The first step is the
    starting configuration and costs nothing.

    min_dwell = 1 is the policy exactly as trained (it may change timers on
    every step).  min_dwell = m > 1 is a simple switching rule: once timers
    change, keep them for at least m steps before allowing another change."""
    out, prev, held = [], None, 0
    for t, (ce, iv) in enumerate(conditions):
        b = traffic_to_bucket(iv)
        a = ACTIONS[agent.choose_action(ce * len(TRAFFIC_BUCKETS) + b, greedy=True)]
        if prev is not None and a != prev and held < min_dwell:
            a = prev                                    # rule: not allowed to switch yet
        held = held + 1 if a == prev else 1
        r = compute_smp(a[0], a[1], iv, ce)
        st = dict(step=t, ce=ce, iv=iv, bucket=b, action=a, p_mw=r.avg_power_mw,
                  delay_s=r.delay_seconds, switch=False, old=prev, dP_mw=0.0,
                  e_sw_mj=0.0, free=False)
        if prev is not None and a != prev:
            r_old = compute_smp(prev[0], prev[1], iv, ce)
            st.update(switch=True,
                      dP_mw=r_old.avg_power_mw - r.avg_power_mw,   # >0: the switch SAVES power
                      e_sw_mj=E_TAU_BASE_MJ * CE_SYNC_SCALE[ce],
                      free=iv > prev[0])                           # a TAU is already due
        out.append(st)
        prev = a
    return out


def static_avg_reward(conditions, action):
    return float(np.mean([evaluate_action_on_condition(action[0], action[1], ce, iv).reward
                          for ce, iv in conditions]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. POPULATION: many scenarios flattened into arrays so sweeps are fast
# ─────────────────────────────────────────────────────────────────────────────

class Pop:
    def __init__(self, name, traces, best_static, naive_static):
        self.name = name
        self.n = len(traces)
        self.best = np.array(best_static, float)
        self.naive = np.array(naive_static, float)
        self.scen = np.concatenate([[i] * len(t) for i, t in enumerate(traces)]).astype(int)
        flat = [s for t in traces for s in t]
        self.p = np.array([s["p_mw"] for s in flat])
        self.delay_term = np.array([W_DELAY * DELAY_WEIGHT_BY_BUCKET[s["bucket"]] *
                                    normalize_delay(s["delay_s"]) for s in flat])
        self.e_never = np.array([s["e_sw_mj"] for s in flat])
        self.e_due = np.array([0.0 if s["free"] else s["e_sw_mj"] for s in flat])
        self.iv = np.array([s["iv"] for s in flat])
        self.flat = flat
        self.counts = np.bincount(self.scen, minlength=self.n)

    def scen_rewards(self, H=None, mon_uw=0.0, mode="never", k=None):
        """Average adaptive reward per scenario with overhead added.
        H = hold time of one step in seconds (None = ignore switch cost).
        k = alternative to H: a step lasts k reporting intervals, so its
            length is k x that step's traffic interval.
        mon_uw = always-on monitoring power in microwatts (adaptive only)."""
        extra = np.full_like(self.p, mon_uw / 1000.0)
        if H or k:
            hold = H if H else k * self.iv
            extra = extra + (self.e_never if mode == "never" else self.e_due) / hold   # mJ / s = mW
        life = (BATTERY_MWH / (self.p + extra)) / 8760.0
        rew = W_LIFETIME * _vec_norm_lt(life) - self.delay_term
        return np.bincount(self.scen, weights=rew, minlength=self.n) / self.counts

    def stats(self, adaptive):
        gap = adaptive - self.best
        return dict(mean_adaptive=float(adaptive.mean()), mean_best=float(self.best.mean()),
                    mean_gap=float(gap.mean()),
                    pct_of_best=float(adaptive.mean() / self.best.mean() * 100),
                    win_best=float((gap > 0).mean() * 100),
                    win_naive=float((adaptive > self.naive).mean() * 100),
                    worst_loss=float(gap.min()))


def bisect(g, lo, hi, log=False, iters=60):
    """Find x where g(x) crosses 0 (g monotone).  Returns None if no crossing."""
    glo, ghi = g(lo), g(hi)
    if glo == 0:
        return lo
    if np.sign(glo) == np.sign(ghi):
        return None
    for _ in range(iters):
        mid = np.sqrt(lo * hi) if log else (lo + hi) / 2
        gm = g(mid)
        if np.sign(gm) == np.sign(glo):
            lo, glo = mid, gm
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD THE POPULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_named(agent, min_dwell=1):
    traces, best, naive = [], [], []
    for key, sc in SCENARIOS.items():
        cond = sc.flatten()
        traces.append(trace(agent, cond, min_dwell))
        best.append(static_avg_reward(cond, find_best_static_action(sc)))
        naive.append(static_avg_reward(cond, NAIVE_STATIC_ACTION))
    return Pop("Six named scenarios", traces, best, naive), list(SCENARIOS.keys())


def load_random(agent, steps_path, summary_path, min_dwell=1):
    for p in (steps_path, summary_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found - run run_random_scenarios.py first.")
    conds, order = {}, []
    with open(steps_path, newline="") as f:
        for row in csv.DictReader(f):
            k = (row["generator_type"], int(row["scenario_index"]))
            if k not in conds:
                conds[k] = []
                order.append(k)
            # the CSV rounds intervals to 2 decimals, so a tiny Weibull draw can
            # appear as 0.00; the original value was > 0, so use 0.01 instead
            conds[k].append((int(row["step_index"]), int(row["ce_mode"]),
                             max(float(row["traffic_interval_s"]), 0.01)))
    summ = {}
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            summ[(row["generator_type"], int(row["scenario_index"]))] = row

    by_gen = {}
    for k in order:
        by_gen.setdefault(k[0], []).append(k)

    pops, worst_check, n_mismatch = [], 0.0, 0
    for gen, keys in by_gen.items():
        traces, best, naive = [], [], []
        for k in keys:
            cond = [(ce, iv) for _, ce, iv in sorted(conds[k])]
            tr = trace(agent, cond, min_dwell)
            row = summ[k]
            traces.append(tr)
            best.append(float(row["best_static_reward"]))
            naive.append(float(row["naive_static_reward"]))
            # consistency: does MY recomputation of the adaptive reward match the CSV?
            mine = np.mean([evaluate_action_on_condition(s["action"][0], s["action"][1],
                                                         s["ce"], s["iv"]).reward for s in tr])
            d = abs(mine - float(row["adaptive_reward"]))
            worst_check = max(worst_check, d)
            n_mismatch += d > 1e-6
        pops.append(Pop(gen, traces, best, naive))
    return pops, worst_check, int(n_mismatch), len(order)


# ─────────────────────────────────────────────────────────────────────────────
# 4. REPORTS
# ─────────────────────────────────────────────────────────────────────────────

def census(pops, per_switch_path):
    print("\n" + "=" * 100)
    print("A. SWITCH CENSUS - how often does the policy change timers?")
    print("=" * 100)
    print(f"{'Population':<52}{'scen':>6}{'switch/scen':>13}{'%scen>=1':>10}{'max':>6}"
          f"{'%free':>8}{'%saves':>8}")
    rows = []
    for pop in pops:
        sw = [s for s in pop.flat if s["switch"]]
        per = np.bincount(pop.scen[[i for i, s in enumerate(pop.flat) if s["switch"]]],
                          minlength=pop.n) if sw else np.zeros(pop.n, int)
        free = np.mean([s["free"] for s in sw]) * 100 if sw else 0.0
        saves = np.mean([s["dP_mw"] > 0 for s in sw]) * 100 if sw else 0.0
        print(f"{pop.name:<52}{pop.n:>6}{per.mean():>13.2f}{(per > 0).mean() * 100:>9.1f}%"
              f"{per.max():>6}{free:>7.1f}%{saves:>7.1f}%")
        for i, s in enumerate(pop.flat):
            if s["switch"]:
                be = s["e_sw_mj"] / s["dP_mw"] if s["dP_mw"] > 0 else float("inf")
                rows.append([pop.name, int(pop.scen[i]), s["step"], s["ce"], round(s["iv"], 2),
                             s["old"][0], s["old"][1], s["action"][0], s["action"][1],
                             round(s["dP_mw"] * 1000, 4), round(s["e_sw_mj"], 1), s["free"],
                             "" if not np.isfinite(be) else round(be, 1)])
    print("  %free  = switches where a TAU is already due (interval > old T3412), so new timers can ride on it")
    print("  %saves = switches that lower average power; the rest trade energy for lower delay")

    print("\n  Break-even hold time of ENERGY-SAVING switches (E_TAU / power saved):")
    print(f"  {'Population':<52}{'25%':>10}{'median':>10}{'75%':>10}{'90%':>10}{'never pays':>12}")
    for pop in pops:
        sw = [s for s in pop.flat if s["switch"]]
        be = np.array([s["e_sw_mj"] / s["dP_mw"] for s in sw if s["dP_mw"] > 0])
        npay = np.mean([s["dP_mw"] <= 0 for s in sw]) * 100 if sw else 0
        if len(be):
            q = np.percentile(be, [25, 50, 75, 90])
            print(f"  {pop.name:<52}{fmt_s(q[0]):>10}{fmt_s(q[1]):>10}{fmt_s(q[2]):>10}{fmt_s(q[3]):>10}{npay:>11.1f}%")
    with open(per_switch_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "scenario", "step", "ce", "interval_s", "old_T3412", "old_T3324",
                    "new_T3412", "new_T3324", "power_saved_uW", "tau_energy_mJ", "free_ride_if_due",
                    "break_even_hold_s"])
        w.writerows(rows)


def tau_equivalence():
    print("\n" + "=" * 100)
    print("B. WHY ONE EXTRA TAU MATTERS - its energy expressed as hours of deep sleep")
    print("=" * 100)
    for ce in range(3):
        e = E_TAU_BASE_MJ * CE_SYNC_SCALE[ce]
        print(f"  CE-{'ABC'[ce]}: one TAU = {e:7.1f} mJ = {e * 1e-3 / (P_PSM_MW * 1e-3) / 3600:5.1f} hours of PSM at {P_PSM_MW * 1000:.1f} uW")


def switch_sweep(pops, named_keys, sweep_path):
    print("\n" + "=" * 100)
    print("C. SWITCH COST - does adaptive still beat best-static after paying a TAU per change?")
    print("   (gap = adaptive - best static, reward points; step hold time H is swept)")
    print("=" * 100)
    out = [["mode", "population", "H_s", "mean_gap", "pct_of_best", "win_vs_best_%", "win_vs_naive_%"]]
    for mode in MODES:
        label = "every switch pays a full extra TAU" if mode == "never" else "free when a TAU is already due"
        print(f"\n  MODE '{mode}': {label}")
        print(f"  {'Population':<44}" + "".join(f"{h:>12}" for h in ["no cost"] + H_LABELS) + f"{'break-even H*':>16}")
        for pop in pops:
            cells, base = [], pop.stats(pop.scen_rewards())
            cells.append(f"{base['mean_gap']:+.4f}")
            out.append([mode, pop.name, "inf", base["mean_gap"], base["pct_of_best"], base["win_best"], base["win_naive"]])
            for H in H_GRID_S:
                s = pop.stats(pop.scen_rewards(H=H, mode=mode))
                cells.append(f"{s['mean_gap']:+.4f}")
                out.append([mode, pop.name, H, s["mean_gap"], s["pct_of_best"], s["win_best"], s["win_naive"]])
            g = lambda H, p=pop, m=mode: p.stats(p.scen_rewards(H=H, mode=m))["mean_gap"]
            if base["mean_gap"] <= 0:
                be = "already <= 0"
            else:
                x = bisect(g, 60.0, 365 * 86400.0, log=True)
                be = "> 365 d (safe)" if x is None else fmt_s(x)
            print(f"  {pop.name:<44}" + "".join(f"{c:>12}" for c in cells) + f"{be:>16}")
        # win rate line for the random populations at 1 day
        print("  Win rate vs best static at H = 1 day:  " + "   ".join(
            f"{p.name.split(' traffic')[0][:12]}: {p.stats(p.scen_rewards(H=86400, mode=mode))['win_best']:.1f}%"
            for p in pops[1:]))

    # second reading of "how long is a step": k reporting intervals
    print("\n  ALTERNATIVE READING - a step lasts k reporting intervals (k x that step's traffic interval),")
    print("  i.e. conditions can change every few reports.  Gap = adaptive - best static, mode 'never':")
    print(f"  {'Population':<44}" + "".join(f"{'k=' + str(k):>10}" for k in K_GRID) + f"{'break-even k*':>16}")
    for pop in pops:
        cells = [f"{pop.stats(pop.scen_rewards(k=k))['mean_gap']:+.4f}" for k in K_GRID]
        for k in K_GRID:
            s_ = pop.stats(pop.scen_rewards(k=k))
            out.append(["never_k", pop.name, f"k={k}", s_["mean_gap"], s_["pct_of_best"], s_["win_best"], s_["win_naive"]])
        g = lambda k, p=pop: p.stats(p.scen_rewards(k=k))["mean_gap"]
        base = pop.stats(pop.scen_rewards())["mean_gap"]
        if base <= 0:
            be = "already <= 0"
        else:
            x = bisect(g, 0.01, 1e6, log=True)
            be = "> 1e6 (safe)" if x is None else f"{x:.1f}"
        print(f"  {pop.name:<44}" + "".join(f"{c:>10}" for c in cells) + f"{be:>16}")

    # per named scenario
    named = pops[0]
    print("\n  Per named scenario, gap at H = 1 day (mode 'never'):")
    adaptive = named.scen_rewards(H=86400, mode="never")
    base = named.scen_rewards()
    nsw = np.bincount(named.scen[[i for i, s in enumerate(named.flat) if s["switch"]]], minlength=named.n)
    print(f"  {'scenario':<26}{'switches':>9}{'gap(no cost)':>14}{'gap(1 day)':>12}")
    for i, k in enumerate(named_keys):
        print(f"  {k:<26}{nsw[i]:>9}{base[i] - named.best[i]:>+14.4f}{adaptive[i] - named.best[i]:>+12.4f}")
    with open(sweep_path, "w", newline="") as f:
        csv.writer(f).writerows(out)


def monitoring_sweep(pops, named_keys, sweep_path):
    print("\n" + "=" * 100)
    print("D. ALWAYS-ON MONITORING - how many microwatts can the agent afford while asleep?")
    print("   (extra constant power added to the ADAPTIVE device only; best-static pays nothing)")
    print("=" * 100)
    print("  For scale: PSM = %.1f uW.  Average power under the policy:" % (P_PSM_MW * 1000))
    for pop in pops:
        pw = np.bincount(pop.scen, weights=pop.p, minlength=pop.n) / pop.counts
        print(f"    {pop.name:<52} median {np.median(pw) * 1000:9.1f} uW   lowest {pw.min() * 1000:8.1f} uW")
    out = [["population", "monitoring_uW", "mean_gap", "pct_of_best", "win_vs_best_%", "win_vs_naive_%"]]
    print(f"\n  {'Population':<44}" + "".join(f"{v:>9g}" for v in MON_GRID_UW) + f"{'break-even uW':>16}")
    for pop in pops:
        cells = []
        for m in MON_GRID_UW:
            s = pop.stats(pop.scen_rewards(mon_uw=m))
            cells.append(f"{s['mean_gap']:+.4f}")
            out.append([pop.name, m, s["mean_gap"], s["pct_of_best"], s["win_best"], s["win_naive"]])
        g = lambda m, p=pop: p.stats(p.scen_rewards(mon_uw=m))["mean_gap"]
        if g(0.0) <= 0:
            be = "already <= 0"
        else:
            x = bisect(g, 0.0, 5000.0)
            be = "> 5000 (safe)" if x is None else f"{x:.2f}"
        print(f"  {pop.name:<44}" + "".join(f"{c:>9}" for c in cells) + f"{be:>16}")
    print("\n  Win rate vs best static as monitoring power grows (random scenarios):")
    for pop in pops[1:]:
        print(f"    {pop.name.split(' traffic')[0]:<22}" + "".join(
            f"{pop.stats(pop.scen_rewards(mon_uw=m))['win_best']:>9.1f}" for m in MON_GRID_UW))

    named = pops[0]
    print("\n  Per named scenario - break-even monitoring power (uW) where the scenario's gap hits zero:")
    for i, k in enumerate(named_keys):
        gi = lambda m, i=i: named.scen_rewards(mon_uw=m)[i] - named.best[i]
        if gi(0.0) <= 0:
            print(f"    {k:<26}already behind best static without any overhead ({gi(0.0):+.4f})")
        else:
            x = bisect(gi, 0.0, 5000.0)
            print(f"    {k:<26}{'> 5000' if x is None else f'{x:.2f}'} uW   (gap without overhead {gi(0.0):+.4f})")
    with open(sweep_path, "w", newline="") as f:
        csv.writer(f).writerows(out)


def dwell_sweep(agent):
    print("\n" + "=" * 100)
    print("E. A SIMPLE SWITCHING RULE - once timers change, hold them for at least m steps")
    print("   (m = 1 is the policy as trained.  Gap = adaptive - best static; every switch pays a TAU)")
    print("=" * 100)
    steps_p = os.path.join(RESULTS_DIR, "random_scenarios_steps.csv")
    sum_p = os.path.join(RESULTS_DIR, "random_scenarios_results.csv")
    print(f"  {'m':>3}  {'Population':<44}{'switch/scen':>12}{'no cost':>10}{'6 hours':>10}{'1 day':>10}{'win@1d %':>10}")
    for m in DWELL_GRID:
        named, _ = load_named(agent, m)
        rand, _, _, _ = load_random(agent, steps_p, sum_p, m)
        for pop in [named] + rand:
            nsw = np.sum([s["switch"] for s in pop.flat]) / pop.n
            g0 = pop.stats(pop.scen_rewards())["mean_gap"]
            g6 = pop.stats(pop.scen_rewards(H=6 * 3600))["mean_gap"]
            g24 = pop.stats(pop.scen_rewards(H=86400))
            print(f"  {m:>3}  {pop.name:<44}{nsw:>12.2f}{g0:>+10.4f}{g6:>+10.4f}{g24['mean_gap']:>+10.4f}{g24['win_best']:>10.1f}")
        print()


def make_plot(pops, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                    # plotting is optional
        print(f"[plot skipped: {e}]")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Hs = np.logspace(np.log10(600), np.log10(60 * 86400), 40)
    for pop in pops:
        short = pop.name.split(" traffic")[0]
        ax[0].plot(Hs / 86400, [pop.stats(pop.scen_rewards(H=h, mode="never"))["mean_gap"] for h in Hs], label=short)
        ms = np.linspace(0, 50, 41)
        ax[1].plot(ms, [pop.stats(pop.scen_rewards(mon_uw=m))["mean_gap"] for m in ms], label=short)
    for a in ax:
        a.axhline(0, color="k", lw=0.8)
        a.set_ylabel("adaptive minus best static (reward)")
        a.grid(alpha=0.3)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("hold time per step H (days), every switch pays a TAU")
    ax[0].set_title("Switching cost")
    ax[1].set_xlabel("always-on monitoring power (uW), adaptive only")
    ax[1].set_title("Monitoring while asleep")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────────────────────────────────────

def analyse(agent, make_figure=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    named, named_keys = load_named(agent)
    rand, worst, n_bad, n_all = load_random(agent,
                              os.path.join(RESULTS_DIR, "random_scenarios_steps.csv"),
                              os.path.join(RESULTS_DIR, "random_scenarios_results.csv"))
    pops = [named] + rand

    # consistency checks - if these fail, the q_table and the CSVs are from different runs
    zero = named.stats(named.scen_rewards())
    print("CHECK 1 - named scenarios with zero overhead (must equal evaluate.py / Table 4.1):")
    print(f"   adaptive {zero['mean_adaptive']:+.4f}   best static {zero['mean_best']:+.4f}   "
          f"naive {named.naive.mean():+.4f}")
    print(f"CHECK 2 - random scenarios: {n_bad} of {n_all} differ from the CSV's adaptive_reward "
          f"(largest difference {worst:.2e})  "
          f"({'OK' if n_bad <= 0.005 * n_all else 'MISMATCH: q_table.npy is not the one that made the CSV'})")
    for pop in rand:
        s = pop.stats(pop.scen_rewards())
        print(f"   {pop.name:<52} win vs best {s['win_best']:.1f}%   vs naive {s['win_naive']:.1f}%   "
              f"gap {s['mean_gap']:+.4f}")

    census(pops, os.path.join(RESULTS_DIR, "overhead_switches.csv"))
    tau_equivalence()
    switch_sweep(pops, named_keys, os.path.join(RESULTS_DIR, "overhead_switch_sweep.csv"))
    monitoring_sweep(pops, named_keys, os.path.join(RESULTS_DIR, "overhead_monitoring_sweep.csv"))
    dwell_sweep(agent)
    if make_figure:
        make_plot(pops, os.path.join(RESULTS_DIR, "figL_overhead.png"))
    print("\n[SAVED] overhead_switches.csv, overhead_switch_sweep.csv, overhead_monitoring_sweep.csv in", RESULTS_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Energy cost of adapting: switching + always-on monitoring")
    ap.add_argument("--qtable", default=os.path.join(RESULTS_DIR, "q_table.npy"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.qtable):
        sys.exit(f"Q-table not found at {args.qtable} - run train.py first.")
    agent = QLearningAgent(n_states=len(CE_MODES) * len(TRAFFIC_BUCKETS), n_actions=len(ACTIONS))
    agent.load(args.qtable)
    analyse(agent, make_figure=not args.no_plot)
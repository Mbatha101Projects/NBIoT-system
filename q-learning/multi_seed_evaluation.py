#!/usr/bin/env python3
"""
multi_seed_evaluation.py  --  answers review weakness 3 ("seed 11 was chosen
by an audit, so its result is not an average").

WHAT IT DOES
  Trains the agent with several seeds (default 1-20, the same seeds as
  seed_search.py), and for EACH seed measures:
    * audit score  (T3412-only within 0.02, as in seed_search.py) and exact-pair score
    * the 3,000 random scenarios (same generators, same rng(42) draw order as
      run_random_scenarios.py, so every seed faces IDENTICAL scenarios)
    * the 6 named scenarios
  and then reports mean +/- SD, min, max over seeds, with seed 11 shown
  separately so the paper can say "seed 11 = X, 20-seed mean = Y +/- Z".

SAFE: it never touches your production results/q_table.npy. It backs that
file (and training_log.csv) up first and restores them at the end, even if
you Ctrl-C. Per-seed tables go to results/seeds/.

RUN (from your q-learning folder, AFTER the smp_engine.py fix):
    python multi_seed_evaluation.py                 # all 20 seeds, 3000 scenarios (slow: see below)
    python multi_seed_evaluation.py --reuse         # reuse tables already in results/seeds/
    python multi_seed_evaluation.py --seeds 1 2 3 --episodes 3000 --n-per-type 30   # quick smoke test
  Time: roughly (training time of one seed) x 20, plus about a minute for the
  scenario baselines. Run it overnight or with --reuse after the first pass.

OUTPUT
  results/multi_seed_per_seed.csv    one row per seed
  results/multi_seed_summary.csv     mean/SD/min/max per metric
  results/multi_seed_named.csv       per seed x named scenario
"""
import sys, os, csv, shutil, inspect, argparse, time
sys.path.insert(0, ".")
import numpy as np

import evaluate as EV
from evaluate import run_adaptive, run_static, find_best_static_action, summarize
from agent import QLearningAgent
from environment import (ACTIONS, N_ACTIONS, CE_MODES, TRAFFIC_BUCKETS,
                         bucket_to_representative_interval,
                         normalize_lifetime, normalize_delay)
from smp_engine import compute_smp
from scenarios import SCENARIOS
import run_random_scenarios as RRS          # reuse the exact generators (has a __main__ guard)
import train as TR

PROD_FILES = ["results/q_table.npy", "results/training_log.csv"]
SEED_DIR = "results/seeds"
N_STATES = len(CE_MODES) * len(TRAFFIC_BUCKETS)
TOL = 0.02
FOCUS_SEED = 11


# ----------------------------------------------------------------- audit
def true_rewards(ce, b, T3324=None):
    iv = bucket_to_representative_interval(b)
    out = {}
    for (t1, t2) in ACTIONS:
        if T3324 is not None and t2 != T3324:
            continue
        r = compute_smp(t1, t2, iv, ce)
        out[(t1, t2)] = 0.7 * normalize_lifetime(r.lifetime_yr) - 0.45 * normalize_delay(r.delay_seconds)
    return out


def build_audit_reference():
    ref_pair, ref_t = {}, {}
    for ce in CE_MODES:
        for b in range(len(TRAFFIC_BUCKETS)):
            ref_pair[(ce, b)] = true_rewards(ce, b)
            ref_t[(ce, b)] = true_rewards(ce, b, T3324=20)
    return ref_pair, ref_t


def audit(q, ref_pair, ref_t):
    n_t = n_pair_within = n_exact = 0
    for ce in CE_MODES:
        for b in range(len(TRAFFIC_BUCKETS)):
            s = ce * len(TRAFFIC_BUCKETS) + b
            a = ACTIONS[int(np.argmax(q[s]))]
            rp = ref_pair[(ce, b)]
            best = max(rp, key=rp.get)
            n_exact += (a == best)
            n_pair_within += (rp[best] - rp[a] <= TOL)
            rt = ref_t[(ce, b)]
            bt = max(rt.values())
            lt = rt.get((a[0], 20))
            n_t += (lt is not None and bt - lt <= TOL)
    return n_t, n_pair_within, n_exact


# ------------------------------------------------------- scenario baselines
def make_random_scenarios(n_per_type):
    rng = np.random.default_rng(42)          # same seed + same draw order as run_random_scenarios.py
    scen = []
    for label, fn in [("Poisson", RRS.poisson_scenario),
                      ("Weibull-bursty", RRS.weibull_bursty_scenario),
                      ("ON-OFF", RRS.onoff_scenario)]:
        for _ in range(n_per_type):
            scen.append((label, fn(rng)))
    return scen


def precompute_baselines(scen):
    best, naive = [], []
    t0 = time.time()
    for i, (_, s) in enumerate(scen):
        ba = find_best_static_action(s)
        best.append(summarize(run_static(*ba, s))["avg_reward"])
        naive.append(summarize(run_static(3600, 20, s))["avg_reward"])
        if (i + 1) % 500 == 0:
            print(f"    baselines {i+1}/{len(scen)}  ({time.time()-t0:.0f}s)")
    return np.array(best), np.array(naive)


# ------------------------------------------------------------- training
def train_seed(seed, episodes):
    kw = dict(n_episodes=episodes, seed=seed, verbose=False)
    if "save_outputs" in inspect.signature(TR.train).parameters:
        kw["save_outputs"] = False
    return TR.train(**kw)


def get_agent(seed, episodes, reuse):
    path = os.path.join(SEED_DIR, f"q_table_seed{seed}.npy")
    if reuse and os.path.exists(path):
        ag = QLearningAgent(n_states=N_STATES, n_actions=N_ACTIONS)
        ag.load(path)
        return ag, "loaded"
    ag = train_seed(seed, episodes)
    ag.save(path)
    return ag, "trained"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    ap.add_argument("--episodes", type=int, default=60000)
    ap.add_argument("--n-per-type", type=int, default=1000)
    ap.add_argument("--reuse", action="store_true", help="reuse results/seeds/q_table_seed*.npy if present")
    args = ap.parse_args()

    os.makedirs(SEED_DIR, exist_ok=True)
    backups = {}
    for p in PROD_FILES:
        if os.path.exists(p):
            backups[p] = p + ".multiseed.bak"
            shutil.copy2(p, backups[p])

    try:
        ref_pair, ref_t = build_audit_reference()
        scen = make_random_scenarios(args.n_per_type)
        print(f"Random scenarios: {len(scen)}.  Pre-computing best-static / naive baselines ...")
        best, naive = precompute_baselines(scen)
        labels = np.array([l for l, _ in scen])

        per_seed, named_rows = [], []
        for seed in args.seeds:
            t0 = time.time()
            ag, how = get_agent(seed, args.episodes, args.reuse)
            q = ag.q_table
            n_t, n_pair, n_exact = audit(q, ref_pair, ref_t)

            ad = np.array([summarize(run_adaptive(ag, s))["avg_reward"] for _, s in scen])
            gap_best = ad - best
            gap_naive = ad - naive
            row = dict(
                seed=seed, source=how,
                audit_T3412_only=n_t, audit_pair_within=n_pair, audit_pair_exact=n_exact,
                adaptive_mean=ad.mean(), best_static_mean=best.mean(), naive_mean=naive.mean(),
                pct_of_best=100 * ad.mean() / best.mean(),
                mean_gap_vs_best=gap_best.mean(),
                win_vs_best_pct=100 * np.mean(gap_best > 0),
                win_vs_naive_pct=100 * np.mean(gap_naive > 0),
                worst_loss_vs_naive=gap_naive.min(),
                mean_gain_vs_naive=gap_naive.mean(),
            )
            for lab in ["Poisson", "Weibull-bursty", "ON-OFF"]:
                m = labels == lab
                row[f"gap_vs_best_{lab}"] = gap_best[m].mean()
            per_seed.append(row)

            for key, sc in SCENARIOS.items():
                a = summarize(run_adaptive(ag, sc))["avg_reward"]
                bs = find_best_static_action(sc)
                b = summarize(run_static(*bs, sc))["avg_reward"]
                n = summarize(run_static(3600, 20, sc))["avg_reward"]
                named_rows.append(dict(seed=seed, scenario=key, adaptive=a, best_static=b,
                                       naive=n, gap_vs_best=a - b, gap_vs_naive=a - n))
            print(f"  seed {seed:>2} ({how}, {time.time()-t0:.0f}s): audit {n_t}/30 (T3412 only), "
                  f"{n_pair}/30 within, {n_exact}/30 exact | adaptive {ad.mean():+.4f} "
                  f"vs best {best.mean():+.4f} (gap {gap_best.mean():+.4f}) | beats naive "
                  f"{100*np.mean(gap_naive>0):.1f}%")

        # ---------- write
        os.makedirs("results", exist_ok=True)
        with open("results/multi_seed_per_seed.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys())); w.writeheader(); w.writerows(per_seed)
        with open("results/multi_seed_named.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(named_rows[0].keys())); w.writeheader(); w.writerows(named_rows)

        metrics = ["audit_T3412_only", "audit_pair_within", "audit_pair_exact", "adaptive_mean",
                   "pct_of_best", "mean_gap_vs_best", "win_vs_best_pct", "win_vs_naive_pct",
                   "worst_loss_vs_naive", "mean_gain_vs_naive",
                   "gap_vs_best_Poisson", "gap_vs_best_Weibull-bursty", "gap_vs_best_ON-OFF"]
        focus = next((r for r in per_seed if r["seed"] == FOCUS_SEED), None)
        summ = []
        print("\n" + "=" * 92)
        print(f"ACROSS {len(per_seed)} SEEDS  (best-static mean {best.mean():+.4f}, naive mean {naive.mean():+.4f}, identical scenarios for all seeds)")
        print("=" * 92)
        print(f"{'metric':<30}{'mean':>10}{'SD':>10}{'min':>10}{'max':>10}" + (f"{'seed 11':>10}" if focus else ""))
        for m in metrics:
            v = np.array([r[m] for r in per_seed], dtype=float)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            summ.append(dict(metric=m, mean=v.mean(), sd=sd, min=v.min(), max=v.max(),
                             seed11=(focus[m] if focus else "")))
            print(f"{m:<30}{v.mean():>10.4f}{sd:>10.4f}{v.min():>10.4f}{v.max():>10.4f}"
                  + (f"{focus[m]:>10.4f}" if focus else ""))
        with open("results/multi_seed_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

        if focus is not None:
            ranks = sorted(per_seed, key=lambda r: -r["adaptive_mean"])
            rk = [r["seed"] for r in ranks].index(FOCUS_SEED) + 1
            print(f"\nSeed {FOCUS_SEED} ranks #{rk} of {len(per_seed)} on mean adaptive reward.")
        print("\nSuggested wording: 'Across 20 independent training seeds the adaptive policy reached "
              "X +/- Y % of the best static reward (seed 11, selected by audit, reached Z %).'")
        print("[SAVED] results/multi_seed_per_seed.csv, multi_seed_summary.csv, multi_seed_named.csv, results/seeds/")
    finally:
        for p, b in backups.items():          # always restore the production files
            shutil.copy2(b, p); os.remove(b)
        print("[RESTORED] production q_table.npy / training_log.csv")


if __name__ == "__main__":
    main()
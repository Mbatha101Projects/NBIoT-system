#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh  —  NB-IoT SMP Benchmark Sweep Runner  (v3)
#
# University of Zululand | Honours CS | 2026
# Student : Mawande Mbatha (202396115)
# Supervisor: Dr Paul Tarwireyi
#
# SETUP — the CC file must be in scratch/ BEFORE running this script:
#   cp nbiot-smp-benchmark.cc ~/ns-3-dev/scratch/
#   cp run_benchmark.sh       ~/ns-3-dev/
#   cp plot_results.py        ~/ns-3-dev/analysis/
#   chmod +x ~/ns-3-dev/run_benchmark.sh
#   cd ~/ns-3-dev
#   bash run_benchmark.sh --debug     # verify one run first
#   bash run_benchmark.sh             # full 106-config sweep
#
# CONFIGURATION GROUPS (106 total)
# ─────────────────────────────────────────────────────────────────────────────
# Group A1 (25): T3412 increases 60s → 86400s, T3324=20s fixed, CE-A
# Group A2 (25): T3324 increases 5s → 3595s,   T3412=3600s fixed, CE-A
# Group B  ( 9): CE mode comparison (CE-A/B/C × T3412∈{600,3600,14400})
# Group C  ( 6): Traffic profiles TC1-TC5
# Group D  ( 5): TAU threshold sweep (interval=7200s, T3412 varies)
# Group E  (36): Literature comparison grid — T3412∈{0,3,6,9,12,15}h
#                × T3324∈{5,10,15,20,25,30}s, CE-A, no TAU (see Section below)
#
# USAGE
# ─────
# bash run_benchmark.sh          # parallel sweep (all CPU cores)
# bash run_benchmark.sh --debug  # single test config, raw output shown
# bash run_benchmark.sh --seq    # sequential (safer on low-RAM machines)
# =============================================================================

set -euo pipefail

NS3_DIR="${HOME}/ns-3-dev"
CC_IN_SCRATCH="${NS3_DIR}/scratch/nbiot-smp-benchmark.cc"
RESULTS_DIR="${NS3_DIR}/results"
CSV_FILE="${RESULTS_DIR}/results.csv"
LOG_FILE="${RESULTS_DIR}/sweep_debug.log"
SIM_TIME=120    # seconds. v3: lowered from 7200 — see run_group_e_only.sh
                # for full root-cause explanation. NS-3's LTE PHY/MAC
                # housekeeping runs for the full simTime window regardless
                # of traffic, so a 7200s window costs real wall-clock time
                # even when interval >> simTime means almost nothing happens
                # after attach. 120s is enough to observe attach + first
                # packet; the analytical SMP engine doesn't depend on
                # simTime at all, so this has zero effect on energy/
                # lifetime/PSF results — only NS-3's own latency/PDR sample.

MODE="parallel"
if [[ "${1:-}" == "--debug" ]]; then MODE="debug"; fi
if [[ "${1:-}" == "--seq"   ]]; then MODE="seq";   fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Sanity checks ──────────────────────────────────────────────────────────
[[ -d "${NS3_DIR}" ]]      || error "NS-3 directory not found: ${NS3_DIR}"
[[ -f "${CC_IN_SCRATCH}" ]] || error "CC file not found: ${CC_IN_SCRATCH}
  → Fix: cp nbiot-smp-benchmark.cc ~/ns-3-dev/scratch/"

cd "${NS3_DIR}"
mkdir -p "${RESULTS_DIR}/figures"

# ── Configure NS-3 if not yet done ─────────────────────────────────────────
if [[ ! -f "${NS3_DIR}/CMakeCache.txt" ]]; then
    warn "CMakeCache.txt missing — running ./ns3 configure"
    ./ns3 configure --enable-examples 2>&1 | tail -5
fi

# ── Build ──────────────────────────────────────────────────────────────────
info "Building nbiot-smp-benchmark from scratch/..."
./ns3 build nbiot-smp-benchmark 2>&1 | grep -E "(error:|warning:|Built target)" || true
info "Build complete."

# ── Debug mode ─────────────────────────────────────────────────────────────
if [[ "${MODE}" == "debug" ]]; then
    info "DEBUG MODE — CE-A, T3412=3600s, T3324=20s, interval=3600s"
    ./ns3 run nbiot-smp-benchmark -- \
        --T3412=3600 --T3324=20 --interval=3600 \
        --simTime="${SIM_TIME}" --ceMode=0 --label=DEBUG
    exit 0
fi

# ── Full sweep via embedded Python ─────────────────────────────────────────
info "Starting sweep (mode=${MODE})..."

python3 - <<PYEOF
import subprocess, os, csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed

NS3_DIR     = "${NS3_DIR}"
LOG_FILE    = "${LOG_FILE}"
CSV_FILE    = "${CSV_FILE}"
SIM_TIME    = ${SIM_TIME}
PARALLEL    = ("${MODE}" == "parallel")
MAX_WORKERS = os.cpu_count() or 2

CSV_HEADER = [
    "group","label","T3412","T3324","interval","ceMode",
    "sent","received","PDR","latency_ms",
    "energy_mWh","lifetime_yr","PSF_pct",
    "tauFired","p_avg_mw"
]

# ══════════════════════════════════════════════════════════════════════════════
# GROUP A1 — T3412 SWEEP (25 configs)
# T3412: log-spaced from 60s to 86400s, T3324=20s fixed, CE-A
# interval = T3412 → tx_frac=1.0 always, no TAU, clean monotonic curve
# ══════════════════════════════════════════════════════════════════════════════
import math

t3412_raw = [int(round(60 * (86400/60) ** (i/24))) for i in range(25)]
t3412_vals = sorted(set(t3412_raw))[:25]

GROUP_A1 = []
for idx, t3412 in enumerate(t3412_vals):
    GROUP_A1.append((
        "A1",
        f"A1_T3412_{idx+1:02d}_{t3412}s",
        t3412, 20, t3412, 0
    ))

# ══════════════════════════════════════════════════════════════════════════════
# GROUP A2 — T3324 SWEEP (25 configs)
# T3324: from 5s to 3595s, T3412=3600s, CE-A
# Expected: near-flat curves — proves ACTIVE ≈ 0.015mW (v2 correction)
# ══════════════════════════════════════════════════════════════════════════════
t3324_raw = [int(round(5 * (3595/5) ** (i/24))) for i in range(25)]
t3324_vals = sorted(set(t3324_raw))[:25]

GROUP_A2 = []
for idx, t3324 in enumerate(t3324_vals):
    GROUP_A2.append((
        "A2",
        f"A2_T3324_{idx+1:02d}_{t3324}s",
        3600, t3324, 3600, 0
    ))

# ══════════════════════════════════════════════════════════════════════════════
# GROUP B — CE MODE COMPARISON (9 configs)
# T3412 ∈ {600, 3600, 14400} × CE ∈ {0, 1, 2}
# ══════════════════════════════════════════════════════════════════════════════
GROUP_B = []
for t3412 in [600, 3600, 14400]:
    for ce in [0, 1, 2]:
        ce_name = ["CEA","CEB","CEC"][ce]
        t_label = {600:"10m", 3600:"1h", 14400:"4h"}[t3412]
        GROUP_B.append((
            "B", f"B_{ce_name}_{t_label}",
            t3412, 20, t3412, ce
        ))

# ══════════════════════════════════════════════════════════════════════════════
# GROUP C — TRAFFIC PROFILES (6 configs)
# ══════════════════════════════════════════════════════════════════════════════
GROUP_C = [
    ("C", "TC1_SmartMeter",    7200,  20,  7200, 0),
    ("C", "TC2_EnvSensor",     3600,  20,  3600, 0),
    ("C", "TC3a_Agri_dry",     7200,  20,  7200, 1),
    ("C", "TC3b_Agri_irr",      900,  10,   900, 1),
    ("C", "TC4_Wildlife_CEC",   300,   5,   300, 2),
    ("C", "TC5_Industrial",    1800,  20,  1800, 0),
]

# ══════════════════════════════════════════════════════════════════════════════
# GROUP D — TAU THRESHOLD SWEEP (5 configs)
# interval=7200s fixed; T3412 varies across the boundary
# T3412 < 7200 → TAU fires;  T3412 >= 7200 → no TAU
# ══════════════════════════════════════════════════════════════════════════════
GROUP_D = []
for t3412 in [1800, 3600, 7200, 14400, 28800]:
    tag   = "TAU"  if t3412 < 7200 else ("EDGE" if t3412 == 7200 else "noTAU")
    GROUP_D.append((
        "D", f"D_{tag}_{t3412}s",
        t3412, 20, 7200, 0
    ))

# ══════════════════════════════════════════════════════════════════════════════
# GROUP E — LITERATURE COMPARISON GRID (36 configs)
# X-axis variable : T3412 in {0, 3, 6, 9, 12, 15} hours (0h shown as 5 min anchor)
# Lines (6)       : T3324 in {5, 10, 15, 20, 25, 30} seconds
# CE-A fixed. interval fixed at 54000s (15h) — longer than every T3412 value
# used here, so tx_frac=1.0 and NO TAU fires for any point in this grid. This
# isolates the pure T3412/T3324 timer relationship for direct comparison
# against literature lifetime-vs-T3412 curves (e.g. Sørensen Fig.7/8,
# Andres-Maldonado Fig.3) without TAU onset complicating the shape.
#
# NOTE: T3412=0h is not physically meaningful for PSM (device would never
# sleep), so it is represented here as 300s (5 min) — a near-zero anchor
# point that still lets the x-axis display a "0" tick per the requested
# 0,3,6,9,12,15 spacing.
# ══════════════════════════════════════════════════════════════════════════════
T3412_HOURS  = [0, 3, 6, 9, 12, 15]
T3412_SECONDS = {0: 300}  # 0h anchor → 300s (5 min), avoids T3412=0 division issues
for h in T3412_HOURS[1:]:
    T3412_SECONDS[h] = h * 3600

T3324_SECONDS_LIST = [5, 10, 15, 20, 25, 30]

# interval fixed and longer than the largest T3412 (15h = 54000s) so that
# tx_frac=1.0 and tauFired=0 for every single point in this grid
E_INTERVAL_S = 54000  # 15 hours — equal to the largest T3412 tested

GROUP_E = []
for t3412_h in T3412_HOURS:
    t3412_s = T3412_SECONDS[t3412_h]
    for t3324_s in T3324_SECONDS_LIST:
        label = f"E_T3412_{t3412_h}h_T3324_{t3324_s}s"
        GROUP_E.append((
            "E", label, t3412_s, t3324_s, E_INTERVAL_S, 0
        ))

ALL_CONFIGS = GROUP_A1 + GROUP_A2 + GROUP_B + GROUP_C + GROUP_D + GROUP_E
print(f"[SWEEP] Total configs : {len(ALL_CONFIGS)}")
print(f"[SWEEP]   A1 (T3412 sweep)        : {len(GROUP_A1)}")
print(f"[SWEEP]   A2 (T3324 sweep)        : {len(GROUP_A2)}")
print(f"[SWEEP]   B  (CE modes)           : {len(GROUP_B)}")
print(f"[SWEEP]   C  (Traffic)            : {len(GROUP_C)}")
print(f"[SWEEP]   D  (TAU sweep)          : {len(GROUP_D)}")
print(f"[SWEEP]   E  (Literature compare) : {len(GROUP_E)}")

# ══════════════════════════════════════════════════════════════════════════════
def run_one(cfg):
    group, label, T3412, T3324, interval, ce = cfg
    cmd = [
        f"{NS3_DIR}/ns3", "run", "nbiot-smp-benchmark", "--",
        f"--T3412={T3412}",
        f"--T3324={T3324}",
        f"--interval={interval}",
        f"--simTime={SIM_TIME}",
        f"--ceMode={ce}",
        f"--label={label}",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=600, cwd=NS3_DIR
        )
        return cfg, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return cfg, "", "TIMEOUT after 600s"
    except Exception as ex:
        return cfg, "", str(ex)

def parse_result(line):
    """
    Parse RESULT line — 15 tokens (indices 0-14):
    [0]  RESULT
    [1]  label
    [2]  T3412      [3]  T3324    [4]  interval
    [5]  ceMode     [6]  sent     [7]  received
    [8]  PDR        [9]  latency  [10] energy_mWh
    [11] lifetime   [12] PSF      [13] tauFired
    [14] p_avg_mw
    """
    parts = line.split()
    if len(parts) < 13 or parts[0] != "RESULT":
        return None
    try:
        return {
            "label":       parts[1],
            "T3412":       int(parts[2]),
            "T3324":       int(parts[3]),
            "interval":    int(parts[4]),
            "ceMode":      int(parts[5]),
            "sent":        int(parts[6]),
            "received":    int(parts[7]),
            "PDR":         float(parts[8]),
            "latency_ms":  float(parts[9]),
            "energy_mWh":  float(parts[10]),
            "lifetime_yr": float(parts[11]),
            "PSF_pct":     float(parts[12]),
            "tauFired":    int(parts[13])   if len(parts) > 13 else 0,
            "p_avg_mw":    float(parts[14]) if len(parts) > 14 else None,
        }
    except (ValueError, IndexError):
        return None

print(f"[SWEEP] Mode: {'parallel' if PARALLEL else 'sequential'}, "
      f"Workers: {MAX_WORKERS if PARALLEL else 1}")

rows, log_lines = [], []
t0 = time.time()

def process(cfg, output, error_msg):
    group, label = cfg[0], cfg[1]
    log_lines.append(f"\n=== [{group}] {label} ===\n{output}")
    if error_msg:
        print(f"  [FAIL] [{group}] {label}: {error_msg}")
        return None
    for line in output.splitlines():
        row = parse_result(line.strip())
        if row:
            row["group"] = group
            tau_str = "  TAU=YES" if row.get("tauFired") else ""
            print(f"  [OK]  [{group}] {label:38s} "
                  f"T3412={row['T3412']:6d}s  T3324={row['T3324']:5d}s  "
                  f"E={row['energy_mWh']:8.4f}mWh  "
                  f"L={row['lifetime_yr']:7.2f}yr  "
                  f"PSF={row['PSF_pct']:6.2f}%{tau_str}")
            return row
    print(f"  [WARN] [{group}] {label}: no RESULT line in output")
    return None

if PARALLEL:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, cfg): cfg for cfg in ALL_CONFIGS}
        for fut in as_completed(futures):
            cfg, output, err = fut.result()
            row = process(cfg, output, err)
            if row: rows.append(row)
else:
    for cfg in ALL_CONFIGS:
        cfg_out, output, err = run_one(cfg)
        row = process(cfg_out, output, err)
        if row: rows.append(row)

# Sort to preserve group order in CSV
order = {"A1":0,"A2":1,"B":2,"C":3,"D":4,"E":5}
rows.sort(key=lambda r: (
    order.get(r.get("group","Z"), 99),
    r.get("T3412", 0),
    r.get("T3324", 0)
))

with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})

elapsed = time.time() - t0
print(f"\n[SWEEP] Done. {len(rows)}/{len(ALL_CONFIGS)} configs parsed.")
print(f"[SWEEP] Elapsed : {elapsed/60:.1f} min")
print(f"[SWEEP] CSV     : {CSV_FILE}")
print(f"[SWEEP] Log     : {LOG_FILE}")
PYEOF

info "Sweep complete. Results in ${RESULTS_DIR}"
info "Next: cd ~/ns-3-dev/analysis && python3 plot_results.py"

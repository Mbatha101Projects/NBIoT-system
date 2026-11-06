#!/usr/bin/env bash
# =============================================================================
# run_group_f_with_seeds.sh  —  Group F Full Factorial WITH Seed Repetition
#
# University of Zululand | Honours CS | 2026
# Student : Mawande Mbatha (202396115)
#
# PURPOSE
# ───────
# Runs every Group F configuration (504 combinations) N_REPS times each,
# with a genuinely independent RNG sub-stream per repetition (via the v4
# CC file's --rngRun argument and RngSeedManager::SetRun()). This converts
# the static 504-row point-estimate dataset into a dataset with real
# statistical variance on latency/PDR — mean ± std-dev per configuration —
# which is the academically defensible way to use additional simulation
# time, rather than simply testing more (redundant, interpolatable) timer
# combinations. See conversation history for the analysis showing that
# computeSmp() is a smooth analytical function with no hidden behaviour
# between existing grid points — finer timer resolution adds no new
# information, but repeated stochastic NS-3 trials do.
#
# IMPORTANT — PREREQUISITE
# ──────────────────────────
# This script REQUIRES the v4 CC file (with --rngRun support and
# RngSeedManager seeding) to be in scratch/. The earlier v2/v3 CC file
# does NOT have this fix — running this script against the old CC file
# will silently produce N identical copies of every row, since NS-3's
# RNG would not actually vary between repetitions.
#
#   cp nbiot-smp-benchmark.cc ~/ns-3-dev/scratch/   # v4, with --rngRun
#
# OUTPUT
# ──────
# Writes results_group_f_seeded.csv — does NOT touch results.csv,
# results_group_e.csv, or results_group_f.csv (the unseeded version).
# Total rows = 504 configs × N_REPS repetitions.
#
# USAGE
#   cp run_group_f_with_seeds.sh ~/ns-3-dev/
#   chmod +x ~/ns-3-dev/run_group_f_with_seeds.sh
#   cd ~/ns-3-dev
#   bash run_group_f_with_seeds.sh            # default 5 reps, parallel
#   bash run_group_f_with_seeds.sh --reps=10  # custom rep count
#   bash run_group_f_with_seeds.sh --seq      # sequential mode
# =============================================================================

set -euo pipefail

NS3_DIR="${HOME}/ns-3-dev"
CC_IN_SCRATCH="${NS3_DIR}/scratch/nbiot-smp-benchmark.cc"
RESULTS_DIR="${NS3_DIR}/results"
CSV_FILE="${RESULTS_DIR}/results_group_f_seeded.csv"
LOG_FILE="${RESULTS_DIR}/sweep_group_f_seeded_debug.log"
SIM_TIME=120
N_REPS=5

MODE="parallel"
for arg in "$@"; do
    case "$arg" in
        --seq) MODE="seq" ;;
        --reps=*) N_REPS="${arg#--reps=}" ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ -d "${NS3_DIR}" ]]       || error "NS-3 directory not found: ${NS3_DIR}"
[[ -f "${CC_IN_SCRATCH}" ]] || error "CC file not found: ${CC_IN_SCRATCH}"

# ── Prerequisite check: confirm the v4 CC file (with --rngRun) is in place ──
if ! grep -q "rngRun" "${CC_IN_SCRATCH}"; then
    error "scratch/nbiot-smp-benchmark.cc does NOT contain --rngRun support.
  This means RNG repetitions would be IDENTICAL, not independent.
  → Fix: copy the v4 CC file (with RngSeedManager seeding) to scratch/ first."
fi

cd "${NS3_DIR}"
mkdir -p "${RESULTS_DIR}/figures"

if [[ ! -f "${NS3_DIR}/CMakeCache.txt" ]]; then
    ./ns3 configure --enable-examples 2>&1 | tail -5
fi

info "Building (skips if already up to date)..."
./ns3 build nbiot-smp-benchmark 2>&1 | grep -E "(error:|Built target)" || true

info "Starting Group F seeded sweep: 504 configs × ${N_REPS} reps = $((504*N_REPS)) runs (mode=${MODE})"

python3 - <<PYEOF
import subprocess, os, csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed

NS3_DIR     = "${NS3_DIR}"
CSV_FILE    = "${CSV_FILE}"
LOG_FILE    = "${LOG_FILE}"
SIM_TIME    = ${SIM_TIME}
N_REPS      = ${N_REPS}
PARALLEL    = ("${MODE}" == "parallel")
MAX_WORKERS = os.cpu_count() or 2

CSV_HEADER = [
    "group","label","T3412","T3324","interval","ceMode",
    "sent","received","PDR","latency_ms",
    "energy_mWh","lifetime_yr","PSF_pct",
    "tauFired","p_avg_mw","rngRun"
]

# ══════════════════════════════════════════════════════════════════════════════
# GROUP F GRID — same 504-config structure as run_group_f_full_grid.sh
# ══════════════════════════════════════════════════════════════════════════════
T3412_VALS    = [int(round(300 * (86400/300) ** (i/6))) for i in range(7)]
T3324_VALS    = [5, 10, 15, 20, 25, 30]
INTERVAL_VALS = [300, 1800, 7200, 86400]
CE_VALS       = [0, 1, 2]

BASE_CONFIGS = []
idx = 0
for t3412 in T3412_VALS:
    for t3324 in T3324_VALS:
        for interval in INTERVAL_VALS:
            for ce in CE_VALS:
                idx += 1
                ce_name = ["CEA","CEB","CEC"][ce]
                base_label = f"F_{idx:03d}_T{t3412}_D{t3324}_I{interval}_{ce_name}"
                BASE_CONFIGS.append((base_label, t3412, t3324, interval, ce))

# Expand each base config into N_REPS runs, each with a distinct rngRun index
ALL_RUNS = []
for base_label, t3412, t3324, interval, ce in BASE_CONFIGS:
    for rep in range(1, N_REPS + 1):
        label = f"{base_label}_r{rep}"
        ALL_RUNS.append(("F", label, t3412, t3324, interval, ce, rep))

print(f"[SWEEP] Group F seeded: {len(BASE_CONFIGS)} base configs × {N_REPS} reps "
      f"= {len(ALL_RUNS)} total runs")

# ══════════════════════════════════════════════════════════════════════════════
def run_one(cfg):
    group, label, T3412, T3324, interval, ce, rep = cfg
    cmd = [
        f"{NS3_DIR}/ns3", "run", "nbiot-smp-benchmark", "--",
        f"--T3412={T3412}", f"--T3324={T3324}",
        f"--interval={interval}", f"--simTime={SIM_TIME}",
        f"--ceMode={ce}", f"--rngRun={rep}", f"--label={label}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=120, cwd=NS3_DIR)
        return cfg, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return cfg, "", "TIMEOUT after 120s"
    except Exception as ex:
        return cfg, "", str(ex)

def parse_result(line):
    parts = line.split()
    if len(parts) < 13 or parts[0] != "RESULT":
        return None
    try:
        return {
            "label": parts[1], "T3412": int(parts[2]), "T3324": int(parts[3]),
            "interval": int(parts[4]), "ceMode": int(parts[5]),
            "sent": int(parts[6]), "received": int(parts[7]),
            "PDR": float(parts[8]), "latency_ms": float(parts[9]),
            "energy_mWh": float(parts[10]), "lifetime_yr": float(parts[11]),
            "PSF_pct": float(parts[12]),
            "tauFired": int(parts[13]) if len(parts) > 13 else 0,
            "p_avg_mw": float(parts[14]) if len(parts) > 14 else None,
            "rngRun": int(parts[15]) if len(parts) > 15 else None,
        }
    except (ValueError, IndexError):
        return None

rows, log_lines = [], []
t0 = time.time()
completed = 0

def process(cfg, output, error_msg):
    global completed
    completed += 1
    group, label = cfg[0], cfg[1]
    log_lines.append(f"\\n=== [{group}] {label} ===\\n{output}")
    if error_msg:
        print(f"  [FAIL] ({completed}/{len(ALL_RUNS)}) {label}: {error_msg}")
        return None
    for line in output.splitlines():
        row = parse_result(line.strip())
        if row:
            row["group"] = group
            if completed % 100 == 0 or completed == len(ALL_RUNS):
                print(f"  [OK]  ({completed}/{len(ALL_RUNS)}) {label}  "
                      f"PDR={row['PDR']:.1f}%  lat={row['latency_ms']:.2f}ms  "
                      f"rngRun={row.get('rngRun')}")
            return row
    print(f"  [WARN] {label}: no RESULT line found")
    return None

if PARALLEL:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, cfg): cfg for cfg in ALL_RUNS}
        for fut in as_completed(futures):
            cfg, output, err = fut.result()
            row = process(cfg, output, err)
            if row: rows.append(row)
else:
    for cfg in ALL_RUNS:
        cfg_out, output, err = run_one(cfg)
        row = process(cfg_out, output, err)
        if row: rows.append(row)

rows.sort(key=lambda r: (r.get("T3412",0), r.get("T3324",0),
                         r.get("interval",0), r.get("ceMode",0),
                         r.get("rngRun",0)))

with open(LOG_FILE, "w") as f:
    f.write("\\n".join(log_lines))

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})

elapsed = time.time() - t0
print(f"\\n[SWEEP] Done. {len(rows)}/{len(ALL_RUNS)} runs parsed.")
print(f"[SWEEP] Elapsed: {elapsed/60:.1f} min")
print(f"[SWEEP] CSV: {CSV_FILE}")

# ── Quick sanity check: confirm repetitions are NOT identical ────────────────
print("\\n[SANITY CHECK] Verifying repetitions show real variance...")
from collections import defaultdict
groups_by_base = defaultdict(list)
for row in rows:
    base = row["label"].rsplit("_r", 1)[0]
    groups_by_base[base].append(row["latency_ms"])

identical_count = 0
varying_count = 0
for base, lats in groups_by_base.items():
    if len(lats) > 1:
        if len(set(lats)) == 1:
            identical_count += 1
        else:
            varying_count += 1

print(f"[SANITY CHECK] Base configs with >1 rep: {identical_count + varying_count}")
print(f"[SANITY CHECK]   Showing real variance (GOOD): {varying_count}")
print(f"[SANITY CHECK]   All identical (BAD — RNG not working): {identical_count}")
if identical_count > 0:
    print(f"[SANITY CHECK] ⚠ WARNING: {identical_count} configs show ZERO variance")
    print(f"[SANITY CHECK]   across repetitions. This means RngSeedManager is")
    print(f"[SANITY CHECK]   not taking effect. Check that scratch/ has the v4")
    print(f"[SANITY CHECK]   CC file with --rngRun support, then rebuild.")
PYEOF

info "Group F seeded sweep complete: ${CSV_FILE}"

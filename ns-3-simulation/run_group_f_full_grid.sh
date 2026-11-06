#!/usr/bin/env bash
# =============================================================================
# run_group_f_full_grid.sh  —  Full Factorial Validation Grid (Group F)
#
# University of Zululand | Honours CS | 2026
# Student : Mawande Mbatha (202396115)
#
# PURPOSE
# ───────
# A genuine full-factorial sweep across the four key dimensions, sized to
# serve RQ2 (how static configs affect energy/latency/reliability across
# varying traffic AND coverage) and to provide a dense static-baseline pool
# for Objective 3 / RQ4 (adaptive vs static comparison, done later by the
# Q-learning agent).
#
# GRID STRUCTURE (504 rows = 7 × 6 × 4 × 3)
# ───────────────────────────────────────────
#   T3412    : 7 values, log-spaced 300s (5min) → 86400s (24h)
#   T3324    : 6 values, 5/10/15/20/25/30 seconds
#   interval : 4 values, 300s/1800s/7200s/86400s (5min/30min/2h/24h)
#   CE mode  : 3 values, 0=CE-A (Good) / 1=CE-B (Medium) / 2=CE-C (Poor)
#
# This uniform log-spaced design naturally includes TAU-boundary-crossing
# configurations (~18% of T3412×interval pairs sit within 2× of the
# boundary) without needing manual oversampling — verified analytically
# before building this grid.
#
# DOES NOT TOUCH your existing results.csv, results_group_e.csv, or any
# other sweep output. Writes only to results_group_f.csv.
#
# USAGE
#   cp run_group_f_full_grid.sh ~/ns-3-dev/
#   chmod +x ~/ns-3-dev/run_group_f_full_grid.sh
#   cd ~/ns-3-dev
#   bash run_group_f_full_grid.sh           # parallel (default)
#   bash run_group_f_full_grid.sh --seq      # sequential
# =============================================================================

set -euo pipefail

NS3_DIR="${HOME}/ns-3-dev"
CC_IN_SCRATCH="${NS3_DIR}/scratch/nbiot-smp-benchmark.cc"
RESULTS_DIR="${NS3_DIR}/results"
CSV_FILE="${RESULTS_DIR}/results_group_f.csv"
LOG_FILE="${RESULTS_DIR}/sweep_group_f_debug.log"
SIM_TIME=120

MODE="parallel"
if [[ "${1:-}" == "--seq" ]]; then MODE="seq"; fi

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ -d "${NS3_DIR}" ]]       || error "NS-3 directory not found: ${NS3_DIR}"
[[ -f "${CC_IN_SCRATCH}" ]] || error "CC file not found: ${CC_IN_SCRATCH}
  → Fix: cp nbiot-smp-benchmark.cc ~/ns-3-dev/scratch/"

cd "${NS3_DIR}"
mkdir -p "${RESULTS_DIR}/figures"

if [[ ! -f "${NS3_DIR}/CMakeCache.txt" ]]; then
    ./ns3 configure --enable-examples 2>&1 | tail -5
fi

info "Building (skips if already up to date)..."
./ns3 build nbiot-smp-benchmark 2>&1 | grep -E "(error:|Built target)" || true

info "Starting Group F full-factorial sweep (504 configs, mode=${MODE})..."

python3 - <<PYEOF
import subprocess, os, csv, time, math
from concurrent.futures import ProcessPoolExecutor, as_completed

NS3_DIR     = "${NS3_DIR}"
CSV_FILE    = "${CSV_FILE}"
LOG_FILE    = "${LOG_FILE}"
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
# GROUP F — FULL FACTORIAL GRID (504 configs)
# T3412 × T3324 × interval × ceMode, all combinations
# ══════════════════════════════════════════════════════════════════════════════
T3412_VALS    = [int(round(300 * (86400/300) ** (i/6))) for i in range(7)]
T3324_VALS    = [5, 10, 15, 20, 25, 30]
INTERVAL_VALS = [300, 1800, 7200, 86400]
CE_VALS       = [0, 1, 2]

GROUP_F = []
idx = 0
for t3412 in T3412_VALS:
    for t3324 in T3324_VALS:
        for interval in INTERVAL_VALS:
            for ce in CE_VALS:
                idx += 1
                ce_name = ["CEA","CEB","CEC"][ce]
                label = f"F_{idx:03d}_T{t3412}_D{t3324}_I{interval}_{ce_name}"
                GROUP_F.append(("F", label, t3412, t3324, interval, ce))

print(f"[SWEEP] Group F: {len(GROUP_F)} configs "
      f"({len(T3412_VALS)}×{len(T3324_VALS)}×{len(INTERVAL_VALS)}×{len(CE_VALS)})")

# ══════════════════════════════════════════════════════════════════════════════
def run_one(cfg):
    group, label, T3412, T3324, interval, ce = cfg
    cmd = [
        f"{NS3_DIR}/ns3", "run", "nbiot-smp-benchmark", "--",
        f"--T3412={T3412}", f"--T3324={T3324}",
        f"--interval={interval}", f"--simTime={SIM_TIME}",
        f"--ceMode={ce}", f"--label={label}",
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
        print(f"  [FAIL] ({completed}/{len(GROUP_F)}) {label}: {error_msg}")
        return None
    for line in output.splitlines():
        row = parse_result(line.strip())
        if row:
            row["group"] = group
            if completed % 25 == 0 or completed == len(GROUP_F):
                print(f"  [OK]  ({completed}/{len(GROUP_F)}) {label}  "
                      f"L={row['lifetime_yr']:.2f}yr  PSF={row['PSF_pct']:.1f}%")
            return row
    print(f"  [WARN] {label}: no RESULT line found")
    return None

if PARALLEL:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, cfg): cfg for cfg in GROUP_F}
        for fut in as_completed(futures):
            cfg, output, err = fut.result()
            row = process(cfg, output, err)
            if row: rows.append(row)
else:
    for cfg in GROUP_F:
        cfg_out, output, err = run_one(cfg)
        row = process(cfg_out, output, err)
        if row: rows.append(row)

rows.sort(key=lambda r: (r.get("T3412",0), r.get("T3324",0),
                         r.get("interval",0), r.get("ceMode",0)))

with open(LOG_FILE, "w") as f:
    f.write("\\n".join(log_lines))

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})

elapsed = time.time() - t0
print(f"\\n[SWEEP] Done. {len(rows)}/{len(GROUP_F)} configs parsed.")
print(f"[SWEEP] Elapsed: {elapsed/60:.1f} min")
print(f"[SWEEP] CSV: {CSV_FILE}")
PYEOF

info "Group F sweep complete: ${CSV_FILE}"

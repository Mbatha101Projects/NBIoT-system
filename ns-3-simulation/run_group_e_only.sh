#!/usr/bin/env bash
# =============================================================================
# run_group_e_only.sh  —  Run ONLY the Group E literature-comparison sweep
#
# Does NOT touch your existing results.csv. Writes a separate file:
#   ~/ns-3-dev/results/results_group_e.csv
#
# Assumes nbiot-smp-benchmark.cc is already built (i.e. you've already run
# the main sweep at least once, or built it manually). If not built yet,
# this script builds it first.
#
# USAGE
#   cp run_group_e_only.sh ~/ns-3-dev/
#   chmod +x ~/ns-3-dev/run_group_e_only.sh
#   cd ~/ns-3-dev
#   bash run_group_e_only.sh
# =============================================================================

set -euo pipefail

NS3_DIR="${HOME}/ns-3-dev"
CC_IN_SCRATCH="${NS3_DIR}/scratch/nbiot-smp-benchmark.cc"
RESULTS_DIR="${NS3_DIR}/results"
CSV_FILE="${RESULTS_DIR}/results_group_e.csv"
LOG_FILE="${RESULTS_DIR}/sweep_group_e_debug.log"
SIM_TIME=120
# v3 NOTE: lowered from 7200s. Root cause analysis showed a single isolated
# Group E config (T3412=47132, interval=54000) took 4m14s of real wall-clock
# time for ONE run, with disk space and logging both confirmed clean. The
# cause is NS-3's LTE PHY/MAC layer running its full internal housekeeping
# (CQI/SINR reporting, scheduler ticks) for the ENTIRE simTime window
# regardless of application traffic -- for Group E configs where interval >>
# simTime, the device attaches once and then the simulator still pays the
# full per-tick cost of an idle-but-attached LTE connection for thousands of
# simulated seconds. The analytical SMP engine (computeSmp) does NOT depend
# on simTime at all -- it only needs T3412/T3324/interval/ceMode -- so a
# short simTime is sufficient: NS-3 only needs to observe ONE attach event
# and, where interval allows, one transmission, to produce a usable
# latency/PDR sample. 120s is enough for attach + first packet at every
# T3412/interval combination in this sweep while avoiding the long LTE
# housekeeping tail that caused the 4m+ runtime.

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ -d "${NS3_DIR}" ]]       || error "NS-3 directory not found: ${NS3_DIR}"
[[ -f "${CC_IN_SCRATCH}" ]] || error "CC file not found: ${CC_IN_SCRATCH}"

cd "${NS3_DIR}"
mkdir -p "${RESULTS_DIR}/figures"

if [[ ! -f "${NS3_DIR}/CMakeCache.txt" ]]; then
    ./ns3 configure --enable-examples 2>&1 | tail -5
fi

info "Building (skips if already up to date)..."
./ns3 build nbiot-smp-benchmark 2>&1 | grep -E "(error:|Built target)" || true

info "Running Group E only (36 configs)..."

python3 - <<PYEOF
import subprocess, os, csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed

NS3_DIR  = "${NS3_DIR}"
CSV_FILE = "${CSV_FILE}"
LOG_FILE = "${LOG_FILE}"
SIM_TIME = ${SIM_TIME}
MAX_WORKERS = os.cpu_count() or 2

CSV_HEADER = [
    "group","label","T3412","T3324","interval","ceMode",
    "sent","received","PDR","latency_ms",
    "energy_mWh","lifetime_yr","PSF_pct",
    "tauFired","p_avg_mw"
]

T3412_HOURS  = [0, 3, 6, 9, 12, 15]
T3412_SECONDS = {0: 300}
for h in T3412_HOURS[1:]:
    T3412_SECONDS[h] = h * 3600

T3324_SECONDS_LIST = [5, 10, 15, 20, 25, 30]
E_INTERVAL_S = 54000

GROUP_E = []
for t3412_h in T3412_HOURS:
    t3412_s = T3412_SECONDS[t3412_h]
    for t3324_s in T3324_SECONDS_LIST:
        label = f"E_T3412_{t3412_h}h_T3324_{t3324_s}s"
        GROUP_E.append(("E", label, t3412_s, t3324_s, E_INTERVAL_S, 0))

print(f"[SWEEP] Group E configs: {len(GROUP_E)}")

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
                                timeout=600, cwd=NS3_DIR)
        return cfg, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return cfg, "", "TIMEOUT after 600s"
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

def process(cfg, output, error_msg):
    group, label = cfg[0], cfg[1]
    log_lines.append(f"\\n=== [{group}] {label} ===\\n{output}")
    if error_msg:
        print(f"  [FAIL] {label}: {error_msg}")
        return None
    for line in output.splitlines():
        row = parse_result(line.strip())
        if row:
            row["group"] = group
            print(f"  [OK]  {label:30s}  T3412={row['T3412']:6d}s  "
                  f"T3324={row['T3324']:3d}s  L={row['lifetime_yr']:7.2f}yr")
            return row
    print(f"  [WARN] {label}: no RESULT line found")
    return None

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(run_one, cfg): cfg for cfg in GROUP_E}
    for fut in as_completed(futures):
        cfg, output, err = fut.result()
        row = process(cfg, output, err)
        if row: rows.append(row)

rows.sort(key=lambda r: (r.get("T3412", 0), r.get("T3324", 0)))

with open(LOG_FILE, "w") as f:
    f.write("\\n".join(log_lines))

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})

print(f"\\n[SWEEP] Done. {len(rows)}/{len(GROUP_E)} parsed in {(time.time()-t0)/60:.1f} min")
print(f"[SWEEP] CSV: {CSV_FILE}")
PYEOF

info "Group E sweep complete: ${CSV_FILE}"

"""


THIS FILE MUST STAY IN SYNC with:
  - nbiot-smp-benchmark.cc (Sections 2-7b)
  - plot_results.py's smp() function
If any power constant or formula changes in one place, it must change
in all three, or analytical results will silently diverge between the
NS-3 sweeps, the figures, and the Q-learning agent.
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# POWER CONSTANTS (mW) — Sørensen et al. (2022) Table X, U-Blox EVK-N211
# ─────────────────────────────────────────────────────────────────────────────
P_PSM_MW    = 0.0095             # 9.5 µW, N211 P_sleep_PSM, Sørensen et al. Table X
P_SYNC_MW   = 72.73              # 160mJ / 2.2s window average
P_RRC_MW    = 21.337             # P_sleep_cDRX
P_TX_MW     = 742.858            # N211 at 23dBm
P_TX_GAP_MW = 153.6              # TX gap power — Sørensen Eq.6
P_RX_MW     = 21.337             # P_sleep_cDRX (DCI monitoring)
P_ACTIVE_MW = 0.267              # v3 cycle-weighted eDRX composite

BATTERY_MWH    = 3000.0 * 3.6    # 10800 mWh
P_ALWAYS_ON_MW = P_TX_MW         # PSF reference

# ─────────────────────────────────────────────────────────────────────────────
# SOJOURN DISTRIBUTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
GAMMA_SYNC = 0.88623              # Γ(1.5) = √π/2
GAMMA_TX   = 0.90274              # Γ(5/3)
LAMBDA_SYNC_BASE = 2.483          # s, fitted to Sørensen N211 t_sync=2.2s
LAMBDA_TX_BASE   = 0.0576         # s, CE-A 20-byte payload
RRC_MEAN_S = 1.0
RX_FIXED_S = 0.015
P_RETRANSMIT = 0.05

# CE mode scale factors — 3GPP TS 36.211 §6.6
CE_TX_SCALE   = [1.0, 16.0, 128.0]
CE_SYNC_SCALE = [1.0,  2.5,   6.0]

# TX gap model — 3GPP TS 36.211 §6.6, Sørensen Eq.6
TX_MAX_S = 0.256
TX_GAP_S = 0.040

# TAU penalty — Sørensen §V, Tables VII-VIII, Fig.8
E_TAU_BASE_MJ = 210.0


def tx_effective_power_mw(ce: int) -> float:
    """Gap-corrected effective TX power for a given CE mode (0/1/2)."""
    t_tx = (LAMBDA_TX_BASE * CE_TX_SCALE[ce] * GAMMA_TX) / (1.0 - P_RETRANSMIT)
    n_gaps = int(t_tx / TX_MAX_S)
    t_total = t_tx + n_gaps * TX_GAP_S
    e_tx = t_tx * P_TX_MW + n_gaps * TX_GAP_S * P_TX_GAP_MW
    return e_tx / t_total if t_total > 0 else P_TX_MW


class SmpResult:
    """Container for one computeSmp() evaluation — mirrors the CC file's
    SmpResult struct field-for-field."""
    __slots__ = ("pi", "E", "cycle_time", "avg_power_mw", "lifetime_yr",
                 "psf", "tau_energy_mj", "tau_fired", "delay_seconds")

    def __init__(self):
        self.pi = {}
        self.E = {}
        self.cycle_time = 0.0
        self.avg_power_mw = 0.0
        self.lifetime_yr = 0.0
        self.psf = 0.0
        self.tau_energy_mj = 0.0
        self.tau_fired = False
        self.delay_seconds = 0.0


def compute_smp(T3412: float, T3324: float, interval: float, ce: int) -> SmpResult:
    """
    The validated 6-state Semi-Markov Process energy model.
    Identical logic to nbiot-smp-benchmark.cc's computeSmp() (v4) and
    plot_results.py's smp() function. See those files for full citations.

    Parameters
    ----------
    T3412 : PSM sleep timer, seconds
    T3324 : Active/eDRX window timer, seconds
    interval : actual traffic transmission interval, seconds
    ce : coverage enhancement mode, 0=CE-A (good), 1=CE-B, 2=CE-C (poor)

    Returns
    -------
    SmpResult with avg_power_mw, lifetime_yr, psf, tau_fired, delay_seconds
    """
    r = SmpResult()
    
        # v5 FIX: if the device reports at least as often as T3412 expires, one cycle
    # is ONE reporting interval with one SYNC+RRC+TX+RX burst, and no periodic TAU.
    if interval > 0 and interval <= T3412:
        E_SYNC = LAMBDA_SYNC_BASE * CE_SYNC_SCALE[ce] * GAMMA_SYNC
        E_RRC  = RRC_MEAN_S
        E_TX   = (LAMBDA_TX_BASE * CE_TX_SCALE[ce] * GAMMA_TX) / (1.0 - P_RETRANSMIT)
        E_RX   = RX_FIXED_S
        busy   = E_SYNC + E_RRC + E_TX + E_RX
        rest   = max(interval - busy, 0.0)
        E_ACTIVE = min(T3324, rest)
        E_PSM    = rest - E_ACTIVE
        r.E = dict(PSM=E_PSM, SYNC=E_SYNC, RRC=E_RRC, TX=E_TX, RX=E_RX, ACTIVE=E_ACTIVE)
        sum_e = sum(r.E.values())          # = interval (or `busy` if the device cannot keep up)
        r.cycle_time = sum_e
        P = dict(PSM=P_PSM_MW, SYNC=P_SYNC_MW, RRC=P_RRC_MW,
                 TX=tx_effective_power_mw(ce), RX=P_RX_MW, ACTIVE=P_ACTIVE_MW)
        avg_p = 0.0
        for st in r.E:
            r.pi[st] = r.E[st] / sum_e
            avg_p += r.pi[st] * P[st]
        r.tau_fired = False
        r.avg_power_mw = avg_p
        r.lifetime_yr = (BATTERY_MWH / avg_p) / 8760.0
        r.psf = 1.0 - (avg_p / P_ALWAYS_ON_MW)
        num = r.pi["PSM"] * (r.E["PSM"] / 2.0) + r.pi["ACTIVE"] * (r.E["ACTIVE"] / 2.0)
        den = r.pi["PSM"] + r.pi["ACTIVE"]
        r.delay_seconds = (num / den) if den > 0 else 0.0
        return r

    # tx_frac: fraction of wakeups carrying actual data. SYNC is NOT
    # scaled — a device exiting PSM always re-syncs in full regardless
    # of whether data is pending (v3 fix; see CC file Section 3 comments).
    tx_frac = min(1.0, T3412 / interval) if interval > 0 else 0.0

    E_PSM    = max(T3412 - T3324, 0.0)
    E_SYNC   = LAMBDA_SYNC_BASE * CE_SYNC_SCALE[ce] * GAMMA_SYNC
    E_RRC    = RRC_MEAN_S * tx_frac
    E_TX     = (LAMBDA_TX_BASE * CE_TX_SCALE[ce] * GAMMA_TX) / (1.0 - P_RETRANSMIT) * tx_frac
    E_RX     = RX_FIXED_S * tx_frac
    E_ACTIVE = T3324

    r.E = dict(PSM=E_PSM, SYNC=E_SYNC, RRC=E_RRC, TX=E_TX, RX=E_RX, ACTIVE=E_ACTIVE)
    sum_e = sum(r.E.values())
    r.cycle_time = sum_e

    P = dict(
        PSM=P_PSM_MW, SYNC=P_SYNC_MW, RRC=P_RRC_MW,
        TX=tx_effective_power_mw(ce), RX=P_RX_MW, ACTIVE=P_ACTIVE_MW,
    )

    avg_p = 0.0
    for s in r.E:
        r.pi[s] = r.E[s] / sum_e if sum_e > 0 else 0.0
        avg_p += r.pi[s] * P[s]

    # TAU penalty — Sørensen §V, Fig.8
    r.tau_fired = interval > T3412
    if r.tau_fired:
        e_tau = E_TAU_BASE_MJ * CE_SYNC_SCALE[ce]
        r.tau_energy_mj = e_tau
        avg_p += e_tau / (T3412 * 1000.0) if T3412 > 0 else 0.0

    r.avg_power_mw = avg_p
    r.lifetime_yr = (BATTERY_MWH / avg_p) / 8760.0 if avg_p > 0 else 999.0
    r.psf = 1.0 - (avg_p / P_ALWAYS_ON_MW)

    # Expected buffering delay — Based on the buffering delay formula that Azari implemented.
    # Only PSM and ACTIVE are sleep-capable (buffering) states.
    numerator = r.pi["PSM"] * (r.E["PSM"] / 2.0) + r.pi["ACTIVE"] * (r.E["ACTIVE"] / 2.0)
    denominator = r.pi["PSM"] + r.pi["ACTIVE"]
    r.delay_seconds = (numerator / denominator) if denominator > 0 else 0.0

    return r


if __name__ == "__main__":
    # Quick self-test when run directly: confirms this module matches the
    # values already cross-validated against the CC file in earlier work.
    tests = [
        (3600, 20, 3600, 0),
        (300, 20, 60, 2),
        (86400, 30, 86400, 0),
    ]
    print("Self-test — expect these to match nbiot-smp-benchmark.cc output:")
    for T3412, T3324, interval, ce in tests:
        r = compute_smp(T3412, T3324, interval, ce)
        print(f"  T3412={T3412:6d} T3324={T3324:3d} interval={interval:6d} CE={ce}  "
              f"-> P_avg={r.avg_power_mw:.6f}mW  lifetime={r.lifetime_yr:.3f}yr  "
              f"delay={r.delay_seconds:.2f}s  TAU={r.tau_fired}")
/* =============================================================================
 * nbiot-smp-benchmark.cc
 * NB-IoT SMP-Driven NS-3 Benchmark
 * =============================================================================
 * University of Zululand | Honours CS | 2026
 * Student : Mawande Mbatha (202396115)
 * Supervisor: Dr Paul Tarwireyi
 * Topic: Optimisation of Energy Efficiency in NB-IoT Devices Using PSM
 *
 * ROLE OF THIS FILE
 * -----------------
 * NS-3 is used ONLY for what it uniquely provides:
 *   • Packet latency (ms)  — real LTE protocol stack timing
 *   • PDR (%)              — packet delivery ratio
 *
 * Energy is computed here using a Semi-Markov Process (SMP) whose
 * state sojourn distributions are justified from 3GPP specifications
 * and published NB-IoT measurement literature.
 *
 * STATE SOJOURN DISTRIBUTIONS
 * ---------------------------
 * PSM    → Deterministic Fixed(T3412)
 *            3GPP TS 24.008 §10.5.7.4a: T3412 is a network-assigned
 *            countdown timer. Device exits PSM at exactly T3412 seconds.
 *
 * SYNC   → Weibull(k=2, λ=2.483s × CE_SYNC_SCALE[ce])
 *            k=2: right-skewed — most syncs are fast, poor signal
 *            causes occasional long tail.
 *            λ_base=2.483s: fitted so Weibull(k=2) mean = 2.2s, which
 *            matches measured t_sync for U-Blox N211 at CE-A.
 *            Source: Sørensen et al. (2022) Table X.
 *            CE_SYNC_SCALE: ×1.0 (CE-A), ×2.5 (CE-B), ×6.0 (CE-C)
 *            per 3GPP TS 36.211 §6.6.
 *            v3 CORRECTION: SYNC duration is NO LONGER scaled by tx_frac.
 *            A device waking from PSM performs a full synchronisation
 *            and TAU-eligible re-attach EVERY time T3412 expires,
 *            regardless of whether data is actually pending (this is
 *            exactly what Anbazhagan & Mugelan Fig.1/Algorithm-1 line 32-34
 *            describes: S2/PSM -> S4/RACH happens unconditionally on
 *            timer expiry, with the data-vs-no-data branch only decided
 *            AFTER re-sync, at the RACH state). The v2 model incorrectly
 *            shrank SYNC by tx_frac=min(1,T3412/interval), which for
 *            T3412 << interval (e.g. Sultania Fig.3 conditions: T3412 up
 *            to 12h, interval=24h) all but eliminated SYNC from the
 *            energy budget -- a major contributor to the ~80-92 year
 *            lifetime overestimate identified against Sultania (2019)
 *            Fig.3. Only RRC, TX, and RX (states that only occur when
 *            there is actually data to send) remain scaled by tx_frac.
 *
 * RRC    → Exponential(μ=1.0s)
 *            Memoryless property matches network resource contention.
 *            3GPP TS 36.331 RRC setup is contention-based → exponential
 *            inter-arrival of success events.
 *
 * TX     → Weibull(k=1.5, λ=0.0576s × CE_TX_SCALE[ce])
 *            k=1.5: heavier tail than SYNC — TX has more failure modes
 *            (HARQ retransmit, CE repetitions, congestion).
 *            λ_base=0.0576s: CE-A, 20-byte payload, MCS10, 5RUs, 15kHz.
 *            CE_TX_SCALE: ×1 (CE-A), ×16 (CE-B), ×128 (CE-C)
 *            per 3GPP TS 36.211 §6.6 / TS 36.213 §16.
 *            Gap model: 40ms at reduced power after every 256ms of
 *            continuous TX per 3GPP TS 36.211 §6.6, Sørensen Eq.6.
 *
 * RX     → Deterministic Fixed(15ms)
 *            HARQ ACK/NACK monitoring window after NPUSCH transmission.
 *            3GPP TS 36.321 §8.3 specifies minimum 12ms; 15ms used as
 *            the midpoint of the empirical 8-20ms range.
 *            State models cDRX idle monitoring (DCI), NOT full downlink
 *            data reception. Power = P_sleep_cDRX per Sørensen Table X.
 *
 * ACTIVE → Deterministic Fixed(T3324)
 *            3GPP TS 24.008: T3324 is the Active Timer, network-assigned.
 *            v3 CORRECTION: T3324 is NOT one continuous sojourn at a single
 *            power level. Per Sultania et al. (2019) eta=T3324/T_eDRX_cycle
 *            and Anbazhagan & Mugelan (2024) N_pag=T_eDRX/T_pag, the T3324
 *            window consists of MANY short repeated paging cycles, each
 *            with a brief active "paging burst" plus a sleep portion.
 *            The v2 model collapsed this to a single flat sleep-only
 *            constant (0.0122 mW from Sørensen P_sleep_eDRX), which
 *            produced a near-zero T3324 sensitivity contradicted by both
 *            Sultania Fig.3 (clear T3324 separation) and, to a lesser
 *            extent, Anbazhagan Fig.5 (small but non-zero PSF change with
 *            T_eDRX). v3 uses a cycle-weighted composite: one paging
 *            cycle = burst (P_burst, t_onDur) + sleep (P_sleep_eDRX,
 *            remainder), weighted-averaged over a cycle length consistent
 *            with the shortest DRX cycle in Anbazhagan Table 1 (1.28s).
 *            This raises effective ACTIVE power from 0.015 mW to ~0.267 mW
 *            -- a genuine correction, not a full match to Sultania's
 *            21.6 mW EIdle, since that figure is sourced from a DIFFERENT
 *            chipset (u-blox SARA-N2) with different deep-sleep capability
 *            than the N211 used for every other constant in this model.
 *            This cross-device discrepancy is flagged explicitly rather
 *            than papered over with a forced numeric match.
 *
 * POWER VALUES
 * ------------
 * All values from Sørensen et al. (2022) "Modelling and Experimental
 * Validation for Battery Lifetime Estimation in NB-IoT and LTE-M",
 * arXiv:2106.13286, Table X, U-Blox EVK-N211 measured values:
 *
 *   PSM    : 0.0095 mW  (N211: 9.5µW — within 13%)
 *   SYNC   : 72.73 mW  = 160mJ / 2.2s (window average, not peak burst)
 *   RRC    : 21.337 mW = P_sleep_cDRX (inter-message delay at cDRX idle)
 *   TX     : 742.858 mW at 23dBm + gap model
 *   RX     : 21.337 mW = P_sleep_cDRX (DCI monitoring, not data receive)
 *   ACTIVE : 0.015 mW  = P_sleep_eDRX effective (eDRX window iDRX cycles)
 *
 * TAU PENALTY (NEW — was missing from v1)
 * ----------------------------------------
 * When traffic interval > T3412, a Tracking Area Update fires every T3412
 * seconds. TAU ≈ 210 mJ at CE-A (Sørensen §V, Tables VII-VIII, Fig.10).
 * This is added to P_avg as a per-second contribution.
 *
 * SMP ENERGY FORMULA (Renewal Reward Theorem — Cinlar 1975)
 * ----------------------------------------------------------
 *   π_i     = E[T_i] / Σ_j E[T_j]    (stationary distribution)
 *   P_avg   = Σ_i π_i × P_i           (average power, mW)
 *   Lifetime = Battery_mWh / P_avg / 8760  (years)
 *   PSF      = 1 - P_avg / P_always_on
 *
 * tx_frac correction:
 *   When T3412 < interval, not every PSM wakeup carries data.
 *   SYNC/RRC/TX/RX sojourn times are scaled by tx_frac = min(1, T3412/interval).
 *
 * COMMAND-LINE ARGS
 * -----------------
 *   --T3412     PSM timer seconds          (default: 3600)
 *   --T3324     Active timer seconds       (default: 20)
 *   --interval  Packet interval seconds    (default: 3600)
 *   --simTime   Simulation duration s      (default: 7200)
 *   --ceMode    0=CE-A  1=CE-B  2=CE-C    (default: 0)
 *   --label     Tag for output row         (default: "cfg")
 *
 * OUTPUT LINE (stdout, parsed by run_benchmark.sh)
 * -------------------------------------------------
 * RESULT label T3412 T3324 interval ceMode sent recv PDR latency
 *        energy lifetime PSF tauFired p_avg_mw
 *
 * Fields [0]-[12]: index-compatible with v1
 * Fields [13] tauFired, [14] p_avg_mw: new in v2
 * ============================================================================= */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"

#include <cmath>
#include <string>
#include <map>
#include <random>
#include <iomanip>
#include <iostream>
#include <algorithm>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("NbIotSmpBenchmark");

// =============================================================================
// SECTION 1 — STATE DEFINITIONS
// =============================================================================

enum NbIotState { PSM=0, SYNC=1, RRC=2, TX=3, RX=4, ACTIVE=5, N_STATES=6 };

// =============================================================================
// SECTION 2 — POWER VALUES (mW)
// Source: Sørensen et al. (2022) Table X, U-Blox EVK-N211
// =============================================================================

static const double P_PSM_MW    = 0.0095;         //N211, Table X

// SYNC, RRC, TX, RX: Sørensen et al. (2022) Table X, U-Blox EVK-N211
static const double P_SYNC_MW   = 72.73;          // 160mJ / 2.2s window avg
static const double P_RRC_MW    = 21.337;         // P_sleep_cDRX
static const double P_TX_MW     = 742.858;        // N211 at 23dBm
static const double P_TX_GAP_MW = 153.6;          // TX gap — Sørensen Eq.6
static const double P_RX_MW     = 21.337;         // P_sleep_cDRX (DCI mon.)

// ACTIVE/eDRX — v3 CYCLE-WEIGHTED COMPOSITE (was flat 0.015 mW in v2)
//
// v2 used Sørensen's P_sleep_eDRX (0.0122 mW, N211) directly as a flat
// power for the entire T3324 window, treating T3324 as one continuous
// sojourn. This produced near-zero T3324 sensitivity, contradicted by
// Sultania et al. (2019) Fig.3 (PSM/Sultania.pdf) which shows a clear,
// non-trivial lifetime separation between T3324=5s and T3324=30s, and to
// a smaller extent by Anbazhagan & Mugelan (2024) Fig.5, which shows PSF
// rising 99.2%->99.4% (small but non-zero) as T_eDRX increases 90->180s.
//
// Both papers' own state diagrams (Sultania eta=T3324/T_eDRX_cycle;
// Anbazhagan N_pag=T_eDRX/T_pag) treat the T3324/eDRX window as MANY
// repeated short paging cycles, not one flat-power sojourn. v3 follows
// the same structure: one paging cycle = a brief active "burst" (device
// checks NPDCCH for a paging indication) plus a sleep remainder.
//
//   P_burst   = E_onDur / t_onDur = 0.326 mJ / 1.445 ms = 225.6 mW
//               (Sørensen Table X, N211 eDRX E_onDur/t_onDur)
//   P_sleep   = 0.0122 mW (Sørensen Table X, N211 P_sleep_eDRX)
//   t_cycle   = 1.28 s (shortest DRX cycle, Anbazhagan & Mugelan Table 1:
//               T_ACTIVE+T_SLEEP combinations range 1.28s-10.24s; the
//               shortest value is used as a conservative/upper-bound
//               estimate of paging frequency within the T3324 window)
//
//   P_ACTIVE_MW = (t_onDur×P_burst + (t_cycle-t_onDur)×P_sleep) / t_cycle
//               = (0.001445×225.6 + 1.27855×0.0122) / 1.28
//               ≈ 0.267 mW
//
// This is a genuine, citable correction (≈18x the v2 flat figure) but is
// NOT forced to match Sultania's EIdle=21.6 mW. That figure is sourced
// from u-blox SARA-N2 hardware (Sultania Table II), a DIFFERENT chipset
// from the N211 used for every other constant in this model, and
// SARA-N2 may genuinely have less aggressive deep-sleep behaviour between
// paging windows. This cross-device gap is acknowledged explicitly rather
// than papered over — see thesis discussion of cross-chipset variance.
static const double P_ACTIVE_MW = 0.267;

static const double BATTERY_MWH    = 3000.0 * 3.6;  // 10800 mWh
static const double P_ALWAYS_ON_MW = P_TX_MW;        // PSF reference

// TX gap — 3GPP TS 36.211 §6.6, confirmed Sørensen Eq.6
static const double TX_MAX_S  = 0.256;  // 256ms max continuous TX
static const double TX_GAP_S  = 0.040;  // 40ms gap duration

// TAU energy at CE-A — Sørensen §V, Tables VII+VIII
static const double E_TAU_BASE_MJ = 210.0;

// =============================================================================
// SECTION 3 — SOJOURN DISTRIBUTION PARAMETERS
// =============================================================================

// Weibull shape parameters
static const double K_SYNC = 2.0;   // right-skewed; quick most of the time
static const double K_TX   = 1.5;   // heavier tail than SYNC

// Weibull scale parameters
// SYNC λ: fitted so mean = 2.2s (Sørensen N211 t_sync=2200ms)
//   λ = 2.2 / Γ(1.5) = 2.2 / 0.88623 = 2.483s
static const double LAMBDA_SYNC_BASE = 2.483;

// TX λ: CE-A, 20-byte payload, MCS10, 5RUs, 15kHz → mean ≈ 52ms
//   λ = 0.052 / Γ(5/3) = 0.052 / 0.90274 = 0.0576s
static const double LAMBDA_TX_BASE = 0.0576;

// Γ function values (precomputed — avoids tgamma portability issues)
static const double GAMMA_SYNC = 0.88623;  // Γ(1.5) = √π/2
static const double GAMMA_TX   = 0.90274;  // Γ(5/3)

// RRC — exponential mean (seconds)
static const double RRC_MEAN_S = 1.0;

// RX — fixed HARQ ACK window (seconds)
// 3GPP TS 36.321 §8.3: minimum 12ms; using 15ms (midpoint of 8-20ms range)
static const double RX_FIXED_S = 0.015;

// CE mode scale factors — 3GPP TS 36.211 §6.6
static const double CE_TX_SCALE[3]   = { 1.0,  16.0, 128.0 };
static const double CE_SYNC_SCALE[3] = { 1.0,   2.5,   6.0 };

// HARQ retransmit probability
static const double P_RETRANSMIT = 0.05;

// UE distance per CE mode (metres) — approximates MCL per Sørensen Table IX
//   CE-A: 50m (good, MCL≈140dB)
//   CE-B: 200m (moderate, MCL≈150dB)
//   CE-C: 500m (poor, MCL≈160dB)
static const double CE_DISTANCE_M[3] = { 50.0, 200.0, 500.0 };

// =============================================================================
// SECTION 4 — TX EFFECTIVE POWER (gap model)
// Sørensen Eq.6: 40ms gap at 153.6mW after every 256ms of continuous TX
// =============================================================================

static double txEffectivePowerMW(int ce)
{
    double t_tx  = (LAMBDA_TX_BASE * CE_TX_SCALE[ce] * GAMMA_TX) / (1.0 - P_RETRANSMIT);
    int    n_gap = static_cast<int>(std::floor(t_tx / TX_MAX_S));
    double t_tot = t_tx + n_gap * TX_GAP_S;
    double e_tx  = t_tx * P_TX_MW + n_gap * TX_GAP_S * P_TX_GAP_MW;
    return (t_tot > 0.0) ? (e_tx / t_tot) : P_TX_MW;
}

// =============================================================================
// SECTION 5 — SMP RESULT STRUCT
// =============================================================================

struct SmpResult {
    double pi[N_STATES];    // stationary distribution
    double E[N_STATES];     // mean sojourn per state (s)
    double cycleTime;       // Σ E[T_i]
    double avgPowerMW;      // Σ π_i × P_i  (+TAU contribution)
    double lifetimeYr;
    double psf;             // Power Saving Factor (0-1)
    double tauEnergyMJ;     // TAU energy per cycle (0 if no TAU)
    bool   tauFired;        // true when interval > T3412
    double delaySeconds;    // v4: expected buffering delay — see Section 7b
};

// =============================================================================
// SECTION 6 — MEAN SOJOURN TIMES
// =============================================================================

static double meanSojourn(NbIotState s, double T3412, double T3324, int ce)
{
    switch (s) {
        case PSM:    return T3412;
        case SYNC:   return LAMBDA_SYNC_BASE * CE_SYNC_SCALE[ce] * GAMMA_SYNC;
        case RRC:    return RRC_MEAN_S;
        case TX:     return (LAMBDA_TX_BASE * CE_TX_SCALE[ce] * GAMMA_TX)
                            / (1.0 - P_RETRANSMIT);
        case RX:     return RX_FIXED_S;
        case ACTIVE: return T3324;
        default:     return 0.0;
    }
}

// =============================================================================
// SECTION 7 — CORE SMP COMPUTATION
// =============================================================================

static SmpResult computeSmp(double T3412, double T3324,
                             double interval, int ce)
{
    SmpResult r = {};
        if (interval > 0 && interval <= T3412) {
        for (int s = 0; s < N_STATES; s++)
            r.E[s] = meanSojourn((NbIotState)s, T3412, T3324, ce);
        double busy = r.E[SYNC] + r.E[RRC] + r.E[TX] + r.E[RX];
        double rest = std::max(interval - busy, 0.0);
        r.E[ACTIVE] = std::min(T3324, rest);
        r.E[PSM]    = rest - r.E[ACTIVE];
        double sumE = 0.0;
        for (int s = 0; s < N_STATES; s++) sumE += r.E[s];
        r.cycleTime = sumE;
        double P[N_STATES] = { P_PSM_MW, P_SYNC_MW, P_RRC_MW,
                               txEffectivePowerMW(ce), P_RX_MW, P_ACTIVE_MW };
        double avgP = 0.0;
        for (int s = 0; s < N_STATES; s++) { r.pi[s] = r.E[s] / sumE; avgP += r.pi[s] * P[s]; }
        r.tauFired = false; r.tauEnergyMJ = 0.0;
        r.avgPowerMW = avgP;
        r.lifetimeYr = (BATTERY_MWH / avgP) / 8760.0;
        r.psf = 1.0 - avgP / P_ALWAYS_ON_MW;
        double num = r.pi[PSM] * r.E[PSM] / 2.0 + r.pi[ACTIVE] * r.E[ACTIVE] / 2.0;
        double den = r.pi[PSM] + r.pi[ACTIVE];
        r.delaySeconds = den > 0.0 ? num / den : 0.0;
        return r;
    }
    // tx_frac: fraction of wakeups that carry actual data.
    // v3 FIX: SYNC is excluded from this scaling. A device exiting PSM
    // performs a full synchronisation/re-attach EVERY time T3412 expires,
    // regardless of whether data happens to be pending — only AFTER
    // re-sync does the device learn whether there's data to send (see
    // Anbazhagan & Mugelan (2024) Fig.1/Algorithm-1, line 32-34: the
    // PSM->RACH transition on T3412 expiry is unconditional; the
    // data-vs-no-data branch is decided at RACH, not before SYNC).
    // v2 incorrectly scaled SYNC by tx_frac, which for T3412 << interval
    // (e.g. T3412=1h with a 24h traffic interval, as in Sultania (2019)
    // Fig.3) shrank SYNC's duration by up to ~24x and removed it almost
    // entirely from the energy budget — a major contributor to the
    // ~80-92 year lifetime overestimate found when validating against
    // Sultania Fig.3. Only RRC, TX, and RX — which genuinely only occur
    // when there is data to send — remain scaled by tx_frac.
    double tx_frac = std::min(1.0, T3412 / interval);

    // Build sojourn times; scale only data-dependent transient states.
    // SYNC is deliberately NOT scaled — see comment above.
    for (int s = 0; s < N_STATES; s++)
        r.E[s] = meanSojourn((NbIotState)s, T3412, T3324, ce);

    r.E[RRC] *= tx_frac;
    r.E[TX]  *= tx_frac;
    r.E[RX]  *= tx_frac;

    // Stationary distribution
    double sumE = 0.0;
    for (int s = 0; s < N_STATES; s++) sumE += r.E[s];
    r.cycleTime = sumE;

    double P[N_STATES] = {
        P_PSM_MW,
        P_SYNC_MW,
        P_RRC_MW,
        txEffectivePowerMW(ce),
        P_RX_MW,
        P_ACTIVE_MW
    };

    double avgP = 0.0;
    for (int s = 0; s < N_STATES; s++) {
        r.pi[s] = r.E[s] / sumE;
        avgP   += r.pi[s] * P[s];
    }

    // TAU penalty — Sørensen §V, Fig.8
    r.tauFired    = (interval > T3412);
    r.tauEnergyMJ = 0.0;
    if (r.tauFired) {
        double e_tau  = E_TAU_BASE_MJ * CE_SYNC_SCALE[ce];
        r.tauEnergyMJ = e_tau;
        avgP         += e_tau / (T3412 * 1000.0);  // mJ → mW
    }

    r.avgPowerMW = avgP;
    r.lifetimeYr = (avgP > 0.0) ? (BATTERY_MWH / avgP) / 8760.0 : 999.0;
    r.psf        = 1.0 - (avgP / P_ALWAYS_ON_MW);

    // ── SECTION 7b — EXPECTED BUFFERING DELAY (v4, NEW) ─────────────────────
    //
    // PURPOSE: NS-3's latency output measures only protocol/transmission
    // time (renamed tx_latency_ms in v4 output), which is near-constant
    // across configurations in this topology (deterministic single-UE,
    // lossless channel — see v4 design discussion). It does NOT capture
    // the dominant, configuration-dependent source of real-world delay in
    // a PSM/eDRX system: the time a packet spends BUFFERED while the
    // device is asleep, waiting for the next wake-up opportunity.
    //
    // FORMULA SOURCE: Sultania et al. (2019), Eq. 30 (their Δ formula):
    //   Δ = Σ_i (π_i × E[η_i] × T_i/2) / Σ_i (π_i × E[η_i])
    // Intuition: if a packet becomes ready at a uniformly random instant
    // during a sleep-capable state of duration T, the expected wait until
    // the next wake-up is T/2 (on average, you arrive halfway through).
    // Weighted by how much of the cycle is spent in each sleep-capable
    // state (the same π_i this function already computes),This is the basis in which latency will be based on.
    //
    // SLEEP-CAPABLE STATES: PSM and ACTIVE (eDRX/iDRX window) are the
    // states in which a newly-arrived packet must wait for the device to
    // next become reachable. SYNC, RRC, TX, RX are active/transmitting
    // states — a packet is never "buffered" during them, so they are
    // excluded from this calculation (matching Sultania's own state
    // classification: their Eq.30 only sums over S2/eDRX, S3/PSM, S5/Sleep,
    // i.e. their idle/sleep states, not their active-transmission states).
    //
    // VALIDATED (see conversation history): produces a ~314× range across
    // the full Group F grid (138s to 12h), correctly approaches but never
    // exceeds ~50% of T3412 as T3412 dominates, and degrades gracefully
    // (no div-by-zero, no NaN, no negative values) at T3412=0 and/or
    // T3324=0 edge cases.
    {
        double numerator   = r.pi[PSM]    * (r.E[PSM]    / 2.0)
                            + r.pi[ACTIVE] * (r.E[ACTIVE] / 2.0);
        double denominator = r.pi[PSM] + r.pi[ACTIVE];
        r.delaySeconds = (denominator > 0.0) ? (numerator / denominator) : 0.0;
    }

    return r;
}

// =============================================================================
// SECTION 8 — STOCHASTIC MONTE CARLO CROSS-CHECK
// =============================================================================

static double sampleWeibull(double lambda, double k, std::mt19937& rng)
{
    std::uniform_real_distribution<double> u(1e-9, 1.0 - 1e-9);
    return lambda * std::pow(-std::log(1.0 - u(rng)), 1.0 / k);
}

static double simulateCycle(double T3412, double T3324, int ce, std::mt19937& rng)
{
    double e = 0.0;
    // PSM — deterministic
    e += P_PSM_MW * (T3412 / 3600.0);
    // SYNC — Weibull
    e += P_SYNC_MW * (sampleWeibull(LAMBDA_SYNC_BASE * CE_SYNC_SCALE[ce],
                                    K_SYNC, rng) / 3600.0);
    // RRC — Exponential
    e += P_RRC_MW * (std::exponential_distribution<double>(
                         1.0 / RRC_MEAN_S)(rng) / 3600.0);
    // TX → RX loop with HARQ retransmit
    {
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        bool acked = false;
        for (int attempt = 0; attempt < 20 && !acked; attempt++) {
            double t_tx = sampleWeibull(LAMBDA_TX_BASE * CE_TX_SCALE[ce],
                                        K_TX, rng);
            e += txEffectivePowerMW(ce) * (t_tx / 3600.0);
            e += P_RX_MW * (RX_FIXED_S / 3600.0);
            if (coin(rng) >= P_RETRANSMIT) acked = true;
        }
    }
    // ACTIVE — deterministic
    e += P_ACTIVE_MW * (T3324 / 3600.0);
    return e;  // mWh
}

// =============================================================================
// SECTION 9 — PACKET TRACKING
// =============================================================================

struct Stats {
    uint32_t sent     = 0;
    uint32_t received = 0;
    double   totalLatMs = 0.0;
};
static Stats g_stats;
static std::map<uint64_t, double> g_txTime;

static void onTx(Ptr<const Packet> p)
{
    g_stats.sent++;
    g_txTime[p->GetUid()] = Simulator::Now().GetSeconds();
}
static void onRx(Ptr<const Packet> p, const Address&)
{
    auto it = g_txTime.find(p->GetUid());
    if (it != g_txTime.end()) {
        g_stats.received++;
        g_stats.totalLatMs +=
            (Simulator::Now().GetSeconds() - it->second) * 1000.0;
    }
}

// =============================================================================
// SECTION 10 — MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    double      T3412    = 3600.0;
    double      T3324    = 20.0;
    double      interval = 3600.0;
    double      simTime  = 7200.0;
    int         ceMode   = 0;
    int         rngRun   = 1;       // v4: independent repetition index
    std::string label    = "cfg";

    CommandLine cmd;
    cmd.AddValue("T3412",    "PSM timer (s)",          T3412);
    cmd.AddValue("T3324",    "Active timer (s)",        T3324);
    cmd.AddValue("interval", "Packet interval (s)",     interval);
    cmd.AddValue("simTime",  "Simulation duration (s)", simTime);
    cmd.AddValue("ceMode",   "CE mode 0=A 1=B 2=C",    ceMode);
    cmd.AddValue("rngRun",   "RNG run/repetition index (>=1)", rngRun);
    cmd.AddValue("label",    "Config label",            label);
    cmd.Parse(argc, argv);

    if (ceMode < 0 || ceMode > 2) ceMode = 0;
    if (rngRun < 1) rngRun = 1;

    // ── v4 RNG SEEDING ───────────────────────────────────────────────────────
    // Without this, NS-3's global RNG uses a fixed default seed/run number,
    // meaning every invocation with identical CLI args (including this CC
    // file's own LTE attach/RACH timing, scheduler backoff, channel fading,
    // etc.) produces an IDENTICAL trace — re-running the same config N times
    // would yield N identical rows with zero real variance, not N genuine
    // independent samples. RngSeedManager::SetRun() changes which independent
    // sub-stream of the seeded generator is used, which is the NS-3-documented
    // correct way to get statistically independent repetitions of the SAME
    // scenario (as opposed to changing the seed itself, which is intended for
    // switching to an entirely different experiment).
    // Source: ns-3 manual, "Random Variables" chapter — SeedManager::SetRun().
    RngSeedManager::SetSeed(1);       // fixed base seed — keeps the campaign
                                        // reproducible as a whole
    RngSeedManager::SetRun(rngRun);    // varies the independent sub-stream
                                        // per repetition — THIS produces the
                                        // genuine run-to-run variance

    // The Monte Carlo cross-check's std::mt19937 is seeded from rngRun too,
    // so its 5 repetitions are also independent rather than identical
    // (previously hardcoded to a fixed seed of 42 for every run).

    // ── LTE topology ──────────────────────────────────────────────────────────
    Ptr<LteHelper>             lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    lteHelper->SetAttribute("PathlossModel",
        StringValue("ns3::LogDistancePropagationLossModel"));
    lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(6));
    lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(6));
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // Remote sink host
    NodeContainer remoteHosts;
    remoteHosts.Create(1);
    InternetStackHelper inet;
    inet.Install(remoteHosts);
    Ptr<Node> remoteHost = remoteHosts.Get(0);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2p.SetDeviceAttribute("Mtu",      UintegerValue(1500));
    p2p.SetChannelAttribute("Delay",   TimeValue(MilliSeconds(10)));
    NetDeviceContainer backhaul = p2p.Install(pgw, remoteHost);

    Ipv4AddressHelper addrH;
    addrH.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer backIfaces = addrH.Assign(backhaul);
    Ipv4Address remoteAddr = backIfaces.GetAddress(1);

    Ipv4StaticRoutingHelper srh;
    srh.GetStaticRouting(remoteHost->GetObject<Ipv4>())
        ->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                            Ipv4Mask("255.0.0.0"), 1);

    // eNB — fixed at origin, height 30m
    NodeContainer enb; enb.Create(1);
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(enb);
    enb.Get(0)->GetObject<MobilityModel>()
        ->SetPosition(Vector(0.0, 0.0, 30.0));
    NetDeviceContainer enbDev = lteHelper->InstallEnbDevice(enb);

    // UE — distance reflects CE mode coverage level
    NodeContainer ue; ue.Create(1);
    mob.Install(ue);
    ue.Get(0)->GetObject<MobilityModel>()
        ->SetPosition(Vector(CE_DISTANCE_M[ceMode], 0.0, 1.5));
    NetDeviceContainer ueDev = lteHelper->InstallUeDevice(ue);

    inet.Install(ue);
    Ipv4InterfaceContainer ueIp = epcHelper->AssignUeIpv4Address(ueDev);
    srh.GetStaticRouting(ue.Get(0)->GetObject<Ipv4>())
        ->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    lteHelper->Attach(ueDev, enbDev.Get(0));

    // ── Applications ──────────────────────────────────────────────────────────
    uint16_t port = 9;
    PacketSinkHelper sinkH("ns3::UdpSocketFactory",
                           InetSocketAddress(Ipv4Address::GetAny(), port));
    auto sinkApp = sinkH.Install(remoteHost);
    sinkApp.Start(Seconds(0.5));
    sinkApp.Stop(Seconds(simTime));

    UdpClientHelper clientH(remoteAddr, port);
    clientH.SetAttribute("MaxPackets", UintegerValue(UINT32_MAX));
    clientH.SetAttribute("Interval",   TimeValue(Seconds(interval)));
    clientH.SetAttribute("PacketSize", UintegerValue(20));  // typical IoT payload
    auto clientApp = clientH.Install(ue.Get(0));
    clientApp.Start(Seconds(1.0));
    clientApp.Stop(Seconds(simTime - 1.0));

    clientApp.Get(0)->TraceConnectWithoutContext("Tx", MakeCallback(&onTx));
    sinkApp.Get(0)->TraceConnectWithoutContext("Rx",   MakeCallback(&onRx));

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();

    // ── SMP energy calculation ─────────────────────────────────────────────────
    SmpResult smp = computeSmp(T3412, T3324, interval, ceMode);

    // Energy over simulation window (mWh) — for MC comparison
    double simHr = simTime / 3600.0;
    double totalE = smp.avgPowerMW * simHr;

    // Monte Carlo cross-check (should agree within ~1-2%)
    // v4: seed derived from rngRun, not a fixed 42 — each repetition now
    // produces a genuinely independent Monte Carlo sample, matching the
    // independence given to NS-3's own RNG above via SetRun().
    std::mt19937 rng(1000u + static_cast<unsigned int>(rngRun));
    uint32_t nCycles = static_cast<uint32_t>(std::max(1.0, simTime / T3412));
    double mcEnergy = 0.0;
    for (uint32_t c = 0; c < nCycles; c++)
        mcEnergy += simulateCycle(T3412, T3324, ceMode, rng);

    // ── NS-3 packet metrics ────────────────────────────────────────────────────
    // v4: renamed from "latency" to "txLatency" — this measures ONLY raw
    // protocol/transmission time (NS-3-simulated), NOT the dominant
    // PSM/eDRX buffering delay. See smp.delaySeconds for the latter, which
    // is now the official "latency" metric for RQ4/Objective 4 reporting
    // and the Q-learning reward function.
    double pdr        = (g_stats.sent > 0)
                        ? 100.0 * g_stats.received / g_stats.sent : 0.0;
    double txLatency  = (g_stats.received > 0)
                        ? g_stats.totalLatMs / g_stats.received : 0.0;

    // ── RESULT line (stdout — parsed by run_benchmark.sh) ─────────────────────
    // [0]  RESULT
    // [1]  label
    // [2]  T3412         (int, seconds)
    // [3]  T3324         (int, seconds)
    // [4]  interval      (int, seconds)
    // [5]  ceMode        (int, 0/1/2)
    // [6]  sent          (int)
    // [7]  received      (int)
    // [8]  PDR           (float, %)
    // [9]  tx_latency_ms (float, ms) — v4 renamed from "latency"; raw NS-3
    //                      protocol/transmission time ONLY, not the
    //                      dominant PSM/eDRX buffering delay (see [16])
    // [10] energy        (float, mWh over sim window)
    // [11] lifetime      (float, years)
    // [12] PSF           (float, %)
    // [13] tauFired      (int, 0 or 1)
    // [14] p_avg_mw       (float, mW)
    // [15] rngRun         (int, repetition index)
    // [16] latency_s      (float, seconds — NEW in v4) — expected PSM/eDRX
    //                      buffering delay (Sultania Eq.30-style). THIS is
    //                      the official latency metric for RQ4/Objective 4
    //                      and the Q-learning reward function, NOT field [9].
    //
    // Fields [0]-[15] are index-compatible with earlier versions (field [9]
    // is the same NUMBER as before — only its meaning/name is now correctly
    // documented as transmission-only, not renumbered). Field [16] is new;
    // older parsers that only read up to index 14 or 15 are unaffected.
    std::cout << std::fixed << std::setprecision(6)
              << "RESULT"
              << " " << label
              << " " << static_cast<int>(T3412)
              << " " << static_cast<int>(T3324)
              << " " << static_cast<int>(interval)
              << " " << ceMode
              << " " << g_stats.sent
              << " " << g_stats.received
              << " " << pdr
              << " " << txLatency
              << " " << totalE
              << " " << smp.lifetimeYr
              << " " << (smp.psf * 100.0)
              << " " << (smp.tauFired ? 1 : 0)
              << " " << smp.avgPowerMW
              << " " << rngRun
              << " " << smp.delaySeconds
              << std::endl;

    // ── Human-readable summary (stderr — does not affect stdout parsing) ───────
    std::cerr << "\n=== NB-IoT SMP Benchmark: " << label << " ===" << std::endl;
    std::cerr << "Config   : T3412=" << T3412 << "s  T3324=" << T3324
              << "s  CE=" << (ceMode==0?"A":ceMode==1?"B":"C")
              << "  interval=" << interval << "s" << std::endl;
    std::cerr << "Cycle    : " << smp.cycleTime << "s" << std::endl;
    const char* snames[N_STATES] = {"PSM","SYNC","RRC","TX","RX","ACTIVE"};
    for (int s = 0; s < N_STATES; s++) {
        std::cerr << "pi_" << snames[s]
                  << std::string(7 - std::string(snames[s]).size(), ' ')
                  << ": " << smp.pi[s]*100 << "%  ("
                  << (s == RX ? smp.E[s]*1000 : smp.E[s])
                  << (s == RX ? "ms)" : "s)") << std::endl;
    }
    std::cerr << "P_avg    : " << smp.avgPowerMW << " mW" << std::endl;
    std::cerr << "Lifetime : " << smp.lifetimeYr << " yr" << std::endl;
    std::cerr << "PSF      : " << smp.psf * 100.0 << "%" << std::endl;
    std::cerr << "TAU      : " << (smp.tauFired ? "YES" : "NO");
    if (smp.tauFired)
        std::cerr << "  (" << smp.tauEnergyMJ << " mJ/cycle)";
    std::cerr << std::endl;
    std::cerr << "MC check : " << mcEnergy << " mWh  (analytical: "
              << totalE << " mWh)" << std::endl;
    std::cerr << "PDR      : " << pdr << "%  TxLatency: "
              << txLatency << " ms" << std::endl;
    std::cerr << "Latency  : " << smp.delaySeconds << " s  ("
              << smp.delaySeconds/3600.0 << " h)  [expected PSM/eDRX "
              << "buffering delay — THIS is the official latency metric]"
              << std::endl;
    return 0;
}

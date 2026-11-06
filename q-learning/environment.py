"""

PURPOSE
───────
Wraps the validated smp_engine.compute_smp() as an RL environment: an
agent observes a (CE mode, traffic bucket) state, chooses a (T3412,
T3324) action, and receives a reward combining lifetime and expected
buffering delay. This directly implements Objective 3 ("design and
implement an adaptive optimisation model... based on changing network
traffic and coverage conditions") and provides the mechanism RQ3 asks
about.



STATE SPACE

(ce_mode, traffic_bucket) — both discrete, both directly named in
Objective 3's wording ("traffic and coverage conditions").

  ce_mode        : 0=CE-A (good), 1=CE-B (medium), 2=CE-C (poor)
  traffic_bucket : 0=very frequent (<=15min), 1=frequent (15min-2h),
                   2=infrequent (2h-12h), 3=rare (>12h)




REWARD FUNCTION
───────────────
reward = w_lifetime * norm_lifetime
       - (w_delay * DELAY_WEIGHT_BY_BUCKET[traffic_bucket]) * norm_delay




"""

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

from smp_engine import compute_smp


# ─────────────────────────────────────────────────────────────────────────────
# STATE SPACE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

CE_MODES = [0, 1, 2]  # CE-A, CE-B, CE-C

# Traffic bucket boundaries (seconds), log-spaced rather than uniform-width
# -- gives fine resolution at realistic short intervals (where NB-IoT
# devices actually operate) without needing thousands of buckets to also
# cover the long end (up to 24h). First bucket deliberately starts at 0,
# not an arbitrary floor, so no interval value falls outside every bucket.
# Replaces the original 4-bucket scheme (superseded, kept in the project
# changelog for reference) -- tested and confirmed to resolve the original
# discretization-sensitivity concern (Point 4): intervals like 100s and
# 200s now genuinely land in different buckets, where they previously did
# not. Delay weight below updated to match (Point 6 finding).
TRAFFIC_BUCKETS = [
    (0,     124,   "extremely_frequent"),
    (124,   257,   "very_frequent"),
    (257,   532,   "frequent"),
    (532,   1100,  "somewhat_frequent"),
    (1100,  2277,  "moderate"),
    (2277,  4712,  "somewhat_infrequent"),
    (4712,  9750,  "infrequent"),
    (9750,  20176, "very_infrequent"),
    (20176, 41752, "rare"),
    (41752, 86400, "very_rare"),
]


def traffic_to_bucket(interval_s: float) -> int:
    """Map a raw traffic interval (seconds) to its discrete bucket index."""
    for i, (lo, hi, _name) in enumerate(TRAFFIC_BUCKETS):
        if lo <= interval_s < hi:
            return i
    return len(TRAFFIC_BUCKETS) - 1  # fallback: rare


def bucket_to_representative_interval(bucket: int) -> float:
    """A representative interval value for a bucket, used when the
    environment needs one concrete number (e.g. for compute_smp).
    Geometric midpoint -- appropriate given buckets are log-spaced."""
    lo, hi, _name = TRAFFIC_BUCKETS[bucket]
    return math.sqrt(lo * hi) if lo > 0 else hi / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# ACTION SPACE DEFINITION
# Same grid structure as Group F (run_group_f_full_grid.sh) for direct
# comparability against the static baseline pool.
# ─────────────────────────────────────────────────────────────────────────────

T3412_ACTIONS = [int(round(300 * (86400/300) ** (i/6))) for i in range(7)]
T3324_ACTIONS = [5, 10, 15, 20, 25, 30]

ACTIONS: List[Tuple[int, int]] = [
    (t3412, t3324) for t3412 in T3412_ACTIONS for t3324 in T3324_ACTIONS
]
N_ACTIONS = len(ACTIONS)  # 7 x 6 = 42


# ─────────────────────────────────────────────────────────────────────────────
# REWARD NORMALIZATION BOUNDS
# Pre-computed from the realistic config range (matches Group F grid) so
# normalization is stable and doesn't need re-deriving on every call.
# A small safety margin is added beyond the exact observed min/max so
# floating-point variation at the boundary doesn't get silently clipped
# (verified: exact action-space max delay is 43197.50s, right at the
# original bound — margin avoids clipping legitimate boundary values).
# ─────────────────────────────────────────────────────────────────────────────

LIFETIME_MIN_YR, LIFETIME_MAX_YR = 0.05, 100.0    # realistic range + margin
DELAY_MIN_S, DELAY_MAX_S         = 100.0, 50000.0  # realistic range + margin

_LOG_LT_MIN, _LOG_LT_MAX = math.log(LIFETIME_MIN_YR), math.log(LIFETIME_MAX_YR)
_LOG_DL_MIN, _LOG_DL_MAX = math.log(DELAY_MIN_S), math.log(DELAY_MAX_S)


def _log_normalize(value: float, log_min: float, log_max: float) -> float:
    """Clamp value to [min,max], log-transform, then scale to [0,1]."""
    value = max(value, 1e-6)
    log_v = math.log(value)
    log_v = max(log_min, min(log_max, log_v))
    return (log_v - log_min) / (log_max - log_min) if log_max > log_min else 0.5


def normalize_lifetime(lifetime_yr: float) -> float:
    """0 = worst (shortest) lifetime, 1 = best (longest) lifetime."""
    return _log_normalize(lifetime_yr, _LOG_LT_MIN, _LOG_LT_MAX)


def normalize_delay(delay_s: float) -> float:
    """0 = best (shortest) delay, 1 = worst (longest) delay."""
    return _log_normalize(delay_s, _LOG_DL_MIN, _LOG_DL_MAX)


# ─────────────────────────────────────────────────────────────────────────────
# THE ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepInfo:
    """Diagnostic info returned alongside the reward, for logging/debugging."""
    T3412: int
    T3324: int
    interval: float
    ce_mode: int
    lifetime_yr: float
    delay_s: float
    norm_lifetime: float
    norm_delay: float
    tau_fired: bool
    raw_reward: float


# DELAY_WEIGHT_BY_BUCKET: flat 0.45 across every bucket. Superseded the
# original state-dependent scheme (0.60/0.45/0.30/0.15 by bucket) after
# rigorous testing (Point 6 investigation, 9 candidate schemes x 10 seeds
# each, 90 total training runs): a flat weight of 0.45 produced both the
# highest mean policy accuracy (10.8/12) AND the lowest run-to-run
# variance (std 1.55) of every scheme tested, including the original
# state-dependent one -- direct, tested evidence that the delay weight's
# overall STRENGTH mattered more than varying it by traffic bucket. See
# project changelog Change 6 for the full investigation and evidence.
DELAY_WEIGHT_BY_BUCKET = {i: 0.45 for i in range(len(TRAFFIC_BUCKETS))}


class NbIotEnv:
    def __init__(self, w_lifetime: float = 0.7, w_delay: float = 1.0,
                 episode_length: int = 20, seed: int = None):
        self.w_lifetime = w_lifetime
        self.w_delay = w_delay
        self.episode_length = episode_length
        self.rng = random.Random(seed)

        self._step_count = 0
        self._ce_mode = 0
        self._traffic_bucket = 0

    @property
    def n_states(self) -> int:
        return len(CE_MODES) * len(TRAFFIC_BUCKETS)

    def _state_index(self, ce_mode: int, traffic_bucket: int) -> int:
        return ce_mode * len(TRAFFIC_BUCKETS) + traffic_bucket

    def current_state(self) -> int:
        return self._state_index(self._ce_mode, self._traffic_bucket)

    def reset(self) -> int:
        self._step_count = 0
        self._ce_mode = self.rng.choice(CE_MODES)
        self._traffic_bucket = self.rng.choice(range(len(TRAFFIC_BUCKETS)))
        return self.current_state()

    def step(self, action_idx: int) -> Tuple[int, float, bool, StepInfo]:
        T3412, T3324 = ACTIONS[action_idx]
        interval = bucket_to_representative_interval(self._traffic_bucket)
        ce = self._ce_mode

        result = compute_smp(T3412, T3324, interval, ce)

        norm_lt = normalize_lifetime(result.lifetime_yr)
        norm_dl = normalize_delay(result.delay_seconds)

        effective_w_delay = self.w_delay * DELAY_WEIGHT_BY_BUCKET[self._traffic_bucket]
        reward = self.w_lifetime * norm_lt - effective_w_delay * norm_dl

        info = StepInfo(
            T3412=T3412, T3324=T3324, interval=interval, ce_mode=ce,
            lifetime_yr=result.lifetime_yr, delay_s=result.delay_seconds,
            norm_lifetime=norm_lt, norm_delay=norm_dl,
            tau_fired=result.tau_fired, raw_reward=reward,
        )

        self._drift_conditions()

        self._step_count += 1
        done = self._step_count >= self.episode_length
        return self.current_state(), reward, done, info

    def _drift_conditions(self, p_ce_change: float = 0.15,
                          p_traffic_change: float = 0.25) -> None:
        if self.rng.random() < p_ce_change:
            self._ce_mode = self.rng.choice(CE_MODES)
        if self.rng.random() < p_traffic_change:
            self._traffic_bucket = self.rng.choice(range(len(TRAFFIC_BUCKETS)))
            
            
if __name__ == "__main__":
    env = NbIotEnv(seed=11)
    state = env.reset()
    print(f"Initial state: {state}  (ce={env._ce_mode}, bucket={env._traffic_bucket})")
    print(f"Action space size: {N_ACTIONS}")
    print(f"State space size: {env.n_states}")
    print()

    total_reward = 0.0
    for step in range(10):
        action_idx = env.rng.randrange(N_ACTIONS)
        next_state, reward, done, info = env.step(action_idx)
        total_reward += reward
        print(f"step={step:2d}  T3412={info.T3412:6d} T3324={info.T3324:3d}  "
              f"lifetime={info.lifetime_yr:7.2f}yr  delay={info.delay_s:8.1f}s  "
              f"reward={reward:+.4f}  next_state={next_state}")
        if done:
            print("  [episode ended]")
            state = env.reset()

    print(f"\nTotal reward over 10 steps: {total_reward:.4f}")
    print(f"Reward bounds check: all rewards should be in [-{env.w_delay:.1f}, +{env.w_lifetime:.1f}]")
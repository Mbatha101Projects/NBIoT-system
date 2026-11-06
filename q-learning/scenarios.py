"""


PURPOSE
───────
Defines specific, named, REPRODUCIBLE sequences of changing conditions,
used by evaluate.py to test the trained agent and compare it against
static baselines. This directly answers RQ4/Objective 4 ("how does the
adaptive model compare to static benchmarks... in use cases such as
asset tracking, smart agriculture etc.").

WHY THIS IS SEPARATE FROM environment.py's DRIFT LOGIC
──────────────────────────────────────────────────────
environment.py's NbIotEnv._drift_conditions() produces GENERIC,
RANDOMIZED changing conditions during TRAINING — this is deliberate,
so the agent sees a wide variety of situations and doesn't overfit to
one specific pattern. It is NOT reproducible (two runs give different
condition sequences) and NOT meant to represent any one real use case.

scenarios.py instead produces SPECIFIC, NAMED, REPRODUCIBLE sequences
for EVALUATION — you need to be able to say "here is exactly what
happened during the agricultural-irrigation test" and have it be
identical every time you check it. Each scenario extends this
project's own TC1-TC5 traffic profiles (already used in the NS-3
Group C sweep — run_benchmark.sh) from a single fixed configuration
into a SEQUENCE of conditions over time, which is what actually lets
evaluate.py test whether the agent adapts as conditions change within
a realistic deployment, rather than just picking one config and
freezing there.


"""

from dataclasses import dataclass
from typing import List, Tuple

from environment import traffic_to_bucket, CE_MODES
from environment import traffic_to_bucket, CE_MODES, TRAFFIC_BUCKETS


@dataclass
class ScenarioStep:
    """One held condition within a scenario."""
    ce_mode: int
    traffic_interval_s: float
    duration_steps: int
    label: str = ""  # optional human-readable note for this phase


@dataclass
class Scenario:
    """A named, reproducible sequence of conditions."""
    name: str
    description: str
    steps: List[ScenarioStep]

    def total_steps(self) -> int:
        return sum(s.duration_steps for s in self.steps)

    def flatten(self) -> List[Tuple[int, float]]:
        """Expand into one (ce_mode, traffic_interval_s) pair per step —
        the exact per-step condition sequence evaluate.py will replay."""
        flat = []
        for step in self.steps:
            flat.extend([(step.ce_mode, step.traffic_interval_s)] * step.duration_steps)
        return flat


# ─────────────────────────────────────────────────────────────────────────────
# NAMED SCENARIOS
#
# Each extends this project's original TC1-TC5 traffic profiles
# (run_benchmark.sh Group C) from a single fixed point into a realistic
# SEQUENCE — e.g. an irrigation system isn't ALWAYS reporting frequently,
# it bursts during active watering and goes quiet otherwise. This is
# exactly the kind of condition change an adaptive system should be able
# to respond to, and a static configuration structurally cannot.
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {

    "smart_meter": Scenario(
        name="Smart Meter (TC1 extended)",
        description=(
            "A utility meter reporting on a stable, infrequent schedule "
            "with consistently good coverage — the least demanding use "
            "case, where a static long-T3412 configuration should already "
            "perform close to optimally. Included as a baseline sanity "
            "check: an adaptive system should NOT meaningfully underperform "
            "a good static choice here, even though there's little room "
            "for it to add value."
        ),
        steps=[
            ScenarioStep(ce_mode=0, traffic_interval_s=7200, duration_steps=20,
                        label="steady infrequent reporting, good coverage"),
        ],
    ),

    "environmental_sensor": Scenario(
        name="Environmental Sensor (TC2 extended)",
        description=(
            "An environmental monitoring node with moderate, fairly "
            "constant reporting frequency and good coverage."
        ),
        steps=[
            ScenarioStep(ce_mode=0, traffic_interval_s=3600, duration_steps=20,
                        label="steady moderate reporting"),
        ],
    ),

    "agricultural_irrigation": Scenario(
        name="Agricultural Irrigation (TC3a/TC3b extended)",
        description=(
            "The scenario this project's central finding is built around: "
            "an irrigation controller that is normally in a low-traffic, "
            "dry-season mode (long reporting interval, medium coverage), "
            "but BURSTS into frequent reporting during active watering "
            "periods before returning to quiet mode. A static configuration "
            "must compromise between these two regimes; an adaptive system "
            "should switch T3412/T3324 as the burst starts and ends."
        ),
        steps=[
            ScenarioStep(ce_mode=1, traffic_interval_s=7200, duration_steps=8,
                        label="dry period, infrequent reporting"),
            ScenarioStep(ce_mode=1, traffic_interval_s=900, duration_steps=6,
                        label="active watering, frequent reporting burst"),
            ScenarioStep(ce_mode=1, traffic_interval_s=7200, duration_steps=8,
                        label="dry period resumes"),
        ],
    ),

    "wildlife_tracker": Scenario(
        name="Wildlife Tracker (TC4 extended)",
        description=(
            "An asset/wildlife tracker in poor coverage (deep terrain or "
            "dense cover), reporting frequently — the most demanding "
            "combination in this project's grid (short interval AND poor "
            "coverage), where the tradeoff between energy efficiency and "
            "responsiveness is sharpest. Occasionally moves into better "
            "coverage (e.g. reaching open ground), testing whether the "
            "agent adapts to coverage changes, not just traffic changes."
        ),
        steps=[
            ScenarioStep(ce_mode=2, traffic_interval_s=300, duration_steps=10,
                        label="poor coverage, frequent reporting"),
            ScenarioStep(ce_mode=0, traffic_interval_s=300, duration_steps=5,
                        label="brief move into open ground, good coverage"),
            ScenarioStep(ce_mode=2, traffic_interval_s=300, duration_steps=10,
                        label="returns to poor coverage"),
        ],
    ),

    "industrial_monitor": Scenario(
        name="Industrial Monitor (TC5 extended)",
        description=(
            "An industrial monitoring node with moderate reporting "
            "frequency and good coverage, but with an occasional shift to "
            "degraded coverage (e.g. equipment/machinery interference) — "
            "tests coverage-mode adaptation specifically, with traffic "
            "held relatively stable."
        ),
        steps=[
            ScenarioStep(ce_mode=0, traffic_interval_s=1800, duration_steps=12,
                        label="normal operation, good coverage"),
            ScenarioStep(ce_mode=1, traffic_interval_s=1800, duration_steps=8,
                        label="interference period, degraded coverage"),
        ],
    ),

    "mixed_stress_test": Scenario(
        name="Mixed Stress Test",
        description=(
            "Not tied to one specific use case — deliberately cycles "
            "through EVERY combination of CE mode and a wide traffic "
            "range, to test whether the agent's learned policy holds up "
            "across the full state space, not just the specific use-case "
            "patterns above."
        ),
        steps=[
            ScenarioStep(ce_mode=ce, traffic_interval_s=interval, duration_steps=3,
                        label=f"CE-{ce}, interval={interval}s")
            for ce in [0, 1, 2]
            for interval in [300, 1800, 7200, 43200]
        ],
    ),

}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios() -> None:
    for key, s in SCENARIOS.items():
        print(f"  {key:24s}  ({s.total_steps():3d} steps)  {s.name}")


if __name__ == "__main__":
    # Smoke test: confirm every scenario is well-formed and can be
    # flattened into a per-step condition sequence without error.
    print("Registered scenarios:")
    list_scenarios()
    print()

    for key, scenario in SCENARIOS.items():
        flat = scenario.flatten()
        assert len(flat) == scenario.total_steps(), \
            f"{key}: flatten() length mismatch"

        # Confirm every condition in the scenario maps to a valid
        # (ce_mode, traffic_bucket) state the environment/agent can handle
        for ce_mode, interval in flat:
            assert ce_mode in CE_MODES, f"{key}: invalid ce_mode {ce_mode}"
            bucket = traffic_to_bucket(interval)
            assert 0 <= bucket <= len(TRAFFIC_BUCKETS)-1, f"{key}: invalid bucket {bucket} for interval {interval}"

    print(f"[OK] All {len(SCENARIOS)} scenarios are well-formed.")
    print()

    print("Detail — 'agricultural_irrigation' step-by-step:")
    for ce_mode, interval in SCENARIOS["agricultural_irrigation"].flatten():
        bucket = traffic_to_bucket(interval)
        print(f"  CE={ce_mode}  interval={interval:6.0f}s  -> bucket={bucket}")
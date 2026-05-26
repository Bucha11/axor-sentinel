from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccessEvent:
    """
    Thin serialisation DTO for a resource access within a bench SessionRecord.

    Uses plain ``signal_type: str`` (not the runtime ``SignalType`` enum) so bench
    scenario files can be serialised/deserialised without importing the sentinel
    package. Compare with ``sentinel.cycle.ResourceAccess`` — the strongly-typed
    runtime counterpart.
    """
    resource_id: str
    signal_type: str        # SignalType value string
    container_id: str = ""


@dataclass
class ResourceNode:
    """Resource node as recorded in a bench scenario."""
    resource_id: str
    normalization_method: str   # "provider_id" | "path" | "heuristic"
    canonical_confidence: float
    container_id: str = ""
    service: str = ""


@dataclass
class SessionRecord:
    """Synthetic session record for bench dataset generation."""
    session_id: str
    agent_id: str
    started_at: float
    duration_seconds: float
    accessed_resources: list[AccessEvent]
    had_taint: bool
    had_export_attempt: bool
    had_failed_export: bool
    had_escalation: bool


@dataclass
class GroundTruth:
    """
    Known outcomes for a bench scenario.

    expected_flagged_ids: resources axor-sentinel MUST flag at the end of
    the scenario (the detection target).
    """
    tainted_session_ids: list[str]
    export_session_id: str | None
    staging_resource_ids: list[str]     # resources used for staging
    expected_flagged_ids: list[str]     # resources axor-sentinel must flag
    gap_sessions: int
    gap_days: float


@dataclass
class Scenario:
    """
    A complete synthetic scenario with ground truth for eval.

    label:        "ATTACK" | "BENIGN"
    attack_class: "slow_and_low" | "fanout" | "distributed_staging" |
                  "single_session_burst" | None
    """
    scenario_id: str
    label: str                  # "ATTACK" | "BENIGN"
    attack_class: str | None    # None for benign scenarios
    attack_params: dict         # for reproducibility
    agent_profile: str          # "narrow" | "broad" | "noisy" | "etl" | "research"
    topology_id: str
    sessions: list[SessionRecord]
    resources: list[ResourceNode]
    ground_truth: GroundTruth

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class ReputationEvent:
    """
    Immutable evidence record for every suspicion_score change.

    Stored as (:ReputationEvent) nodes linked to (:Resource) in Neo4j.
    Accessible via the same token-gated path as DecisionTrace.

    signal_type values:
        "read"                  — hot, read-only
        "read_summarize"        — hot, read + summarized
        "read_export_adjacent"  — hot, read + export nearby
        "read_export_failed"    — hot, failed export / escalation
        "caution_adjacent"      — caution spread to adjacent resource
        "fanout"                — fanout signal contribution
    """
    event_id: str
    resource_id: str
    session_id: str
    taint_source: str           # TaintSource.value — for poisoning-mitigation tracking
    signal_type: str
    raw_weight: float
    effective_weight: float     # after canonical_confidence * diversity * dampening
    score_before: float
    score_after: float
    reason: str
    timestamp: float

    @classmethod
    def create(
        cls,
        resource_id: str,
        session_id: str,
        taint_source: str,
        signal_type: str,
        raw_weight: float,
        effective_weight: float,
        score_before: float,
        score_after: float,
        reason: str,
        timestamp: float,
    ) -> "ReputationEvent":
        return cls(
            event_id=uuid.uuid4().hex,
            resource_id=resource_id,
            session_id=session_id,
            taint_source=taint_source,
            signal_type=signal_type,
            raw_weight=raw_weight,
            effective_weight=effective_weight,
            score_before=score_before,
            score_after=score_after,
            reason=reason,
            timestamp=timestamp,
        )


@dataclass
class FanoutSignal:
    """
    Emitted when a tainted session touches containers at > 2.5σ above agent baseline.

    Contributes a flat 0.5 weight to all touched resources (invariant A-10).
    """
    origin_session_id: str
    agent_id: str
    taint_source: str           # TaintSource.value
    affected_resources: list[str]
    unique_containers: int
    baseline_mean: float
    z_score: float              # deviation from agent's historical baseline
    window_minutes: float


@dataclass
class AgentContainerBaseline:
    """
    Per-agent baseline for fanout detection.

    Computed from the last BASELINE_WINDOW_SESSIONS (default: 50) completed sessions.
    Cold-start guard: detection disabled when session_count < FANOUT_MIN_SESSIONS (10).
    """
    agent_id: str
    mean_containers_per_session: float
    std_containers_per_session: float
    session_count: int          # sessions used to compute this baseline
    last_updated: float

"""Declared decidable predicates — the deterministic verdict layer.

A resource's verdict is a level in a small monotone lattice, derived by
predicates over its windowed evidence set. Every constant here is DECLARED
POLICY (semantic, auditable — the same class as core's detection_floor or
excision's diffuse limit), not a calibrated statistic: nothing is fitted to a
dataset, so a measured false-positive rate on the bench is a property, not an
overfit. This mirrors the axor-probe rework: verdict = decidable fact;
scalar scores are demoted to telemetry.

Predicates (any fired → FLAGGED; facts name the evidence):

  P1  export-denied     an EXPORT_FAILED fact exists (a denied export touching
                        this resource — the hardest single fact; today's weight
                        1.0 ≥ 0.7 flags on it alone, so P1 preserves behavior).
  P2  export-adjacency  EXPORT_ADJACENT facts from ≥ export_adjacent_origins
                        DISTINCT origins.
  P3  staging count     ≥ staging_sessions DISTINCT tainted sessions with
                        rank ≥ READ_SUMMARIZE.
  P4  staging sequence  read-under-taint in one session, then EXPORT_ADJACENT+
                        on the same resource in a LATER, different session —
                        the slow-and-low structure itself, as a conjunction.

WATCH (grey tier): any single tainted fact with rank ≥ READ_SUMMARIZE, or a
P1–P4 hit supported only by heuristic-resolved ids (the resolution ceiling:
an id the normalizer only guessed cannot FLAG on its own), or graph adjacency
to a FLAGGED resource (applied by the cycle, not here).

The wire codomain is FINITE: LEVEL_SUSPICION maps levels to the snapshot
floats core's enricher converts (suspicion 1.0 → reputation 1e-3 crosses a
detection_floor of 0.3; WATCH 0.4 → 0.6 does not) — so the decidability holds
end-to-end through core's floor comparison, with no calibrated threshold
anywhere on the path.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.evidence import Evidence


class ReputationLevel(IntEnum):
    CLEAN = 0
    WATCH = 1
    FLAGGED = 2


# Finite wire codomain: the only suspicion values a deterministic sentinel
# emits. Operator guidance: detection_floor in (0.001, 0.6) tightens on FLAGGED
# only (the existing 0.3 default keeps its exact meaning); floor ≥ 0.6 also
# tightens on WATCH.
LEVEL_SUSPICION: dict[ReputationLevel, float] = {
    ReputationLevel.CLEAN: 0.0,
    ReputationLevel.WATCH: 0.4,
    ReputationLevel.FLAGGED: 1.0,
}


@dataclass(frozen=True)
class SentinelPolicy:
    """Declared predicate constants — policy, not calibration.

    Each value is a semantic statement an operator can read and change with
    understanding ("3 tainted staging sessions inside 30 days"), unlike a
    ROC-tuned threshold. Defaults are sanity-checked against the bench, but
    never fitted to it.
    """
    window_days: float = 30.0          # evidence TTL (replaces the decay half-life)
    staging_sessions: int = 3          # P3: distinct tainted sessions, rank ≥ READ_SUMMARIZE
    export_adjacent_origins: int = 2   # P2: distinct origins with EXPORT_ADJACENT
    fanout_containers: int = 7         # quota: distinct containers in one tainted session
    container_flagged_members: int = 2 # container FLAGGED at ≥ this many flagged members


@dataclass(frozen=True)
class Verdict:
    """Decidable outcome for one resource: a lattice level plus the facts that
    produced it (human-readable, in the spirit of core's AnomalyResult.reasons)."""
    level: ReputationLevel
    facts: tuple[str, ...] = ()


_STAGING_MIN_RANK = SignalType.READ_SUMMARIZE
_EXPORT_ADJACENT = SignalType.READ_EXPORT_ADJACENT
_EXPORT_FAILED = SignalType.READ_EXPORT_FAILED


def _fired_predicates(evidence: Sequence[Evidence], policy: SentinelPolicy) -> list[str]:
    """Evaluate P1–P4 over an evidence subset; return the facts that fired."""
    facts: list[str] = []

    failed = [e for e in evidence if e.rank == _EXPORT_FAILED]
    if failed:
        facts.append(f"P1:export_denied:sessions={sorted({e.session_id for e in failed})}")

    adj_origins = sorted({e.origin for e in evidence if e.rank >= _EXPORT_ADJACENT})
    if len(adj_origins) >= policy.export_adjacent_origins:
        facts.append(f"P2:export_adjacent_origins={adj_origins}")

    staging = sorted({
        e.session_id for e in evidence
        if e.tainted and e.rank >= _STAGING_MIN_RANK
    })
    if len(staging) >= policy.staging_sessions:
        facts.append(f"P3:staging_sessions={staging}")

    reads = [e for e in evidence if e.tainted and e.rank < _EXPORT_ADJACENT]
    exports = [e for e in evidence if e.rank >= _EXPORT_ADJACENT]
    for r in reads:
        later = [
            x for x in exports
            if x.observed_at > r.observed_at and x.session_id != r.session_id
        ]
        if later:
            facts.append(
                f"P4:staged_then_export:read={r.session_id}"
                f":export={later[0].session_id}"
            )
            break

    return facts


def evaluate_resource(
    evidence: Sequence[Evidence],
    policy: SentinelPolicy,
    now: float,
) -> Verdict:
    """Decidable verdict for one resource from its windowed evidence."""
    horizon = now - policy.window_days * 86400.0
    in_window = [e for e in evidence if e.observed_at >= horizon]
    if not in_window:
        return Verdict(ReputationLevel.CLEAN)

    strong = [e for e in in_window if e.resolution != "heuristic"]

    flag_facts = _fired_predicates(strong, policy)
    if flag_facts:
        return Verdict(ReputationLevel.FLAGGED, tuple(flag_facts))

    watch_facts: list[str] = []
    # Resolution ceiling: predicates that fire only with heuristic-resolved ids
    # reach WATCH, never FLAGGED — a guessed id must not tighten core on its own.
    ceiling = _fired_predicates(in_window, policy)
    if ceiling:
        watch_facts.extend(f"heuristic_ceiling:{f}" for f in ceiling)

    grey = sorted({
        e.session_id for e in in_window
        if e.tainted and e.rank >= _STAGING_MIN_RANK
    })
    if grey:
        watch_facts.append(f"W1:tainted_staging_sessions={grey}")

    if watch_facts:
        return Verdict(ReputationLevel.WATCH, tuple(watch_facts))
    return Verdict(ReputationLevel.CLEAN)


def evaluate_container(
    member_levels: Iterable[ReputationLevel],
    policy: SentinelPolicy,
) -> Verdict:
    """Container verdict by counting member verdicts (replaces the mean-over-
    threshold aggregate): FLAGGED at ≥ container_flagged_members flagged
    members, WATCH when any member is above CLEAN."""
    levels = list(member_levels)
    flagged = sum(1 for lvl in levels if lvl == ReputationLevel.FLAGGED)
    if flagged >= policy.container_flagged_members:
        return Verdict(ReputationLevel.FLAGGED, (f"C1:flagged_members={flagged}",))
    if flagged or any(lvl >= ReputationLevel.WATCH for lvl in levels):
        return Verdict(ReputationLevel.WATCH, ("C2:member_above_clean",))
    return Verdict(ReputationLevel.CLEAN)


def fanout_exceeded(
    tainted: bool,
    container_ids: Iterable[str],
    max_rank: SignalType | None,
    policy: SentinelPolicy,
) -> bool:
    """Deterministic fanout quota (replaces the self-trained z-score baseline).

    A tainted session touching more than fanout_containers DISTINCT containers
    at rank ≥ READ_SUMMARIZE is a fanout fact — exact counting against a
    declared quota: no cold start, no smoothing, no baseline an attacker can
    walk upward (closes limitation F5 by construction).
    """
    if not tainted or max_rank is None or max_rank < _STAGING_MIN_RANK:
        return False
    return len(set(container_ids)) > policy.fanout_containers

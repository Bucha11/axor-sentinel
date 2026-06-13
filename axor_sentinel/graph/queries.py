from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # neo4j.Session imported at call sites to keep base package dep-free

# ── Cypher query strings ───────────────────────────────────────────────────────

# Apply time decay to all resources with non-zero score.
# Always runs first in each audit cycle (invariant A-4).
# last_decay_at is updated here; last_signal_at is NOT (invariant A-3).
DECAY_QUERY = """
MATCH (r:Resource)
WHERE r.suspicion_score > 0
WITH r, (timestamp() - r.last_decay_at) / 86400000.0 AS days_elapsed
SET r.suspicion_score = r.suspicion_score * (0.5 ^ (days_elapsed / 30.0))
SET r.flagged = (r.suspicion_score >= $flag_threshold)
SET r.last_decay_at = timestamp()
"""

# Apply graded hot weights after a tainted session.
# effective_weight = raw_weight * canonical_confidence (A-8 base; diversity/dampening pre-computed).
# last_signal_at updated; last_decay_at NOT updated (invariant A-3).
HOT_WEIGHT_QUERY = """
MATCH (s:Session {had_taint: true, session_id: $session_id})
MATCH (s)-[a:ACCESSED {signal_type: $signal_type}]->(r:Resource)
WITH r, $raw_weight * r.canonical_confidence AS eff_weight
WITH r, r.suspicion_score + eff_weight * (1.0 - r.suspicion_score) AS new_score
SET r.suspicion_score = CASE WHEN new_score > 1.0 THEN 1.0 ELSE new_score END
SET r.flagged = (r.suspicion_score >= $flag_threshold)
SET r.last_signal_at = timestamp()
"""

# Apply caution to adjacent resources not directly accessed in the session.
# Caution weight = BASE_CAUTION * topology_factor * time_decay * canonical_confidence.
# last_signal_at updated; last_decay_at NOT updated (invariant A-3).
CAUTION_ADJACENT_QUERY = """
MATCH (s:Session {had_taint: true, session_id: $session_id})
MATCH (s)-[:ACCESSED]->(hot:Resource)
MATCH (hot)-[adj:ADJACENT_TO]->(neighbor:Resource)
WHERE NOT (s)-[:ACCESSED]->(neighbor)
WITH neighbor, adj.topology_factor AS tf,
     (timestamp() - neighbor.last_decay_at) / 86400000.0 AS days_elapsed
WITH neighbor, tf,
     0.3 * tf * (0.5 ^ (days_elapsed / 30.0)) AS caution_weight,
     neighbor.canonical_confidence AS conf
WITH neighbor,
     caution_weight * conf AS eff_weight
WITH neighbor,
     neighbor.suspicion_score + eff_weight * (1.0 - neighbor.suspicion_score) AS new_score
SET neighbor.suspicion_score = CASE WHEN new_score > 1.0 THEN 1.0 ELSE new_score END
SET neighbor.flagged = (neighbor.suspicion_score >= $flag_threshold)
SET neighbor.last_signal_at = timestamp()
"""

# Apply fanout flat weight to a batch of resources (invariant A-10).
# Called after hot weights when a FanoutSignal is emitted.
# last_signal_at updated; last_decay_at NOT updated (invariant A-3).
FANOUT_WEIGHT_QUERY = """
UNWIND $resource_ids AS rid
MATCH (r:Resource {id: rid})
WITH r, r.suspicion_score + $fanout_weight * (1.0 - r.suspicion_score) AS new_score
SET r.suspicion_score = CASE WHEN new_score > 1.0 THEN 1.0 ELSE new_score END
SET r.flagged = (r.suspicion_score >= $flag_threshold)
SET r.last_signal_at = timestamp()
"""

# Slow-and-low detection: agent with a tainted session followed by an export session,
# where flagged resources were accessed during the tainted session.
SLOW_AND_LOW_QUERY = """
MATCH (ag:Agent)-[:IN_SESSION]->(s1:Session {had_taint: true})
MATCH (ag)-[:IN_SESSION]->(s2:Session {had_export_attempt: true})
WHERE s2.started_at > s1.started_at
AND s2.started_at - s1.started_at > $min_gap_ms
WITH ag, s1, s2,
     (s2.started_at - s1.started_at) / 86400000.0 AS gap_days
MATCH (s1)-[:ACCESSED]->(r:Resource)
WHERE r.flagged = true
RETURN ag.agent_id,
       s1.session_id, s2.session_id,
       gap_days,
       collect(r.id) AS flagged_resources,
       collect(r.suspicion_score) AS scores
ORDER BY gap_days DESC
"""


# ── Query runner functions ─────────────────────────────────────────────────────

def apply_decay(session: Any, flag_threshold: float) -> None:
    """
    Apply time decay to all resources with non-zero suspicion_score.

    Args:
        session:        neo4j.Session
        flag_threshold: threshold above which flagged=True (default FLAG_THRESHOLD=0.7)
    """
    session.run(DECAY_QUERY, flag_threshold=flag_threshold)


def apply_hot_weight(
    session: Any,
    session_id: str,
    signal_type: str,
    raw_weight: float,
    flag_threshold: float,
) -> None:
    """
    Apply graded hot weight to resources accessed in a tainted session.

    Args:
        session:        neo4j.Session
        session_id:     ID of the tainted session
        signal_type:    SignalType value string
        raw_weight:     pre-computed raw weight (before canonical_confidence scaling)
        flag_threshold: threshold for flagged=True
    """
    session.run(
        HOT_WEIGHT_QUERY,
        session_id=session_id,
        signal_type=signal_type,
        raw_weight=raw_weight,
        flag_threshold=flag_threshold,
    )


def apply_caution_adjacent(
    session: Any,
    session_id: str,
    flag_threshold: float,
) -> None:
    """
    Apply caution weight to resources adjacent to those accessed in the tainted session.

    Args:
        session:        neo4j.Session
        session_id:     ID of the tainted session
        flag_threshold: threshold for flagged=True
    """
    session.run(
        CAUTION_ADJACENT_QUERY,
        session_id=session_id,
        flag_threshold=flag_threshold,
    )


def apply_fanout_weight(
    session: Any,
    resource_ids: list[str],
    fanout_weight: float,
    flag_threshold: float,
) -> None:
    """
    Apply fanout flat weight to all touched resources in one batch (invariant A-10).

    Called after hot weights when a FanoutSignal fires.  The fanout contribution
    is written to Neo4j as a separate accumulate call, keeping it distinct from
    the per-signal hot weight (spec §5.8).

    Args:
        session:        neo4j.Session
        resource_ids:   list of resource IDs to boost
        fanout_weight:  flat weight to apply (FANOUT_WEIGHT = 0.5)
        flag_threshold: threshold for flagged=True
    """
    if not resource_ids:
        return
    session.run(
        FANOUT_WEIGHT_QUERY,
        resource_ids=resource_ids,
        fanout_weight=fanout_weight,
        flag_threshold=flag_threshold,
    )


def slow_and_low_detection(
    session: Any,
    min_gap_ms: float,
) -> list[dict]:
    """
    Run the slow-and-low staging detection query.

    Args:
        session:    neo4j.Session
        min_gap_ms: minimum millisecond gap between tainted and export session

    Returns:
        List of dicts with keys: agent_id, s1.session_id, s2.session_id,
        gap_days, flagged_resources, scores. Ordered by gap_days descending.
    """
    result = session.run(SLOW_AND_LOW_QUERY, min_gap_ms=min_gap_ms)
    return [dict(record) for record in result]

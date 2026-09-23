from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # neo4j.Session imported at call sites to keep base package dep-free

# ── Cypher query strings ───────────────────────────────────────────────────────

# Apply time decay to all resources with non-zero score.
# Always runs first in each audit cycle (invariant A-4).
# last_decay_at is updated here; last_signal_at is NOT (invariant A-3).
#
# Score-0 nodes are skipped, so their last_decay_at is NOT advanced here.  That
# is only sound because every write that lifts a node off 0 (hot / caution /
# fanout below) restarts its decay clock at that moment: last_decay_at means
# "the score has been decaying since", and a node that sat at 0 from day 0 to
# day 100 has nothing to decay for those 100 days.  Without the restart the
# first decay after a day-100 hit applied 0.5^(100/30) ≈ 0.1 to a fresh signal.
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
# last_signal_at updated; last_decay_at NOT updated (invariant A-3) — except when
# the score was 0 before this write: then the decay clock starts now (see
# DECAY_QUERY; a score-0 node's last_decay_at is stale by design).
#
# Matched on the SPECIFIC resource id (not just signal_type): a session can access
# several resources sharing one signal_type, and the cycle calls this once per
# access — without the id filter every call would re-apply to all of them, which is
# now a live double-count since the snapshot reads these scores back. Returns the
# resource's score before/after so the cycle can record faithful evidence without
# re-deriving it in Python (the source-of-truth is Neo4j).
HOT_WEIGHT_QUERY = """
MATCH (s:Session {had_taint: true, session_id: $session_id})
MATCH (s)-[a:ACCESSED {signal_type: $signal_type}]->(r:Resource {id: $resource_id})
WITH r, r.suspicion_score AS score_before, $raw_weight * r.canonical_confidence AS eff_weight
WITH r, score_before, score_before + eff_weight * (1.0 - score_before) AS new_score
SET r.last_decay_at = CASE
    WHEN coalesce(score_before, 0.0) <= 0.0 THEN timestamp() ELSE r.last_decay_at
END
SET r.suspicion_score = CASE WHEN new_score > 1.0 THEN 1.0 ELSE new_score END
SET r.flagged = (r.suspicion_score >= $flag_threshold)
SET r.last_signal_at = timestamp()
RETURN score_before AS score_before, r.suspicion_score AS score_after
"""

# Apply caution to adjacent resources not directly accessed in the session.
# Caution weight = BASE_CAUTION * topology_factor * canonical_confidence.
# last_signal_at updated; last_decay_at NOT updated (invariant A-3) — except that
# a neighbour lifted off 0 starts its decay clock now (see DECAY_QUERY).
#
# The spec's time_decay(days_since_last_decay) term is 1 at application time and
# is deliberately not computed here.  Caution is written in the same cycle as the
# hot signal that causes it, right after DECAY_QUERY, so for any scored node
# days_since_last_decay ≈ 0 and the factor was ≈ 1 anyway; the only nodes for
# which it differed were score-0 neighbours, whose last_decay_at is stale by
# design — there it shrank caution by the neighbour's AGE (a node created 100
# days ago got ~10% of the caution a new one got), which is not a property of
# the signal at all.  Ageing of the caution contribution is DECAY_QUERY's job on
# later cycles, exactly as for hot weights.
CAUTION_ADJACENT_QUERY = """
MATCH (s:Session {had_taint: true, session_id: $session_id})
MATCH (s)-[:ACCESSED]->(hot:Resource)
MATCH (hot)-[adj:ADJACENT_TO]->(neighbor:Resource)
WHERE NOT (s)-[:ACCESSED]->(neighbor)
WITH neighbor, adj.topology_factor AS tf
WITH neighbor,
     0.3 * tf * neighbor.canonical_confidence AS eff_weight
WITH neighbor, neighbor.suspicion_score AS score_before,
     neighbor.suspicion_score + eff_weight * (1.0 - neighbor.suspicion_score) AS new_score
SET neighbor.last_decay_at = CASE
    WHEN coalesce(score_before, 0.0) <= 0.0 THEN timestamp() ELSE neighbor.last_decay_at
END
SET neighbor.suspicion_score = CASE WHEN new_score > 1.0 THEN 1.0 ELSE new_score END
SET neighbor.flagged = (neighbor.suspicion_score >= $flag_threshold)
SET neighbor.last_signal_at = timestamp()
"""

# Apply fanout flat weight to a batch of resources (invariant A-10).
# Called after hot weights when a FanoutSignal is emitted.
# last_signal_at updated; last_decay_at NOT updated (invariant A-3) — except that
# a resource lifted off 0 starts its decay clock now (see DECAY_QUERY).
FANOUT_WEIGHT_QUERY = """
UNWIND $resource_ids AS rid
MATCH (r:Resource {id: rid})
WITH r, r.suspicion_score AS score_before,
     r.suspicion_score + $fanout_weight * (1.0 - r.suspicion_score) AS new_score
SET r.last_decay_at = CASE
    WHEN coalesce(score_before, 0.0) <= 0.0 THEN timestamp() ELSE r.last_decay_at
END
SET r.suspicion_score = CASE WHEN new_score > 1.0 THEN 1.0 ELSE new_score END
SET r.flagged = (r.suspicion_score >= $flag_threshold)
SET r.last_signal_at = timestamp()
"""

# Read every resource's current suspicion_score back from the graph.  Neo4j is the
# source of truth for the snapshot: after decay + hot + fanout + caution have run,
# the cycle reads scores from HERE rather than re-accumulating in Python.  Resources
# at score 0 carry no reputation, so they are excluded to keep the snapshot lean.
RESOURCE_SCORES_QUERY = """
MATCH (r:Resource)
WHERE r.suspicion_score > 0
RETURN r.id AS id, r.suspicion_score AS score
"""

# Slow-and-low detection: agent with a tainted session followed by an export session,
# where flagged resources were accessed during the tainted session.
#
# Every RETURN column is aliased: an unaliased ``ag.agent_id`` comes back from
# the driver under the key "ag.agent_id", while SlowAndLowReport and the docs
# promise "agent_id".  The session columns keep their documented keys
# ("s1.session_id" / "s2.session_id") via backtick aliases, so no consumer of the
# documented shape changes.
#
# The empty agent_id is excluded: "" is "no agent identity", and every session
# upserted without one hangs off the SAME Agent {agent_id: ""} node — so an
# unrelated tainted session and export session from different anonymous agents
# would be correlated as one agent staging data.
SLOW_AND_LOW_QUERY = """
MATCH (ag:Agent)-[:IN_SESSION]->(s1:Session {had_taint: true})
WHERE ag.agent_id <> ''
MATCH (ag)-[:IN_SESSION]->(s2:Session {had_export_attempt: true})
WHERE s2.started_at > s1.started_at
AND s2.started_at - s1.started_at > $min_gap_ms
WITH ag, s1, s2,
     (s2.started_at - s1.started_at) / 86400000.0 AS gap_days
MATCH (s1)-[:ACCESSED]->(r:Resource)
WHERE r.flagged = true
RETURN ag.agent_id AS agent_id,
       s1.session_id AS `s1.session_id`, s2.session_id AS `s2.session_id`,
       gap_days AS gap_days,
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
    resource_id: str,
) -> tuple[float, float] | None:
    """
    Apply graded hot weight to ONE resource accessed in a tainted session.

    Args:
        session:        neo4j.Session
        session_id:     ID of the tainted session
        signal_type:    SignalType value string
        raw_weight:     pre-computed raw weight (before canonical_confidence scaling)
        flag_threshold: threshold for flagged=True
        resource_id:    the specific resource the signal applies to

    Returns:
        ``(score_before, score_after)`` for the resource, or ``None`` if the
        session/edge/resource did not match (e.g. untainted session).
    """
    rec = session.run(
        HOT_WEIGHT_QUERY,
        session_id=session_id,
        signal_type=signal_type,
        raw_weight=raw_weight,
        flag_threshold=flag_threshold,
        resource_id=resource_id,
    ).single()
    if rec is None:
        return None
    return rec["score_before"], rec["score_after"]


def read_resource_scores(session: Any) -> dict[str, float]:
    """
    Read every resource's current suspicion_score from Neo4j (the snapshot source
    of truth). Returns ``{resource_id: score}`` for all resources with score > 0.
    """
    result = session.run(RESOURCE_SCORES_QUERY)
    return {rec["id"]: rec["score"] for rec in result}


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
        gap_days, flagged_resources, scores (explicit aliases in the query, so
        these are the real driver keys). Ordered by gap_days descending.
        Sessions of agents with an empty agent_id are never correlated.
    """
    result = session.run(SLOW_AND_LOW_QUERY, min_gap_ms=min_gap_ms)
    return [dict(record) for record in result]

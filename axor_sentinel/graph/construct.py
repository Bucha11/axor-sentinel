"""Graph construction — materialise the reputation graph each audit cycle.

The scoring Cypher in :mod:`axor_sentinel.graph.queries` only ever *reads* and
*updates* nodes; nothing in sentinel used to *create* them, so on a real Neo4j
every write matched an empty graph and did nothing.  This module is the missing
producer: per cycle it upserts the ``Agent`` / ``Session`` / ``Resource`` nodes
and the ``ACCESSED`` / ``IN_SESSION`` edges the hot-weight and slow-and-low
queries walk, and derives ``ADJACENT_TO`` edges (which nothing else writes) from
container co-membership so the caution query is no longer inert.

It runs first (Step 0) in ``SentinelCycle._run_once_locked``, before decay and the
hot-weight/caution writes.  Neo4j is the source of truth: the cycle reads the
snapshot back from the graph after all writes (``RESOURCE_SCORES_QUERY``), so this
producer is what makes the scores, caution and slow-and-low detection real rather
than no-ops against an empty graph.
"""
from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axor_sentinel.sentinel.cycle import SessionSummary

# Topology factor for resources that share a container.  The spec grades
# adjacency (same directory/workspace 1.0, same service 0.7, same MCP namespace
# 0.6, …) but the cycle is only handed container *membership*, not the container
# type, so same-container pairs use the strongest factor by default.  A caller
# with richer topology can override it.
SAME_CONTAINER_TOPOLOGY_FACTOR: float = 1.0


# Agent + Session only.  Kept separate from the resource upsert so no single
# query string contains both ``had_taint`` and ``last_signal_at`` — the cycle's
# query-ordering test matches the hot-weight query by exactly that pair, and a
# combined upsert would be a false positive.
UPSERT_SESSION_QUERY = """
MERGE (ag:Agent {agent_id: $agent_id})
MERGE (s:Session {session_id: $session_id})
SET s.had_taint = $had_taint,
    s.had_export_attempt = $had_export_attempt,
    s.had_failed_export = $had_failed_export,
    s.had_escalation = $had_escalation,
    s.started_at = $started_at_ms
MERGE (ag)-[:IN_SESSION]->(s)
"""

# Resource nodes + ACCESSED edges for one session.  ON CREATE seeds the score
# (and a fresh last_decay_at so the next decay does not treat a brand-new node as
# ancient); canonical_confidence is refreshed every time because the hot-weight
# Cypher multiplies by it and must agree with the Python effective weight.
UPSERT_ACCESS_QUERY = """
MATCH (s:Session {session_id: $session_id})
UNWIND $accesses AS acc
MERGE (r:Resource {id: acc.resource_id})
ON CREATE SET r.suspicion_score = acc.seed_score,
              r.flagged = (acc.seed_score >= $flag_threshold),
              r.last_decay_at = timestamp(),
              r.last_signal_at = timestamp()
SET r.canonical_confidence = acc.canonical_confidence
MERGE (s)-[a:ACCESSED {signal_type: acc.signal_type}]->(r)
SET a.at = timestamp()
"""

# Symmetric ADJACENT_TO edges between resources that share a container.  Both
# directions are written because the caution query walks (hot)-[:ADJACENT_TO]->.
ADJACENCY_QUERY = """
UNWIND $pairs AS p
MATCH (a:Resource {id: p.source})
MATCH (b:Resource {id: p.target})
MERGE (a)-[adj:ADJACENT_TO]->(b)
SET adj.topology_factor = $topology_factor
"""


def upsert_graph(
    session: Any,
    sessions: list[SessionSummary],
    resource_scores: dict[str, float],
    container_members: dict[str, list[str]],
    *,
    flag_threshold: float,
    topology_factor: float = SAME_CONTAINER_TOPOLOGY_FACTOR,
) -> None:
    """Materialise this cycle's sessions, resources and adjacency in Neo4j.

    Args:
        session:           live neo4j.Session
        sessions:          every session this cycle (NOT just tainted ones — the
                           slow-and-low query needs the untainted export session)
        resource_scores:   resource_id → current score, used to seed brand-new
                           Resource nodes so they start where Python thinks they are
        container_members: container_id → [resource_ids], source of ADJACENT_TO
        flag_threshold:    score at/above which a seeded resource is flagged
        topology_factor:   weight written on same-container ADJACENT_TO edges
    """
    for s in sessions:
        session.run(
            UPSERT_SESSION_QUERY,
            agent_id=s.agent_id,
            session_id=s.session_id,
            had_taint=s.had_taint,
            had_export_attempt=s.had_export_attempt,
            had_failed_export=s.had_failed_export,
            had_escalation=s.had_escalation,
            started_at_ms=int(s.started_at * 1000),
        )

        accesses = [
            {
                "resource_id": a.resource_id,
                "canonical_confidence": a.canonical_confidence,
                "signal_type": a.signal_type.value,
                "seed_score": float(resource_scores.get(a.resource_id, 0.0)),
            }
            for a in s.accessed_resources
        ]
        if accesses:
            session.run(
                UPSERT_ACCESS_QUERY,
                session_id=s.session_id,
                accesses=accesses,
                flag_threshold=flag_threshold,
            )

    pairs = _adjacency_pairs(container_members)
    if pairs:
        session.run(
            ADJACENCY_QUERY,
            pairs=pairs,
            topology_factor=topology_factor,
        )


def _adjacency_pairs(
    container_members: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Ordered (source, target) resource pairs that share a container.

    Both directions are emitted (permutations, not combinations) so adjacency is
    symmetric; duplicates across overlapping containers are collapsed.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, str]] = []
    for members in container_members.values():
        uniq = list(dict.fromkeys(members))   # dedupe, preserve order
        for source, target in permutations(uniq, 2):
            key = (source, target)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"source": source, "target": target})
    return pairs

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

import logging
from itertools import permutations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axor_sentinel.sentinel.cycle import SessionSummary

log = logging.getLogger("axor.sentinel.graph")

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
#
# The same session is upserted again whenever a later cycle sees another record
# for it (e.g. ProbeTaintBridge's minimal drift summary, which carries no export
# flag and its own started_at).  Plain assignment let that later, thinner record
# CLEAR had_export_attempt / had_taint and move started_at — un-staging a
# slow-and-low pair after the fact.  The node is therefore a union of every
# record ever seen for the session, exactly like cycle._dedupe_sessions within
# one cycle: boolean facts OR together (a fact, once observed, stays observed)
# and started_at only moves earlier.
UPSERT_SESSION_QUERY = """
MERGE (ag:Agent {agent_id: $agent_id})
MERGE (s:Session {session_id: $session_id})
SET s.had_taint = coalesce(s.had_taint, false) OR $had_taint,
    s.had_export_attempt = coalesce(s.had_export_attempt, false) OR $had_export_attempt,
    s.had_failed_export = coalesce(s.had_failed_export, false) OR $had_failed_export,
    s.had_escalation = coalesce(s.had_escalation, false) OR $had_escalation,
    s.started_at = CASE
        WHEN s.started_at IS NULL OR $started_at_ms < s.started_at THEN $started_at_ms
        ELSE s.started_at
    END
MERGE (ag)-[:IN_SESSION]->(s)
"""

# Uniqueness constraints for every label the sentinel MERGEs or CREATEs by id.
# Without them MERGE is a label scan (O(nodes) per upsert, every cycle) and two
# concurrent writers can both miss and CREATE the same id twice — after which
# every MATCH on that id fans out over the duplicates and double-applies
# weights.  A uniqueness constraint is backed by an index, so it fixes both.
# ``IF NOT EXISTS`` makes them idempotent (safe on every start).
#
# Labels as written by this package: Resource.id (construct / queries),
# Session.session_id and Agent.agent_id (UPSERT_SESSION_QUERY), and
# Attestation.attestation_id (attestation.ATTEST_BRANCH_QUERY, whose comment
# already relied on "the unique constraint").  Containers are NOT nodes — they
# exist only as ADJACENT_TO edges derived from co-membership — so there is no
# Container constraint to declare.
SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT sentinel_resource_id IF NOT EXISTS "
    "FOR (r:Resource) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT sentinel_session_id IF NOT EXISTS "
    "FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT sentinel_agent_id IF NOT EXISTS "
    "FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE",
    "CREATE CONSTRAINT sentinel_attestation_id IF NOT EXISTS "
    "FOR (a:Attestation) REQUIRE a.attestation_id IS UNIQUE",
)


def ensure_schema(session: Any) -> bool:
    """Idempotently create the sentinel's uniqueness constraints.

    Each statement runs on its own (schema changes cannot share a transaction
    with each other on every Neo4j version, and one failure should not stop
    the rest).  A failure — a Neo4j older than 4.4 that does not understand
    ``IF NOT EXISTS`` / ``FOR … REQUIRE``, a read-only replica, existing
    duplicate data that violates the constraint, or a test double — is logged
    and swallowed: the constraints are a performance and race guard, and the
    cycle must still run without them (exactly as it did before they existed).

    Returns True when every statement was accepted.
    """
    ok = True
    for statement in SCHEMA_STATEMENTS:
        try:
            result = session.run(statement)
            # Drain the result where the driver returns one: with a lazy
            # driver the statement's error only surfaces on consume().
            consume = getattr(result, "consume", None)
            if callable(consume):
                consume()
        except Exception as exc:  # noqa: BLE001 — see docstring
            ok = False
            log.warning(
                "sentinel: could not ensure graph schema (%s): %s — continuing "
                "without it; MERGE falls back to label scans",
                statement.split(" IF NOT EXISTS")[0], exc,
            )
    return ok

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
            # Defensive: "" is "no resource" (graph.normalizer), never a node. The
            # producers already skip it; an upstream adapter that does not would
            # otherwise MERGE one Resource shared by every path-less call.
            if a.resource_id
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

    ``""`` containers and members are skipped: an empty id is "no resource", and a
    ``""`` container would make every path-less access adjacent to every other.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, str]] = []
    for container_id, members in container_members.items():
        if not container_id:
            continue
        uniq = [m for m in dict.fromkeys(members) if m]   # dedupe, preserve order
        for source, target in permutations(uniq, 2):
            key = (source, target)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"source": source, "target": target})
    return pairs

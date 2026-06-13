"""
Live-Neo4j integration tests for the Cypher in axor_sentinel.graph.queries.

The unit tests (test_cycle.py, test_fanout.py) drive a *mock* Neo4j that only
records query strings — they verify call ordering and parameters, but never
prove the Cypher itself is valid or behaves as intended.  These tests close
that gap by running the real query runner functions against a live Neo4j
instance and asserting on the resulting graph state.

They are gated behind the AXOR_TEST_NEO4J_BOLT environment variable and are
*correctly skipped* when it is unset (e.g. local runs without a database).
CI wires a neo4j service container and sets:

    AXOR_TEST_NEO4J_BOLT=bolt://localhost:7687

Optional auth (defaults to no-auth, matching NEO4J_AUTH=none):

    AXOR_TEST_NEO4J_USER
    AXOR_TEST_NEO4J_PASSWORD
"""
from __future__ import annotations

import os

import pytest

from axor_sentinel.graph.queries import (
    apply_decay,
    apply_fanout_weight,
    apply_hot_weight,
    slow_and_low_detection,
)

_BOLT = os.environ.get("AXOR_TEST_NEO4J_BOLT")

pytestmark = pytest.mark.skipif(
    not _BOLT,
    reason="AXOR_TEST_NEO4J_BOLT not set — live Neo4j Cypher tests skipped",
)

FLAG_THRESHOLD = 0.7
DAY_MS = 86_400_000


def _auth():
    user = os.environ.get("AXOR_TEST_NEO4J_USER")
    password = os.environ.get("AXOR_TEST_NEO4J_PASSWORD")
    if user and password:
        return (user, password)
    return None  # NEO4J_AUTH=none


@pytest.fixture()
def session():
    """A clean, isolated Neo4j session — graph is wiped before and after."""
    from neo4j import GraphDatabase  # imported lazily: optional [neo4j] extra

    driver = GraphDatabase.driver(_BOLT, auth=_auth())
    try:
        with driver.session() as sess:
            sess.run("MATCH (n) DETACH DELETE n")
            try:
                yield sess
            finally:
                sess.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()


def _score(session, rid: str) -> float:
    rec = session.run(
        "MATCH (r:Resource {id: $id}) RETURN r.suspicion_score AS s", id=rid
    ).single()
    assert rec is not None, f"resource {rid!r} not found"
    return rec["s"]


def _flagged(session, rid: str) -> bool:
    rec = session.run(
        "MATCH (r:Resource {id: $id}) RETURN r.flagged AS f", id=rid
    ).single()
    assert rec is not None, f"resource {rid!r} not found"
    return rec["f"]


# ── DECAY_QUERY ────────────────────────────────────────────────────────────────

class TestDecayQuery:
    def test_decay_halves_score_after_one_halflife(self, session) -> None:
        """30-day-old score must halve (half-life = 30 days) and clear flagged."""
        session.run(
            """
            CREATE (r:Resource {
                id: 'r_decay',
                suspicion_score: 0.8,
                canonical_confidence: 1.0,
                flagged: true,
                last_decay_at: timestamp() - $age_ms,
                last_signal_at: timestamp() - $age_ms
            })
            """,
            age_ms=30 * DAY_MS,
        )

        apply_decay(session, flag_threshold=FLAG_THRESHOLD)

        # 0.8 * 0.5^(30/30) = 0.4
        assert abs(_score(session, "r_decay") - 0.4) < 0.02
        # 0.4 < 0.7 → no longer flagged
        assert _flagged(session, "r_decay") is False

    def test_decay_skips_zero_score_resources(self, session) -> None:
        """Resources at score 0 are untouched (WHERE r.suspicion_score > 0)."""
        session.run(
            """
            CREATE (r:Resource {
                id: 'r_zero', suspicion_score: 0.0, canonical_confidence: 1.0,
                flagged: false, last_decay_at: timestamp(), last_signal_at: timestamp()
            })
            """
        )
        apply_decay(session, flag_threshold=FLAG_THRESHOLD)
        assert _score(session, "r_zero") == 0.0


# ── HOT_WEIGHT_QUERY ───────────────────────────────────────────────────────────

class TestHotWeightQuery:
    def test_hot_weight_accumulates_on_accessed_resource(self, session) -> None:
        """
        accumulate(score, eff_weight) = score + eff*(1-score), with
        eff = raw_weight * canonical_confidence.  From 0.0 with raw=0.6, conf=1.0
        the resulting score must be 0.6.
        """
        session.run(
            """
            CREATE (s:Session {session_id: 's_hot', had_taint: true})
            CREATE (r:Resource {
                id: 'r_hot', suspicion_score: 0.0, canonical_confidence: 1.0,
                flagged: false, last_decay_at: timestamp(), last_signal_at: timestamp()
            })
            CREATE (s)-[:ACCESSED {signal_type: 'READ_SUMMARIZE'}]->(r)
            """
        )

        apply_hot_weight(
            session,
            session_id="s_hot",
            signal_type="READ_SUMMARIZE",
            raw_weight=0.6,
            flag_threshold=FLAG_THRESHOLD,
        )

        assert abs(_score(session, "r_hot") - 0.6) < 1e-9

    def test_hot_weight_ignores_untainted_session(self, session) -> None:
        """The MATCH requires had_taint:true — untainted sessions are no-ops."""
        session.run(
            """
            CREATE (s:Session {session_id: 's_clean', had_taint: false})
            CREATE (r:Resource {
                id: 'r_clean', suspicion_score: 0.0, canonical_confidence: 1.0,
                flagged: false, last_decay_at: timestamp(), last_signal_at: timestamp()
            })
            CREATE (s)-[:ACCESSED {signal_type: 'READ'}]->(r)
            """
        )
        apply_hot_weight(
            session,
            session_id="s_clean",
            signal_type="READ",
            raw_weight=0.6,
            flag_threshold=FLAG_THRESHOLD,
        )
        assert _score(session, "r_clean") == 0.0


# ── FANOUT_WEIGHT_QUERY ────────────────────────────────────────────────────────

class TestFanoutWeightQuery:
    def test_fanout_boosts_batch_and_flags(self, session) -> None:
        """UNWIND applies the flat fanout weight to every listed resource."""
        session.run(
            """
            UNWIND ['f0', 'f1', 'f2'] AS rid
            CREATE (r:Resource {
                id: rid, suspicion_score: 0.5, canonical_confidence: 1.0,
                flagged: false, last_decay_at: timestamp(), last_signal_at: timestamp()
            })
            """
        )

        apply_fanout_weight(
            session,
            resource_ids=["f0", "f1", "f2"],
            fanout_weight=0.5,
            flag_threshold=FLAG_THRESHOLD,
        )

        # accumulate(0.5, 0.5) = 0.5 + 0.5*(1-0.5) = 0.75
        for rid in ("f0", "f1", "f2"):
            assert abs(_score(session, rid) - 0.75) < 1e-9
            assert _flagged(session, rid) is True  # 0.75 >= 0.7


# ── SLOW_AND_LOW_QUERY ─────────────────────────────────────────────────────────

class TestSlowAndLowDetection:
    def test_detects_staged_tainted_then_export(self, session) -> None:
        """
        Agent with a tainted session that touched a flagged resource, followed
        (after the min gap) by an export session, must surface in the report.
        """
        session.run(
            """
            CREATE (ag:Agent {agent_id: 'ag_stage'})
            CREATE (s1:Session {session_id: 's1', had_taint: true,
                                had_export_attempt: false, started_at: 0})
            CREATE (s2:Session {session_id: 's2', had_taint: false,
                                had_export_attempt: true, started_at: $gap})
            CREATE (r:Resource {id: 'r_stage', suspicion_score: 0.9, flagged: true,
                                canonical_confidence: 1.0,
                                last_decay_at: timestamp(), last_signal_at: timestamp()})
            CREATE (ag)-[:IN_SESSION]->(s1)
            CREATE (ag)-[:IN_SESSION]->(s2)
            CREATE (s1)-[:ACCESSED]->(r)
            """,
            gap=10 * DAY_MS,
        )

        rows = slow_and_low_detection(session, min_gap_ms=DAY_MS)

        assert len(rows) == 1
        row = rows[0]
        assert row["ag.agent_id"] == "ag_stage"
        assert row["flagged_resources"] == ["r_stage"]

    def test_no_detection_when_gap_too_small(self, session) -> None:
        """Sessions closer than min_gap_ms must not be reported."""
        session.run(
            """
            CREATE (ag:Agent {agent_id: 'ag_fast'})
            CREATE (s1:Session {session_id: 's1', had_taint: true,
                                had_export_attempt: false, started_at: 0})
            CREATE (s2:Session {session_id: 's2', had_taint: false,
                                had_export_attempt: true, started_at: $gap})
            CREATE (r:Resource {id: 'r_fast', suspicion_score: 0.9, flagged: true,
                                canonical_confidence: 1.0,
                                last_decay_at: timestamp(), last_signal_at: timestamp()})
            CREATE (ag)-[:IN_SESSION]->(s1)
            CREATE (ag)-[:IN_SESSION]->(s2)
            CREATE (s1)-[:ACCESSED]->(r)
            """,
            gap=1000,  # 1 second — well under a 1-day min gap
        )
        rows = slow_and_low_detection(session, min_gap_ms=DAY_MS)
        assert rows == []

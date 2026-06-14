"""Live-Neo4j integration tests for the graph queries.

These run only when a Neo4j is reachable — set ``AXOR_TEST_NEO4J_BOLT`` (e.g.
``bolt://127.0.0.1:7687``); otherwise they skip. They guard two things the
query-recording mock cannot: that every Cypher constant actually PARSES on a real
server (three write queries once shipped invalid `min(1.0, …)` aggregations), and
that the executed write path matches the in-memory `accumulate` scorer.

CI: add a `neo4j` service container and set AXOR_TEST_NEO4J_BOLT to exercise these.
"""
from __future__ import annotations

import logging
import os

import pytest

from axor_sentinel.graph import queries as q
from axor_sentinel.sentinel.weight import accumulate

# Skip the whole module unless the neo4j driver is installed.
pytest.importorskip("neo4j")

_BOLT = os.environ.get("AXOR_TEST_NEO4J_BOLT")

# Keep the driver's schema-notification chatter out of test output.
logging.getLogger("neo4j").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def session():
    if not _BOLT:
        pytest.skip("set AXOR_TEST_NEO4J_BOLT to run live-Neo4j tests")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(_BOLT)
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        driver.close()
        pytest.skip(f"no Neo4j at {_BOLT}: {exc}")
    s = driver.session()
    s.run("MATCH (n) DETACH DELETE n")
    yield s
    s.run("MATCH (n) DETACH DELETE n")
    s.close()
    driver.close()


_EXPLAIN_PARAMS = {
    "DECAY_QUERY": {"flag_threshold": 0.7},
    "HOT_WEIGHT_QUERY": {"session_id": "x", "signal_type": "read", "raw_weight": 0.4, "flag_threshold": 0.7},
    "CAUTION_ADJACENT_QUERY": {"session_id": "x", "flag_threshold": 0.7},
    "FANOUT_WEIGHT_QUERY": {"resource_ids": ["r1"], "fanout_weight": 0.5, "flag_threshold": 0.7},
    "SLOW_AND_LOW_QUERY": {"min_gap_ms": 0.0},
}


@pytest.mark.parametrize("name", list(_EXPLAIN_PARAMS))
def test_query_parses_on_real_neo4j(session, name):
    # EXPLAIN parses + plans without executing; a SyntaxError here is a shipped bug.
    session.run("EXPLAIN " + getattr(q, name), **_EXPLAIN_PARAMS[name]).consume()


def test_upsert_builds_graph_and_read_back_matches_accumulate(session):
    # The construction layer must build the graph the weight queries MATCH against,
    # and read_resource_scores must return the authoritative accumulated scores.
    from axor_sentinel.graph.model import SignalType
    from axor_sentinel.sentinel.weight import compute_hot_weight

    session.run("MATCH (n) DETACH DELETE n")
    accesses = [
        {"resource_id": "r1", "canonical_confidence": 1.0, "signal_type": SignalType.READ.value},
        {"resource_id": "r2", "canonical_confidence": 0.7, "signal_type": SignalType.READ_EXPORT_FAILED.value},
    ]
    q.upsert_session(
        session, agent_id="agentA", session_id="s1", had_taint=True,
        had_export_attempt=False, started_at=1000.0, accesses=accesses,
    )
    built = session.run(
        "MATCH (a:Agent)-[:IN_SESSION]->(:Session)-[:ACCESSED]->(r:Resource) RETURN count(r) AS c"
    ).single()["c"]
    assert built == 2

    q.apply_decay(session, flag_threshold=0.7)
    for acc in accesses:
        raw = compute_hot_weight(SignalType(acc["signal_type"]))
        q.apply_hot_weight(
            session, session_id="s1", signal_type=acc["signal_type"],
            raw_weight=raw, flag_threshold=0.7,
        )
    scores = q.read_resource_scores(session)
    assert scores["r1"] == pytest.approx(accumulate(0.0, compute_hot_weight(SignalType.READ) * 1.0))
    assert scores["r2"] == pytest.approx(accumulate(0.0, compute_hot_weight(SignalType.READ_EXPORT_FAILED) * 0.7))


class TestRunOnceLive:
    """run_once end-to-end against a real Neo4j: build graph -> write -> read back.
    These pin the score VALUES the snapshot carries (the mock-based cycle tests can't,
    since the mock does not execute Cypher)."""

    @staticmethod
    def _cycle(session, tmp_path, baselines=None):
        from pathlib import Path
        from axor_sentinel.sentinel.cycle import SentinelCycle
        session.run("MATCH (n) DETACH DELETE n")
        return SentinelCycle(session, Path(tmp_path), agent_baselines=baselines or {})

    @staticmethod
    def _sess(agent, resources, **kw):
        from axor_sentinel.sentinel.cycle import ResourceAccess, SessionSummary
        accesses = [ResourceAccess(resource_id=r, container_id=c, canonical_confidence=cf, signal_type=s)
                    for r, c, cf, s in resources]
        return SessionSummary(
            session_id=kw.get("session_id", f"s_{agent}"), agent_id=agent,
            started_at=1000.0, had_taint=kw.get("had_taint", True),
            had_export_attempt=False, had_failed_export=False, had_escalation=False,
            accessed_resources=accesses, taint_source=kw.get("taint_source", "mcp"),
        )

    def test_tainted_session_produces_positive_score(self, session, tmp_path):
        from axor_sentinel.graph.model import SignalType
        cyc = self._cycle(session, tmp_path)
        snap = cyc.run_once([self._sess("a", [("r1", "c1", 1.0, SignalType.READ)])])
        assert snap.resource_reputation["r1"] > 0.0

    def test_higher_signal_produces_higher_score(self, session, tmp_path):
        from axor_sentinel.graph.model import SignalType
        cyc = self._cycle(session, tmp_path)
        snap = cyc.run_once([
            self._sess("a", [("r1", "c1", 1.0, SignalType.READ),
                             ("r2", "c1", 1.0, SignalType.READ_EXPORT_FAILED)]),
        ])
        assert snap.resource_reputation["r2"] > snap.resource_reputation["r1"]

    def test_container_scores_from_readback(self, session, tmp_path):
        from axor_sentinel.graph.model import SignalType
        cyc = self._cycle(session, tmp_path)
        snap = cyc.run_once(
            [self._sess("a", [("r1", "c1", 1.0, SignalType.READ_EXPORT_FAILED)])],
            container_members={"c1": ["r1"]},
        )
        assert snap.container_reputation["c1"] > 0.0

    def test_fanout_boosts_score(self, session, tmp_path):
        from axor_sentinel.graph.model import SignalType
        from axor_sentinel.sentinel.cycle import FANOUT_MIN_SESSIONS
        from axor_sentinel.sentinel.events import AgentContainerBaseline
        agent = "fan"
        baseline = AgentContainerBaseline(
            agent_id=agent, mean_containers_per_session=1.0, std_containers_per_session=0.5,
            session_count=FANOUT_MIN_SESSIONS, last_updated=0.0,
        )
        # baseline-less run: hot weight only
        plain = self._cycle(session, tmp_path / "plain")
        snap_plain = plain.run_once([self._sess(agent, [("r0", "c0", 1.0, SignalType.READ_SUMMARIZE)])])
        # fanout run: 10 containers -> z >> 2.5 -> fanout boost on top of hot weight
        fan = self._cycle(session, tmp_path / "fan", baselines={agent: baseline})
        res = [(f"r{i}", f"c{i}", 1.0, SignalType.READ_SUMMARIZE) for i in range(10)]
        snap_fan = fan.run_once([self._sess(agent, res)])
        assert snap_fan.resource_reputation["r0"] > snap_plain.resource_reputation["r0"]


def test_score_clamp_is_two_sided(session):
    # The CASE clamp must bound [0, 1] on its own, not rely on inputs being
    # non-negative: a negative weight clamps to 0.0, an over-1 weight to 1.0.
    session.run("MATCH (n) DETACH DELETE n")
    session.run(
        "MERGE (r:Resource {id:'r1'}) "
        "SET r.suspicion_score=0.5, r.canonical_confidence=1.0, r.last_decay_at=timestamp() "
        "MERGE (sn:Session {session_id:'s1', had_taint:true}) "
        "MERGE (sn)-[:ACCESSED {signal_type:'read'}]->(r)"
    )
    q.apply_hot_weight(session, session_id="s1", signal_type="read", raw_weight=-10.0, flag_threshold=0.7)
    assert session.run("MATCH (r:Resource {id:'r1'}) RETURN r.suspicion_score AS sc").single()["sc"] == 0.0
    session.run("MATCH (r:Resource {id:'r1'}) SET r.suspicion_score=0.99")
    q.apply_hot_weight(session, session_id="s1", signal_type="read", raw_weight=5.0, flag_threshold=0.7)
    assert session.run("MATCH (r:Resource {id:'r1'}) RETURN r.suspicion_score AS sc").single()["sc"] == 1.0


def test_hot_weight_and_fanout_match_python_accumulate(session):
    session.run("MATCH (n) DETACH DELETE n")
    session.run(
        "MERGE (r:Resource {id:'r1'}) "
        "SET r.suspicion_score=0.0, r.canonical_confidence=1.0, r.last_decay_at=timestamp() "
        "MERGE (sn:Session {session_id:'s1', had_taint:true}) "
        "MERGE (sn)-[:ACCESSED {signal_type:'read'}]->(r)"
    )

    q.apply_hot_weight(session, session_id="s1", signal_type="read", raw_weight=0.4, flag_threshold=0.7)
    sc1 = session.run("MATCH (r:Resource {id:'r1'}) RETURN r.suspicion_score AS sc").single()["sc"]
    assert sc1 == pytest.approx(accumulate(0.0, 0.4))

    q.apply_hot_weight(session, session_id="s1", signal_type="read", raw_weight=0.4, flag_threshold=0.7)
    q.apply_fanout_weight(session, resource_ids=["r1"], fanout_weight=0.5, flag_threshold=0.7)
    row = session.run(
        "MATCH (r:Resource {id:'r1'}) RETURN r.suspicion_score AS sc, r.flagged AS f"
    ).single()
    assert row["sc"] == pytest.approx(accumulate(accumulate(accumulate(0.0, 0.4), 0.4), 0.5))
    assert row["f"] is True   # 0.82 >= FLAG_THRESHOLD 0.7

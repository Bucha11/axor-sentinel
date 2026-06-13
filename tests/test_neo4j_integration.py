"""Live-Neo4j integration tests for axor_sentinel.graph.queries.

The unit tests (test_cycle.py, test_fanout.py) drive a *mock* Neo4j that only
records query strings — they verify call ordering and parameters, but never
prove the Cypher itself parses or behaves as intended.  These tests close that
gap by running the real query runner functions against a live Neo4j and
asserting on the resulting graph state.  They guard three things the
query-recording mock cannot:

  1. every Cypher constant actually PARSES on a real server (three write queries
     once shipped invalid ``min(1.0, …)`` aggregations — a SyntaxError on Neo4j);
  2. the executed write path matches the in-memory ``accumulate`` scorer;
  3. decay / hot-weight / fanout / slow-and-low produce the expected scores
     and flags end to end.

They run only when a Neo4j is reachable: set ``AXOR_TEST_NEO4J_BOLT`` (e.g.
``bolt://localhost:7687``); otherwise the whole module is skipped.  Auth
defaults to none (matching ``NEO4J_AUTH=none``); set ``AXOR_TEST_NEO4J_USER``
and ``AXOR_TEST_NEO4J_PASSWORD`` to connect with credentials.

CI wires a neo4j service container and sets AXOR_TEST_NEO4J_BOLT to exercise
these.
"""
from __future__ import annotations

import logging
import os

import pytest

import time

from axor_sentinel.graph import construct, queries as q
from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.cycle import ResourceAccess, SentinelCycle, SessionSummary
from axor_sentinel.sentinel.weight import accumulate

# Skip the whole module unless the neo4j driver is installed.
pytest.importorskip("neo4j")

_BOLT = os.environ.get("AXOR_TEST_NEO4J_BOLT")

# Keep the driver's schema-notification chatter out of test output.
logging.getLogger("neo4j").setLevel(logging.ERROR)

FLAG_THRESHOLD = 0.7
DAY_MS = 86_400_000


def _auth():
    user = os.environ.get("AXOR_TEST_NEO4J_USER")
    password = os.environ.get("AXOR_TEST_NEO4J_PASSWORD")
    if user and password:
        return (user, password)
    return None  # NEO4J_AUTH=none


@pytest.fixture(scope="module")
def driver():
    """Module-scoped driver — skips gracefully when no Neo4j is reachable."""
    if not _BOLT:
        pytest.skip("set AXOR_TEST_NEO4J_BOLT to run live-Neo4j tests")
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(_BOLT, auth=_auth())
    try:
        drv.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        drv.close()
        pytest.skip(f"no Neo4j at {_BOLT}: {exc}")
    yield drv
    drv.close()


@pytest.fixture()
def session(driver):
    """Per-test session on a wiped graph — isolates the write-path tests."""
    with driver.session() as sess:
        sess.run("MATCH (n) DETACH DELETE n")
        try:
            yield sess
        finally:
            sess.run("MATCH (n) DETACH DELETE n")


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


# ── Cypher parses on a real server (cheap coverage of every constant) ──────────

_EXPLAIN_PARAMS = {
    "DECAY_QUERY": {"flag_threshold": 0.7},
    "HOT_WEIGHT_QUERY": {
        "session_id": "x", "signal_type": "read", "resource_id": "r1",
        "raw_weight": 0.4, "flag_threshold": 0.7,
    },
    "RESOURCE_SCORES_QUERY": {},
    "CAUTION_ADJACENT_QUERY": {"session_id": "x", "flag_threshold": 0.7},
    "FANOUT_WEIGHT_QUERY": {
        "resource_ids": ["r1"], "fanout_weight": 0.5, "flag_threshold": 0.7,
    },
    "SLOW_AND_LOW_QUERY": {"min_gap_ms": 0.0},
}


@pytest.mark.parametrize("name", list(_EXPLAIN_PARAMS))
def test_query_parses_on_real_neo4j(session, name) -> None:
    # EXPLAIN parses + plans without executing; a SyntaxError here is a shipped bug.
    session.run("EXPLAIN " + getattr(q, name), **_EXPLAIN_PARAMS[name]).consume()


# ── Executed write path matches the Python accumulate scorer ───────────────────

def test_hot_weight_and_fanout_match_python_accumulate(session) -> None:
    session.run(
        "MERGE (r:Resource {id:'r1'}) "
        "SET r.suspicion_score=0.0, r.canonical_confidence=1.0, r.last_decay_at=timestamp() "
        "MERGE (sn:Session {session_id:'s1', had_taint:true}) "
        "MERGE (sn)-[:ACCESSED {signal_type:'read'}]->(r)"
    )

    before, after = q.apply_hot_weight(
        session, session_id="s1", signal_type="read",
        raw_weight=0.4, flag_threshold=FLAG_THRESHOLD, resource_id="r1",
    )
    assert (before, after) == pytest.approx((0.0, accumulate(0.0, 0.4)))
    assert _score(session, "r1") == pytest.approx(accumulate(0.0, 0.4))

    q.apply_hot_weight(
        session, session_id="s1", signal_type="read",
        raw_weight=0.4, flag_threshold=FLAG_THRESHOLD, resource_id="r1",
    )
    q.apply_fanout_weight(
        session, resource_ids=["r1"], fanout_weight=0.5, flag_threshold=FLAG_THRESHOLD,
    )
    expected = accumulate(accumulate(accumulate(0.0, 0.4), 0.4), 0.5)
    assert _score(session, "r1") == pytest.approx(expected)
    assert _flagged(session, "r1") is True   # ~0.82 >= FLAG_THRESHOLD


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

        q.apply_decay(session, flag_threshold=FLAG_THRESHOLD)

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
        q.apply_decay(session, flag_threshold=FLAG_THRESHOLD)
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

        q.apply_hot_weight(
            session,
            session_id="s_hot",
            signal_type="READ_SUMMARIZE",
            raw_weight=0.6,
            flag_threshold=FLAG_THRESHOLD,
            resource_id="r_hot",
        )

        assert _score(session, "r_hot") == pytest.approx(0.6)

    def test_hot_weight_targets_only_its_resource(self, session) -> None:
        """Two resources share one signal_type; each call must hit only its own.

        Before the id filter, the signal_type-wide MATCH double-counted every
        sibling on each per-resource call — now a live bug since scores are read
        back into the snapshot.
        """
        session.run(
            """
            CREATE (s:Session {session_id: 's_multi', had_taint: true})
            CREATE (a:Resource {id: 'r_a', suspicion_score: 0.0, canonical_confidence: 1.0,
                                flagged: false, last_decay_at: timestamp(), last_signal_at: timestamp()})
            CREATE (b:Resource {id: 'r_b', suspicion_score: 0.0, canonical_confidence: 1.0,
                                flagged: false, last_decay_at: timestamp(), last_signal_at: timestamp()})
            CREATE (s)-[:ACCESSED {signal_type: 'READ'}]->(a)
            CREATE (s)-[:ACCESSED {signal_type: 'READ'}]->(b)
            """
        )
        q.apply_hot_weight(
            session, session_id="s_multi", signal_type="READ",
            raw_weight=0.6, flag_threshold=FLAG_THRESHOLD, resource_id="r_a",
        )
        # r_a got the weight exactly once; r_b (same signal_type) is untouched.
        assert _score(session, "r_a") == pytest.approx(0.6)
        assert _score(session, "r_b") == 0.0

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
        result = q.apply_hot_weight(
            session,
            session_id="s_clean",
            signal_type="READ",
            raw_weight=0.6,
            flag_threshold=FLAG_THRESHOLD,
            resource_id="r_clean",
        )
        assert result is None   # no match → nothing applied
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

        q.apply_fanout_weight(
            session,
            resource_ids=["f0", "f1", "f2"],
            fanout_weight=0.5,
            flag_threshold=FLAG_THRESHOLD,
        )

        # accumulate(0.5, 0.5) = 0.5 + 0.5*(1-0.5) = 0.75
        for rid in ("f0", "f1", "f2"):
            assert _score(session, rid) == pytest.approx(0.75)
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

        rows = q.slow_and_low_detection(session, min_gap_ms=DAY_MS)

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
        rows = q.slow_and_low_detection(session, min_gap_ms=DAY_MS)
        assert rows == []


# ── Graph construction: the cycle now populates a real graph ───────────────────

def _count(session, cypher: str) -> int:
    return session.run(cypher).single()[0]


def _access(rid: str, cid: str, sig: SignalType, conf: float = 1.0) -> ResourceAccess:
    return ResourceAccess(
        resource_id=rid, container_id=cid, canonical_confidence=conf, signal_type=sig
    )


def _summary(sid: str, agent: str, accesses, **kw) -> SessionSummary:
    base = dict(
        started_at=time.time(), had_taint=True, had_export_attempt=False,
        had_failed_export=False, had_escalation=False,
    )
    base.update(kw)
    return SessionSummary(
        session_id=sid, agent_id=agent, accessed_resources=list(accesses), **base
    )


class TestGraphConstruction:
    def test_upsert_creates_nodes_and_edges(self, session) -> None:
        """upsert_graph materialises Agent/Session/Resource + ACCESSED/IN_SESSION."""
        sess = _summary("cs1", "a1", [
            _access("r1", "c1", SignalType.READ),
            _access("r2", "c1", SignalType.READ_SUMMARIZE),
        ])
        construct.upsert_graph(
            session, [sess], resource_scores={},
            container_members={"c1": ["r1", "r2"]}, flag_threshold=FLAG_THRESHOLD,
        )

        assert _count(session, "MATCH (a:Agent {agent_id:'a1'}) RETURN count(a)") == 1
        assert _count(session, "MATCH (s:Session {session_id:'cs1'}) RETURN count(s)") == 1
        assert _count(session, "MATCH (:Resource) RETURN count(*)") == 2
        assert _count(
            session, "MATCH (:Agent)-[:IN_SESSION]->(:Session) RETURN count(*)"
        ) == 1
        assert _count(
            session, "MATCH (:Session)-[:ACCESSED]->(:Resource) RETURN count(*)"
        ) == 2
        # ADJACENT_TO is symmetric between the two same-container resources.
        assert _count(
            session, "MATCH (:Resource)-[:ADJACENT_TO]->(:Resource) RETURN count(*)"
        ) == 2
        # canonical_confidence is set so the hot-weight Cypher matches Python.
        rec = session.run(
            "MATCH (r:Resource {id:'r2'}) RETURN r.canonical_confidence AS c"
        ).single()
        assert rec["c"] == pytest.approx(1.0)

    def test_adjacency_makes_caution_propagate(self, session) -> None:
        """A resource adjacent to a hot one (but not accessed) gets caution score."""
        # r_old exists from a prior session so it can be an un-accessed neighbour.
        old = _summary("s_old", "a1", [_access("r_old", "c1", SignalType.READ)],
                       had_taint=False)
        new = _summary("s_new", "a1", [_access("r_hot", "c1", SignalType.READ_SUMMARIZE)])
        construct.upsert_graph(
            session, [old, new], resource_scores={},
            container_members={"c1": ["r_hot", "r_old"]}, flag_threshold=FLAG_THRESHOLD,
        )

        # Hot weight on the accessed resource, then caution on its neighbours.
        q.apply_hot_weight(
            session, session_id="s_new", signal_type="read_summarize",
            raw_weight=0.6, flag_threshold=FLAG_THRESHOLD, resource_id="r_hot",
        )
        q.apply_caution_adjacent(session, session_id="s_new", flag_threshold=FLAG_THRESHOLD)

        # r_old was not accessed by s_new but is ADJACENT_TO r_hot → caution > 0.
        assert _score(session, "r_old") > 0.0

    def test_full_cycle_enables_slow_and_low(self, session, tmp_path) -> None:
        """A tainted-then-export staging pattern surfaces after a real cycle."""
        cycle = SentinelCycle(session, tmp_path, agent_baselines={})
        now = time.time()
        # Tainted session flags r1 (READ_EXPORT_FAILED → score 1.0).
        tainted = _summary(
            "s_taint", "a1", [_access("r1", "c1", SignalType.READ_EXPORT_FAILED)],
            started_at=now,
        )
        # Later export session for the same agent (untainted — still materialised).
        export = _summary(
            "s_export", "a1", [], started_at=now + 5 * 86400,  # +5 days
            had_taint=False, had_export_attempt=True,
        )

        cycle.run_once([tainted, export])

        assert _flagged(session, "r1") is True
        rows = q.slow_and_low_detection(session, min_gap_ms=DAY_MS)
        assert len(rows) == 1
        assert rows[0]["ag.agent_id"] == "a1"
        assert rows[0]["flagged_resources"] == ["r1"]


# ── Full cycle: the snapshot is read back from Neo4j (authoritative store) ──────

class TestFullCycleSnapshot:
    """run_once builds the snapshot from Neo4j, not an in-memory re-accumulation.

    These replace the old mock-based snapshot-score tests in test_cycle.py: with
    Neo4j authoritative, only a live server can assert the resulting magnitudes.
    """

    def test_snapshot_reflects_neo4j_score(self, session, tmp_path) -> None:
        """The snapshot score for a resource equals its Neo4j suspicion_score."""
        cycle = SentinelCycle(session, tmp_path, agent_baselines={})
        sess = _summary("cs1", "a1", [_access("r1", "c1", SignalType.READ_SUMMARIZE)])

        snap = cycle.run_once([sess])

        assert snap.resource_reputation["r1"] > 0.0
        assert snap.resource_reputation["r1"] == pytest.approx(_score(session, "r1"))

    def test_untainted_session_leaves_score_at_zero(self, session, tmp_path) -> None:
        """An untainted session creates the node but applies no weight → absent (0)."""
        cycle = SentinelCycle(session, tmp_path, agent_baselines={})
        sess = _summary("cs1", "a1", [_access("r1", "c1", SignalType.READ)],
                        had_taint=False)

        snap = cycle.run_once([sess])

        # Score 0 → excluded from the read-back, so the consumer sees 0.
        assert snap.resource_reputation.get("r1", 0.0) == 0.0

    def test_higher_signal_type_produces_higher_score(self, session, tmp_path) -> None:
        """READ_EXPORT_FAILED accrues more than READ-only across two cycles."""
        c_read = SentinelCycle(session, tmp_path / "read", agent_baselines={})
        snap_read = c_read.run_once(
            [_summary("s_r", "a", [_access("rr", "c", SignalType.READ)])]
        )
        session.run("MATCH (n) DETACH DELETE n")   # isolate the two cycles
        c_fail = SentinelCycle(session, tmp_path / "fail", agent_baselines={})
        snap_fail = c_fail.run_once(
            [_summary("s_f", "a", [_access("rr", "c", SignalType.READ_EXPORT_FAILED)])]
        )

        assert snap_fail.resource_reputation["rr"] > snap_read.resource_reputation["rr"]

    def test_container_scores_populated(self, session, tmp_path) -> None:
        """Container aggregation is computed from the read-back resource scores."""
        cycle = SentinelCycle(session, tmp_path, agent_baselines={})
        sess = _summary("cs1", "a1", [_access("r1", "c1", SignalType.READ_EXPORT_FAILED)])

        snap = cycle.run_once([sess], container_members={"c1": ["r1"]})

        assert snap.container_reputation["c1"] > 0.0

    def test_caution_is_reflected_in_snapshot(self, session, tmp_path) -> None:
        """The key win: caution (graph-only before) now lands in the snapshot.

        Cycle 1 seeds an un-accessed neighbour; cycle 2 accesses its same-container
        sibling in a tainted session, so caution flows to the neighbour AND shows up
        in the served snapshot — previously invisible because the snapshot came from
        the Python path, which never computed caution.
        """
        cycle = SentinelCycle(session, tmp_path, agent_baselines={})
        members = {"c1": ["r_hot", "r_quiet"]}
        # Cycle 1: r_quiet becomes a known node (untainted, stays at 0).
        cycle.run_once(
            [_summary("s1", "a1", [_access("r_quiet", "c1", SignalType.READ)],
                      had_taint=False)],
            container_members=members,
        )
        # Cycle 2: tainted access to r_hot; r_quiet is adjacent but not accessed.
        snap = cycle.run_once(
            [_summary("s2", "a1", [_access("r_hot", "c1", SignalType.READ_SUMMARIZE)])],
            container_members=members,
        )

        assert snap.resource_reputation["r_hot"] > 0.0
        # Caution reached the neighbour and is visible in the snapshot.
        assert snap.resource_reputation.get("r_quiet", 0.0) > 0.0
        assert snap.resource_reputation["r_quiet"] == pytest.approx(
            _score(session, "r_quiet")
        )

    def test_fanout_boost_in_snapshot(self, session, tmp_path) -> None:
        """A fired fanout adds FANOUT_WEIGHT on top of the hot weight in the snapshot."""
        from axor_sentinel.sentinel.events import AgentContainerBaseline
        from axor_sentinel.sentinel.cycle import FANOUT_MIN_SESSIONS

        agent = "fan"
        baseline = AgentContainerBaseline(
            agent_id=agent, mean_containers_per_session=1.0,
            std_containers_per_session=0.5, session_count=FANOUT_MIN_SESSIONS,
            last_updated=0.0,
        )
        cycle = SentinelCycle(session, tmp_path, agent_baselines={agent: baseline})
        # 10 unique containers → z = (10 - 1.0) / 0.5 = 18 >> 2.5 → fanout fires.
        accesses = [_access(f"r{i}", f"c{i}", SignalType.READ_SUMMARIZE) for i in range(10)]
        snap = cycle.run_once([_summary("s_fan", agent, accesses)])

        # READ_SUMMARIZE hot weight alone ≈ 0.6; with fanout 0.5 on top → ≈ 0.8.
        assert snap.resource_reputation["r0"] > 0.6

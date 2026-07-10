"""
Tests for SentinelCycle:
  - End-to-end run_once with mock Neo4j: hot weights accumulate in snapshot
  - Fanout weight appears in snapshot (invariant A-10)
  - update_baseline: exponential smoothing, sample variance, cold-start guard
  - State persistence: save_state / load_state round-trip
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.cycle import (

    FANOUT_WEIGHT,
    ResourceAccess,
    SentinelCycle,
    SessionSummary,
)
from axor_sentinel.sentinel.events import AgentContainerBaseline



# ── Mock Neo4j session ────────────────────────────────────────────────────────

class _MockResult:
    """Minimal stand-in for a neo4j Result: empty by default.

    The cycle reads `.single()` off the hot-weight write and iterates the
    score read-back; an empty result means no events and an empty snapshot, which
    is exactly what these structure/logic tests want — score behaviour is asserted
    against a live Neo4j in tests/test_neo4j_integration.py instead.
    """

    def single(self):
        return None

    def __iter__(self):
        return iter(())


class _MockNeo4j:
    """Records every query + params for assertion in tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params) -> _MockResult:  # noqa: ANN001
        self.calls.append((query, params))
        return _MockResult()

    def was_called_with(self, fragment: str) -> bool:
        return any(fragment in q for q, _ in self.calls)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session(
    agent_id: str,
    resources: list[tuple[str, str, float, SignalType]],   # (rid, cid, conf, sig)
    *,
    had_taint: bool = True,
    had_export: bool = False,
    had_failed: bool = False,
    taint_source: str = "mcp",
    source_class: str = "",
) -> SessionSummary:
    accesses = [
        ResourceAccess(
            resource_id=rid,
            container_id=cid,
            canonical_confidence=conf,
            signal_type=sig,
        )
        for rid, cid, conf, sig in resources
    ]
    return SessionSummary(
        session_id=f"s_{agent_id}_{len(accesses)}",
        agent_id=agent_id,
        started_at=time.time(),
        had_taint=had_taint,
        had_export_attempt=had_export,
        had_failed_export=had_failed,
        had_escalation=False,
        accessed_resources=accesses,
        taint_source=taint_source,
        source_class=source_class,
    )


def _cycle(tmp_path: Path, baselines: dict | None = None) -> tuple[SentinelCycle, _MockNeo4j]:
    neo4j = _MockNeo4j()
    cycle = SentinelCycle(
        neo4j,
        tmp_path,
        agent_baselines=baselines or {},   # explicit → skip disk load
    )
    return cycle, neo4j


# ── End-to-end hot weight accumulation ───────────────────────────────────────

class TestRunOnceHotWeights:
    # NOTE: the actual scores now live in Neo4j and the snapshot is read back from
    # it, so score-magnitude assertions (tainted → positive, higher signal → higher
    # score, container aggregation, untainted → zero) run against a live Neo4j in
    # tests/test_neo4j_integration.py::TestFullCycleSnapshot. These mock-level tests
    # cover the call structure the cycle must always issue.

    def test_tainted_session_issues_hot_weight_write(self, tmp_path: Path) -> None:
        """A tainted access must drive a hot-weight write targeting that resource."""
        cycle, neo4j = _cycle(tmp_path)
        sess = _session("agent1", [("r1", "c1", 1.0, SignalType.READ)])
        cycle.run_once([sess])

        hot = [p for q, p in neo4j.calls if "had_taint" in q and "$raw_weight" in q]
        assert any(p.get("resource_id") == "r1" for p in hot)

    def test_untainted_session_issues_no_hot_weight_write(self, tmp_path: Path) -> None:
        """Sessions without taint must not issue any hot-weight write."""
        cycle, neo4j = _cycle(tmp_path)
        sess = _session("agent1", [("r1", "c1", 1.0, SignalType.READ)], had_taint=False)
        cycle.run_once([sess])

        hot = [p for q, p in neo4j.calls if "had_taint" in q and "$raw_weight" in q]
        assert hot == []

    def test_snapshot_read_back_from_neo4j(self, tmp_path: Path) -> None:
        """The snapshot resources come from the read-back query (empty under mock)."""
        cycle, neo4j = _cycle(tmp_path)
        sess = _session("agent1", [("r1", "c1", 1.0, SignalType.READ)])
        snap = cycle.run_once([sess])

        assert any("RETURN r.id AS id" in q for q, _ in neo4j.calls)
        # Mock read-back yields no rows → snapshot reflects exactly that.
        assert snap.resource_reputation == {}

    def test_snapshot_is_frozen(self, tmp_path: Path) -> None:
        """ReputationSnapshot must be immutable (frozen=True)."""
        cycle, _ = _cycle(tmp_path)
        snap = cycle.run_once([])
        with pytest.raises((AttributeError, TypeError)):
            snap.version = 999  # type: ignore[misc]

    def test_decay_query_runs_first(self, tmp_path: Path) -> None:
        """DECAY_QUERY must appear before HOT_WEIGHT_QUERY in the call log (invariant A-4)."""
        cycle, neo4j = _cycle(tmp_path)
        sess = _session("a", [("r1", "c1", 1.0, SignalType.READ)])
        cycle.run_once([sess])

        query_names = [q for q, _ in neo4j.calls]
        decay_idx = next((i for i, q in enumerate(query_names) if "last_decay_at" in q and "suspicion_score" in q and "0.5" in q), None)
        hot_idx = next((i for i, q in enumerate(query_names) if "last_signal_at" in q and "had_taint" in q), None)
        assert decay_idx is not None, "DECAY_QUERY not found in Neo4j calls"
        assert hot_idx is not None, "HOT_WEIGHT_QUERY not found in Neo4j calls"
        assert decay_idx < hot_idx, "Decay must run before hot weights (invariant A-4)"


# ── Fanout weight in snapshot (invariant A-10) ────────────────────────────────

class TestFanoutWeightInSnapshot:
    def _make_baseline(self, agent_id: str) -> AgentContainerBaseline:
        return AgentContainerBaseline(
            agent_id=agent_id,
            mean_containers_per_session=1.0,
            std_containers_per_session=0.5,
            session_count=10,
            last_updated=0.0,
        )

    # The fanout score magnitude in the snapshot is asserted against a live Neo4j in
    # tests/test_neo4j_integration.py::TestFullCycleSnapshot.test_fanout_boost_in_snapshot.

    def test_fanout_weight_written_to_neo4j(self, tmp_path: Path) -> None:
        """FANOUT_WEIGHT_QUERY must be sent to Neo4j when a fanout fires (invariant A-10)."""
        agent = "fanout_neo4j"
        baseline = self._make_baseline(agent)

        cycle, neo4j = _cycle(tmp_path, baselines={agent: baseline})
        resources = [
            (f"r{i}", f"c{i}", 1.0, SignalType.READ_SUMMARIZE)
            for i in range(10)
        ]
        cycle.run_once([_session(agent, resources)])

        # FANOUT_WEIGHT_QUERY uses UNWIND + fanout_weight param
        fanout_calls = [
            (q, p) for q, p in neo4j.calls
            if "UNWIND" in q and "fanout_weight" in p
        ]
        assert len(fanout_calls) >= 1, (
            "Expected at least one FANOUT_WEIGHT_QUERY call to Neo4j"
        )
        assert fanout_calls[0][1]["fanout_weight"] == FANOUT_WEIGHT


# ── update_baseline ───────────────────────────────────────────────────────────

class TestUpdateBaseline:
    def _make_sessions(self, n_containers_per_session: list[int]) -> list[SessionSummary]:
        sessions = []
        for i, n in enumerate(n_containers_per_session):
            accesses = [
                ResourceAccess(f"r{j}", f"c{j}", 1.0, SignalType.READ)
                for j in range(n)
            ]
            sessions.append(SessionSummary(
                session_id=f"s{i}",
                agent_id="agt",
                started_at=float(i),
                had_taint=False,
                had_export_attempt=False,
                had_failed_export=False,
                had_escalation=False,
                accessed_resources=accesses,
            ))
        return sessions

    def test_baseline_computed_correctly(self, tmp_path: Path) -> None:
        """Mean and std must match manually-computed values for a known input."""
        cycle, _ = _cycle(tmp_path)
        counts = [2, 4, 6]  # mean=4, sample std=2
        sessions = self._make_sessions(counts)
        cycle.update_baseline("agt", sessions)

        bl = cycle._baselines["agt"]
        assert abs(bl.mean_containers_per_session - 4.0) < 1e-9
        # Sample std: sqrt(((2-4)^2 + (4-4)^2 + (6-4)^2) / (3-1)) = sqrt(8/2) = 2
        assert abs(bl.std_containers_per_session - 2.0) < 1e-9

    def test_fewer_than_2_sessions_no_update(self, tmp_path: Path) -> None:
        """update_baseline must be a no-op when fewer than 2 sessions provided."""
        cycle, _ = _cycle(tmp_path)
        cycle.update_baseline("agt", [])
        assert "agt" not in cycle._baselines

        cycle.update_baseline("agt", self._make_sessions([3]))
        assert "agt" not in cycle._baselines

    def test_exponential_smoothing_applied(self, tmp_path: Path) -> None:
        """Second call blends new stats with existing baseline via alpha=0.3."""
        cycle, _ = _cycle(tmp_path)
        # Seed baseline
        cycle.update_baseline("agt", self._make_sessions([2, 2, 2]))
        old_mean = cycle._baselines["agt"].mean_containers_per_session  # ≈ 2.0

        # New data with higher mean
        cycle.update_baseline("agt", self._make_sessions([10, 10, 10]))
        new_mean = cycle._baselines["agt"].mean_containers_per_session

        # new_mean should be between old and 10 (smoothed)
        assert old_mean < new_mean < 10.0

    def test_session_count_set_to_window_size(self, tmp_path: Path) -> None:
        """session_count must equal the number of sessions used to compute baseline."""
        cycle, _ = _cycle(tmp_path)
        sessions = self._make_sessions([1, 2, 3, 4, 5])
        cycle.update_baseline("agt", sessions)
        assert cycle._baselines["agt"].session_count == 5


# ── State persistence ─────────────────────────────────────────────────────────

class TestStatePersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """save_state / load_state must round-trip signal_history, prior_counts, baselines."""
        cycle, _ = _cycle(tmp_path)

        # Populate some state
        cycle._signal_history["r1"] = ["mcp", "web", "mcp"]
        cycle._prior_counts[("r1", "mcp")] = 2
        cycle._prior_counts[("r1", "web")] = 1
        cycle._baselines["agent_x"] = AgentContainerBaseline(
            agent_id="agent_x",
            mean_containers_per_session=3.5,
            std_containers_per_session=1.2,
            session_count=15,
            last_updated=9999.0,
        )
        cycle._current_version = 7
        cycle.save_state()

        sh, pc, bl, ver, ev = SentinelCycle.load_state(tmp_path / "sentinel_state.json")

        assert sh["r1"] == ["mcp", "web", "mcp"]
        assert pc[("r1", "mcp")] == 2
        assert pc[("r1", "web")] == 1
        assert "agent_x" in bl
        assert bl["agent_x"].session_count == 15
        assert abs(bl["agent_x"].mean_containers_per_session - 3.5) < 1e-9
        assert ver == 7

    def test_missing_state_file_returns_empty(self, tmp_path: Path) -> None:
        """load_state on a non-existent file must return empty dicts and version 0."""
        sh, pc, bl, ver, ev = SentinelCycle.load_state(tmp_path / "nonexistent.json")
        assert sh == {}
        assert pc == {}
        assert bl == {}
        assert ver == 0

    def test_corrupt_state_file_returns_empty(self, tmp_path: Path) -> None:
        """Corrupt JSON must not raise — returns empty state."""
        bad_file = tmp_path / "sentinel_state.json"
        bad_file.write_text("{broken json", encoding="utf-8")
        sh, pc, bl, ver, ev = SentinelCycle.load_state(bad_file)
        assert sh == {}
        assert ver == 0

    def test_run_once_persists_state(self, tmp_path: Path) -> None:
        """run_once must call save_state so sentinel_state.json is written."""
        cycle, _ = _cycle(tmp_path)
        cycle.run_once([])
        assert (tmp_path / "sentinel_state.json").exists()

    def test_init_loads_persisted_state(self, tmp_path: Path) -> None:
        """A new SentinelCycle (no explicit baselines) must load state from disk."""
        # Write state manually
        from axor_sentinel.sentinel.cycle import SentinelCycle as SC
        import json

        state = {
            "version": 5,
            "signal_history": {"r99": ["mcp"]},
            "prior_counts": {"r99\x00mcp": 1},
            "baselines": {},
        }
        (tmp_path / "sentinel_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )

        # New cycle with no explicit state → should load from disk
        from unittest.mock import MagicMock
        neo4j = MagicMock()
        neo4j.run = lambda *a, **kw: None

        cycle2 = SC(neo4j, tmp_path)   # no explicit baselines/signal_history
        assert cycle2._signal_history.get("r99") == ["mcp"]
        assert cycle2._current_version == 5


class TestPoisoningMitigationKeying:
    """F1: dampening/diversity key on the actor identity, not the attacker-controllable
    taint_source label — rotating the label must not reset the mitigation."""

    def test_rotating_taint_source_does_not_reset_dampening(self, tmp_path: Path) -> None:
        cycle, _ = _cycle(tmp_path)
        res = [("r1", "c1", 1.0, SignalType.READ)]
        # Same agent hits r1 twice, rotating the taint_source label each time.
        cycle.run_once([_session("attacker", res, taint_source="web")])
        cycle.run_once([_session("attacker", res, taint_source="mcp")])
        # Keyed on the agent, the count accrued to ONE origin (dampening engages),
        # not split across two source labels (which would keep dampening at 1.0).
        assert cycle._prior_counts.get(("r1", "attacker")) == 2
        assert ("r1", "web") not in cycle._prior_counts
        assert ("r1", "mcp") not in cycle._prior_counts
        assert cycle._signal_history["r1"] == ["attacker", "attacker"]

    def test_distinct_agents_are_treated_as_distinct_origins(self, tmp_path: Path) -> None:
        cycle, _ = _cycle(tmp_path)
        res = [("r1", "c1", 1.0, SignalType.READ)]
        cycle.run_once([_session("agentA", res, taint_source="web")])
        cycle.run_once([_session("agentB", res, taint_source="web")])
        assert cycle._prior_counts.get(("r1", "agentA")) == 1
        assert cycle._prior_counts.get(("r1", "agentB")) == 1

    def test_authenticated_source_class_takes_precedence(self, tmp_path: Path) -> None:
        cycle, _ = _cycle(tmp_path)
        res = [("r1", "c1", 1.0, SignalType.READ)]
        # When core attests a source_class, it keys the mitigation (over agent_id).
        cycle.run_once([_session("agentA", res, source_class="trusted-mcp")])
        cycle.run_once([_session("agentB", res, source_class="trusted-mcp")])
        assert cycle._prior_counts.get(("r1", "trusted-mcp")) == 2


class TestCrashConsistencyAndLocking:
    def test_state_persisted_before_snapshot_swap(self, tmp_path: Path, monkeypatch) -> None:
        # If the swap crashes, state must already be persisted (version ahead of the
        # snapshot) and no snapshot must have become visible — so the next run derives
        # a fresh higher version rather than re-emitting this one with new content.
        import axor_sentinel.sentinel.cycle as cyc

        cycle, _ = _cycle(tmp_path)
        sess = _session("agent1", [("r1", "c1", 1.0, SignalType.READ)])

        def _boom(*a, **kw):
            raise RuntimeError("swap crash")

        monkeypatch.setattr(cyc, "atomic_swap", _boom)
        with pytest.raises(RuntimeError, match="swap crash"):
            cycle.run_once([sess])

        # State was written (version 1) before the swap blew up.
        _, _, _, ver, _ev = SentinelCycle.load_state(tmp_path / "sentinel_state.json")
        assert ver == 1
        # The snapshot never became visible.
        assert not (tmp_path / "snapshot_current").exists()

    def test_lock_held_during_cycle_and_released_after(self, tmp_path: Path, monkeypatch) -> None:
        import axor_sentinel.sentinel.cycle as cyc

        cycle, _ = _cycle(tmp_path)
        seen = {}

        real_swap = cyc.atomic_swap

        def _checking_swap(*a, **kw):
            seen["locked_during"] = cycle._lock.locked()
            return real_swap(*a, **kw)

        monkeypatch.setattr(cyc, "atomic_swap", _checking_swap)
        cycle.run_once([])

        assert seen["locked_during"] is True          # held across the cycle body
        assert cycle._lock.acquire(blocking=False)     # released afterwards
        cycle._lock.release()


# ── _dedupe_sessions: collapse records that share a session_id ────────────────

def _summary(
    session_id: str,
    *,
    agent_id: str = "a",
    had_taint: bool = True,
    had_export: bool = False,
    had_escalation: bool = False,
    resources: list[tuple[str, str, float, SignalType]] | None = None,
    taint_source: str = "mcp",
    source_class: str = "",
    started_at: float = 100.0,
) -> SessionSummary:
    accesses = [
        ResourceAccess(resource_id=r, container_id=c, canonical_confidence=conf, signal_type=sig)
        for r, c, conf, sig in (resources or [])
    ]
    return SessionSummary(
        session_id=session_id,
        agent_id=agent_id,
        started_at=started_at,
        had_taint=had_taint,
        had_export_attempt=had_export,
        had_failed_export=False,
        had_escalation=had_escalation,
        accessed_resources=accesses,
        taint_source=taint_source,
        source_class=source_class,
    )


class TestDedupeSessions:
    def test_distinct_session_ids_preserved_in_order(self) -> None:
        a = _summary("s1", resources=[("r1", "c1", 1.0, SignalType.READ)])
        b = _summary("s2", resources=[("r2", "c2", 1.0, SignalType.READ)])
        out = SentinelCycle._dedupe_sessions([a, b])
        assert [s.session_id for s in out] == ["s1", "s2"]

    def test_same_session_id_merges_union_of_evidence(self) -> None:
        # mirrors core-derived record + axor-probe drift buffer for one session
        core = _summary(
            "s1", resources=[("r1", "c1", 1.0, SignalType.READ)],
            taint_source="mcp", source_class="tool:fs", started_at=100.0,
        )
        probe = _summary(
            "s1", resources=[], had_escalation=True,
            taint_source="behavioral_drift", source_class="", started_at=50.0,
        )
        out = SentinelCycle._dedupe_sessions([core, probe])
        assert len(out) == 1
        m = out[0]
        assert m.had_taint is True
        assert m.had_escalation is True            # OR-ed in from the probe record
        assert m.started_at == 50.0               # earliest
        assert m.source_class == "tool:fs"        # first attested value kept
        assert m.taint_source == "mcp"            # first wins (descriptive only)
        assert [a.resource_id for a in m.accessed_resources] == ["r1"]

    def test_repeated_access_not_double_counted(self) -> None:
        a = _summary("s1", resources=[("r1", "c1", 1.0, SignalType.READ)])
        b = _summary("s1", resources=[
            ("r1", "c1", 1.0, SignalType.READ),       # duplicate of a's access
            ("r2", "c2", 1.0, SignalType.READ),
        ])
        out = SentinelCycle._dedupe_sessions([a, b])
        ids = sorted(x.resource_id for x in out[0].accessed_resources)
        assert ids == ["r1", "r2"]                 # r1 appears once, not twice

    def test_later_record_can_attest_source_class(self) -> None:
        a = _summary("s1", source_class="")
        b = _summary("s1", source_class="tool:fs")
        out = SentinelCycle._dedupe_sessions([a, b])
        assert out[0].source_class == "tool:fs"

    def test_inputs_are_not_mutated(self) -> None:
        a = _summary("s1", resources=[("r1", "c1", 1.0, SignalType.READ)])
        b = _summary("s1", resources=[("r2", "c2", 1.0, SignalType.READ)], had_escalation=True)
        SentinelCycle._dedupe_sessions([a, b])
        assert len(a.accessed_resources) == 1      # caller's list untouched
        assert a.had_escalation is False


def test_run_once_dedupes_duplicate_session(tmp_path: Path) -> None:
    """A session_id present twice in one cycle is counted once for dampening."""
    cycle, _neo4j = _cycle(tmp_path)
    sess = _summary("s1", agent_id="a", resources=[("r1", "c1", 1.0, SignalType.READ)])
    dup = _summary("s1", agent_id="a", resources=[("r1", "c1", 1.0, SignalType.READ)])
    cycle.run_once([sess, dup])
    # origin = source_class or agent_id = "a"; the dampening counter for (r1, a)
    # must increment once — without dedup the duplicate would push it to 2.
    assert cycle._prior_counts[("r1", "a")] == 1

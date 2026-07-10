"""
Adversarial tests for deterministic fanout detection.

The trigger is the declared quota (SentinelPolicy.fanout_containers, default 7):
a tainted session touching MORE than the quota of distinct containers at rank
>= READ_SUMMARIZE fires. There is no cold start and no self-trained baseline to
walk upward (F5 closed by construction); the z-score against the smoothed
baseline is telemetry on the emitted signal only.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.cycle import (
    ResourceAccess,
    SentinelCycle,
    SessionSummary,
)
from axor_sentinel.sentinel.events import AgentContainerBaseline
from axor_sentinel.sentinel.predicates import SentinelPolicy

_QUOTA = SentinelPolicy().fanout_containers


class MockNeo4j:
    def run(self, q, **kw):
        pass


def _make_session(
    agent_id: str,
    n_containers: int,
    signal_type: SignalType = SignalType.READ_SUMMARIZE,
    had_taint: bool = True,
) -> SessionSummary:
    accesses = [
        ResourceAccess(
            resource_id=f"r{i}",
            container_id=f"c{i}",          # each resource in its own container
            canonical_confidence=1.0,
            signal_type=signal_type,
        )
        for i in range(n_containers)
    ]
    return SessionSummary(
        session_id=f"sess_{agent_id}_{n_containers}",
        agent_id=agent_id,
        started_at=0.0,
        had_taint=had_taint,
        had_export_attempt=False,
        had_failed_export=False,
        had_escalation=False,
        accessed_resources=accesses,
        taint_source="mcp",
    )


def _cycle(baselines=None) -> SentinelCycle:
    return SentinelCycle(
        MockNeo4j(),
        Path(tempfile.mkdtemp()),
        agent_baselines=baselines or {},
    )


class TestQuotaTrigger:
    def test_at_quota_no_trigger(self):
        cycle = _cycle()
        s = _make_session("a", _QUOTA)
        assert cycle._check_fanout(s, 0.0) is None

    def test_above_quota_triggers(self):
        cycle = _cycle()
        s = _make_session("a", _QUOTA + 1)
        sig = cycle._check_fanout(s, 0.0)
        assert sig is not None
        assert sig.unique_containers == _QUOTA + 1

    def test_no_baseline_still_triggers(self):
        # No cold start: a quota needs no history (F5 closed by construction).
        cycle = _cycle(baselines={})
        sig = cycle._check_fanout(_make_session("newagent", _QUOTA + 3), 0.0)
        assert sig is not None
        assert sig.z_score == 0.0            # telemetry-only, no baseline

    def test_custom_quota_respected(self):
        cycle = SentinelCycle(
            MockNeo4j(), Path(tempfile.mkdtemp()),
            agent_baselines={}, policy=SentinelPolicy(fanout_containers=2),
        )
        assert cycle._check_fanout(_make_session("a", 3), 0.0) is not None
        assert cycle._check_fanout(_make_session("a", 2), 0.0) is None


class TestGates:
    def test_untainted_never_triggers(self):
        cycle = _cycle()
        s = _make_session("a", _QUOTA + 5, had_taint=False)
        assert cycle._check_fanout(s, 0.0) is None

    def test_read_only_never_triggers(self):
        # Signal-rank gate (A-15): plain READ fanout is not staging.
        cycle = _cycle()
        s = _make_session("a", _QUOTA + 5, signal_type=SignalType.READ)
        assert cycle._check_fanout(s, 0.0) is None

    def test_read_summarize_triggers(self):
        cycle = _cycle()
        s = _make_session("a", _QUOTA + 1, signal_type=SignalType.READ_SUMMARIZE)
        assert cycle._check_fanout(s, 0.0) is not None

    def test_duplicate_containers_collapse(self):
        cycle = _cycle()
        accesses = [
            ResourceAccess(f"r{i}", "same_container", 1.0, SignalType.READ_SUMMARIZE)
            for i in range(_QUOTA * 3)
        ]
        s = SessionSummary(
            session_id="s", agent_id="a", started_at=0.0, had_taint=True,
            had_export_attempt=False, had_failed_export=False,
            had_escalation=False, accessed_resources=accesses,
        )
        assert cycle._check_fanout(s, 0.0) is None


class TestZScoreTelemetry:
    def test_z_computed_when_baseline_exists(self):
        baseline = AgentContainerBaseline(
            agent_id="a", mean_containers_per_session=2.0,
            std_containers_per_session=2.0, session_count=50, last_updated=0.0,
        )
        cycle = _cycle(baselines={"a": baseline})
        sig = cycle._check_fanout(_make_session("a", _QUOTA + 1), 0.0)
        assert sig is not None
        assert sig.baseline_mean == 2.0
        assert sig.z_score == ((_QUOTA + 1) - 2.0) / 2.0

    def test_z_never_gates_the_trigger(self):
        # Huge baseline mean → z is negative, quota still fires: z is telemetry.
        baseline = AgentContainerBaseline(
            agent_id="a", mean_containers_per_session=100.0,
            std_containers_per_session=5.0, session_count=50, last_updated=0.0,
        )
        cycle = _cycle(baselines={"a": baseline})
        sig = cycle._check_fanout(_make_session("a", _QUOTA + 1), 0.0)
        assert sig is not None
        assert sig.z_score < 0

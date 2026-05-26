"""
Adversarial tests for fanout detection.

Covers invariants A-14, A-15 and spec adversarial test matrix:
  - z-score threshold: z=2.0 no trigger, z=2.5 trigger, z=4.0 trigger
  - Cold-start suppression: < 10 sessions never triggers
  - Signal-type gate: read-only + high z-score does not trigger
  - Low-std agent: std≈0 requires absolute delta > FANOUT_MIN_DELTA
"""
from __future__ import annotations

from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.cycle import (
    FANOUT_MIN_DELTA,
    FANOUT_MIN_SESSIONS,
    FANOUT_Z_THRESHOLD,
    ResourceAccess,
    SentinelCycle,
    SessionSummary,
)
from axor_sentinel.sentinel.events import AgentContainerBaseline


# ── Helpers ────────────────────────────────────────────────────────────────────

class MockNeo4j:
    def run(self, q, **kw):
        pass


def _make_baseline(
    agent_id: str,
    mean: float,
    std: float,
    session_count: int = FANOUT_MIN_SESSIONS,
) -> AgentContainerBaseline:
    return AgentContainerBaseline(
        agent_id=agent_id,
        mean_containers_per_session=mean,
        std_containers_per_session=std,
        session_count=session_count,
        last_updated=0.0,
    )


def _make_session(
    agent_id: str,
    n_containers: int,
    signal_type: SignalType = SignalType.READ_SUMMARIZE,
    had_taint: bool = True,
    had_export: bool = False,
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
        had_export_attempt=had_export,
        had_failed_export=False,
        had_escalation=False,
        accessed_resources=accesses,
        taint_source="mcp",
    )


def _cycle_with_baseline(
    agent_id: str,
    mean: float,
    std: float,
    session_count: int = FANOUT_MIN_SESSIONS,
    tmpdir=None,
) -> SentinelCycle:
    import tempfile
    from pathlib import Path
    td = tmpdir or tempfile.mkdtemp()
    baseline = _make_baseline(agent_id, mean, std, session_count)
    return SentinelCycle(
        MockNeo4j(),
        Path(td),
        agent_baselines={agent_id: baseline},
    )


# ── z-score threshold boundary — 3 variants ───────────────────────────────────

class TestZScoreThreshold:
    """Invariant A-15 condition: z_score > FANOUT_Z_THRESHOLD (2.5)."""

    def test_z_below_threshold_no_trigger(self, tmp_path):
        """z=2.0 — does not trigger fanout."""
        # With mean=5, std=1: unique_containers=7 → z=(7-5)/1=2.0
        cycle = _cycle_with_baseline("a1", mean=5.0, std=1.0, tmpdir=tmp_path)
        session = _make_session("a1", n_containers=7)
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is None

    def test_z_at_threshold_no_trigger(self, tmp_path):
        """z=2.5 exactly (not >) — does not trigger (strictly greater than)."""
        # z = (unique - mean) / std = 2.5 exactly: unique=7.5 not integer; use unique=8, std=1, mean=5.5
        cycle = _cycle_with_baseline("a2", mean=5.5, std=1.0, tmpdir=tmp_path)
        session = _make_session("a2", n_containers=8)  # z=(8-5.5)/1=2.5 exactly
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is None  # must be strictly greater than 2.5

    def test_z_above_threshold_triggers(self, tmp_path):
        """z=4.0 — triggers fanout signal."""
        # With mean=5, std=1: unique=9 → z=4.0
        cycle = _cycle_with_baseline("a3", mean=5.0, std=1.0, tmpdir=tmp_path)
        session = _make_session("a3", n_containers=9)
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is not None
        assert fanout.z_score > FANOUT_Z_THRESHOLD


# ── Cold-start suppression — 2 variants ───────────────────────────────────────

class TestColdStartSuppression:
    """Invariant A-14: agents with < FANOUT_MIN_SESSIONS never trigger fanout."""

    def test_no_baseline_no_trigger(self, tmp_path):
        """Agent with no baseline at all — cold start, no trigger."""
        cycle = SentinelCycle(MockNeo4j(), tmp_path)  # no baselines
        session = _make_session("new_agent", n_containers=100)
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is None

    def test_insufficient_session_count_no_trigger(self, tmp_path):
        """Agent with session_count=9 (< 10) — still cold start."""
        cycle = _cycle_with_baseline(
            "a_cold", mean=2.0, std=0.5, session_count=FANOUT_MIN_SESSIONS - 1,
            tmpdir=tmp_path,
        )
        session = _make_session("a_cold", n_containers=50)
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is None


# ── Signal-type gate — 2 variants ─────────────────────────────────────────────

class TestSignalTypeGate:
    """
    Invariant A-15: fanout requires max_signal_type >= READ_SUMMARIZE.
    A tainted session with only READ-level signals must not trigger.
    """

    def test_read_only_high_z_no_trigger(self, tmp_path):
        """Read-only tainted session with z=5.0 does NOT trigger fanout."""
        cycle = _cycle_with_baseline("a_ro", mean=2.0, std=0.5, tmpdir=tmp_path)
        session = _make_session("a_ro", n_containers=10, signal_type=SignalType.READ)
        # z = (10 - 2.0) / 0.5 = 16.0 — far above threshold
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is None

    def test_read_summarize_high_z_triggers(self, tmp_path):
        """READ_SUMMARIZE session with high z-score DOES trigger."""
        cycle = _cycle_with_baseline("a_sum", mean=2.0, std=0.5, tmpdir=tmp_path)
        session = _make_session(
            "a_sum", n_containers=10, signal_type=SignalType.READ_SUMMARIZE
        )
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is not None


# ── Low-std agent — 2 variants ────────────────────────────────────────────────

class TestLowStdAgent:
    """
    Agent with std≈0 (very predictable access pattern) uses absolute delta guard.
    Requires current_unique_containers > baseline.mean + FANOUT_MIN_DELTA.
    """

    def test_low_std_below_absolute_delta_no_trigger(self, tmp_path):
        """std≈0, unique_containers = mean + 2 (< FANOUT_MIN_DELTA=3) → no trigger."""
        mean = 5.0
        cycle = _cycle_with_baseline("a_etl", mean=mean, std=0.0, tmpdir=tmp_path)
        session = _make_session("a_etl", n_containers=int(mean) + 2)
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is None

    def test_low_std_above_absolute_delta_triggers(self, tmp_path):
        """std≈0, unique_containers = mean + FANOUT_MIN_DELTA + 1 → triggers."""
        mean = 5.0
        cycle = _cycle_with_baseline("a_etl2", mean=mean, std=0.0, tmpdir=tmp_path)
        session = _make_session("a_etl2", n_containers=int(mean) + FANOUT_MIN_DELTA + 1)
        fanout = cycle._check_fanout(session, now=0.0)
        assert fanout is not None

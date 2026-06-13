"""
Tests for CoreSessionSink — the structural axor-core SessionSink adapter.

Core's SessionAuditRecord is simulated with a local _FakeRecord (and
_FakeInvocation) exposing the same attributes, so the sink's structural
Protocol compatibility is exercised without importing axor-core.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Mapping

from axor_sentinel.graph.model import SignalType
from axor_sentinel.integration.core_sink import CoreSessionSink
from axor_sentinel.sentinel.cycle import ResourceAccess, SessionSummary


# ── Local stand-ins for the core observation contract ──────────────────────────

@dataclass(frozen=True)
class _FakeInvocation:
    tool: str
    args: Mapping[str, object]
    executed: bool = True


@dataclass(frozen=True)
class _FakeRecord:
    session_id: str = "sess-1"
    agent_id: str = "agent-1"
    started_at: float = 1000.0
    ended_at: float = 1100.0
    taint_active: bool = False
    taint_sources: tuple[str, ...] = ()
    event_kinds: tuple[str, ...] = ()
    tool_invocations: tuple[_FakeInvocation, ...] = field(default_factory=tuple)
    source_class: str = ""


def _run(coro):
    return asyncio.run(coro)


# ── Tests ───────────────────────────────────────────────────────────────────────

def test_fake_record_satisfies_the_structural_contract():
    # The fake must conform to the sentinel-defined CoreSessionRecord Protocol —
    # otherwise these tests would pass against a shape the sink can't really consume.
    from axor_sentinel.integration.core_sink import CoreSessionRecord, ToolInvocationRecord
    assert isinstance(_FakeRecord(), CoreSessionRecord)
    assert isinstance(_FakeInvocation(tool="x", args={}), ToolInvocationRecord)


def test_read_only_session_maps_to_read_grade():
    sink = CoreSessionSink()
    record = _FakeRecord(
        tool_invocations=(
            _FakeInvocation(tool="fs_read", args={"path": "/data/report.txt"}),
        ),
    )

    _run(sink.on_session_closed(record))
    pending = sink.drain_pending()

    assert len(pending) == 1
    summary = pending[0]
    assert isinstance(summary, SessionSummary)
    assert summary.had_taint is False
    assert summary.had_export_attempt is False
    assert summary.had_failed_export is False
    assert len(summary.accessed_resources) == 1
    access = summary.accessed_resources[0]
    assert isinstance(access, ResourceAccess)
    assert access.signal_type is SignalType.READ
    assert access.resource_id  # non-empty derived id


def test_tainted_export_denied_session():
    sink = CoreSessionSink()
    record = _FakeRecord(
        session_id="sess-2",
        agent_id="agent-2",
        taint_active=True,
        taint_sources=("mcp_tool_output", "web_fetch"),
        event_kinds=("intent_approved", "intent_denied", "escalation_granted"),
        tool_invocations=(
            _FakeInvocation(tool="fs_read", args={"item_id": "abc123"}),
            _FakeInvocation(
                tool="email_send",
                args={"path": "/out/leak.csv"},
                executed=False,
            ),
        ),
    )

    _run(sink.on_session_closed(record))
    summary = sink.drain_pending()[0]

    assert summary.had_taint is True
    assert summary.had_export_attempt is True
    assert summary.had_failed_export is True
    assert summary.had_escalation is True
    assert summary.taint_source == "mcp_tool_output"

    # provider_id-derived id is the raw object id, confidence 1.0
    read_access = next(a for a in summary.accessed_resources if a.resource_id == "abc123")
    assert read_access.signal_type is SignalType.READ
    assert read_access.canonical_confidence == 1.0
    # the failed export grades to the strongest export-ish member
    export_access = next(
        a for a in summary.accessed_resources if a.signal_type is SignalType.READ_EXPORT_FAILED
    )
    assert export_access is not None


def test_taint_propagated_event_implies_had_taint():
    sink = CoreSessionSink()
    record = _FakeRecord(
        taint_active=False,
        event_kinds=("taint_propagated",),
    )
    _run(sink.on_session_closed(record))
    assert sink.drain_pending()[0].had_taint is True


def test_executed_export_grades_to_adjacent():
    sink = CoreSessionSink()
    record = _FakeRecord(
        event_kinds=("intent_approved",),
        tool_invocations=(
            _FakeInvocation(tool="slack_post", args={"path": "/msg"}, executed=True),
        ),
    )
    _run(sink.on_session_closed(record))
    summary = sink.drain_pending()[0]
    assert summary.had_export_attempt is True
    assert summary.had_failed_export is False
    assert summary.accessed_resources[0].signal_type is SignalType.READ_EXPORT_ADJACENT


def test_taint_source_defaults_when_absent():
    sink = CoreSessionSink()
    record = _FakeRecord(taint_active=True, taint_sources=())
    _run(sink.on_session_closed(record))
    assert sink.drain_pending()[0].taint_source == "unknown_external"


def test_drain_pending_clears_and_count_tracks():
    sink = CoreSessionSink()
    assert sink.pending_count() == 0
    _run(sink.on_session_closed(_FakeRecord(session_id="a")))
    _run(sink.on_session_closed(_FakeRecord(session_id="b")))
    assert sink.pending_count() == 2

    drained = sink.drain_pending()
    assert [s.session_id for s in drained] == ["a", "b"]
    assert sink.pending_count() == 0
    assert sink.drain_pending() == []


def test_mapping_error_buffers_minimal_summary_and_does_not_raise():
    sink = CoreSessionSink()

    class _BrokenRecord:
        session_id = "broken-1"
        agent_id = "agent-x"
        started_at = 42.0
        taint_active = True

        @property
        def taint_sources(self):
            raise RuntimeError("boom")

        @property
        def event_kinds(self):
            raise RuntimeError("boom")

        @property
        def tool_invocations(self):
            raise RuntimeError("boom")

    # Must not raise.
    _run(sink.on_session_closed(_BrokenRecord()))

    pending = sink.drain_pending()
    assert len(pending) == 1
    summary = pending[0]
    assert summary.session_id == "broken-1"
    assert summary.agent_id == "agent-x"
    assert summary.started_at == 42.0
    assert summary.had_taint is True          # preserved from taint_active
    assert summary.accessed_resources == []   # minimal: nothing derived
    assert summary.taint_source == "unknown_external"

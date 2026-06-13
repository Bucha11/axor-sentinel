"""Tests for ProbeTaintBridge — the axor-probe SentinelSessionSink adapter.

The bridge buffers probe-flagged (behavioral-drift) sessions for the next audit
cycle. It has no resource-level signal, so the buffered summaries carry an empty
accessed_resources list and contribute only had_taint / fanout pressure.
"""
from __future__ import annotations

import asyncio

from axor_sentinel.integration.probe_bridge import ProbeTaintBridge
from axor_sentinel.sentinel.cycle import SessionSummary


def _run(coro):
    return asyncio.run(coro)


def test_marked_session_is_buffered_as_tainted():
    bridge = ProbeTaintBridge()
    _run(bridge.mark_session_tainted("sess-1", "agent-1"))
    assert bridge.pending_count() == 1
    pending = bridge.drain_pending()
    assert len(pending) == 1
    s = pending[0]
    assert isinstance(s, SessionSummary)
    assert s.session_id == "sess-1" and s.agent_id == "agent-1"
    assert s.had_taint is True
    assert s.accessed_resources == []          # probe has no resource-level signal
    assert s.taint_source == "behavioral_drift"


def test_drain_clears_the_buffer():
    bridge = ProbeTaintBridge()
    _run(bridge.mark_session_tainted("s1", "a1"))
    _run(bridge.mark_session_tainted("s2", "a1"))
    assert bridge.pending_count() == 2
    assert len(bridge.drain_pending()) == 2
    assert bridge.pending_count() == 0
    assert bridge.drain_pending() == []        # idempotent once drained


def test_probe_session_origin_falls_back_to_agent_id():
    # No source_class on a probe session -> mitigation keys on agent_id, never the
    # behavioral_drift taint_source label (F1).
    bridge = ProbeTaintBridge()
    _run(bridge.mark_session_tainted("s1", "agent-9"))
    s = bridge.drain_pending()[0]
    assert s.mitigation_origin == "agent-9"

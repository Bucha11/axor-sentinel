from __future__ import annotations

import logging
import time

from axor_sentinel.sentinel.cycle import SessionSummary

log = logging.getLogger("axor.sentinel.probe_bridge")

# taint_source value used for sessions flagged by axor-probe behavioral drift
_PROBE_TAINT_SOURCE = "behavioral_drift"


class ProbeTaintBridge:
    """
    Implements axor-probe's SentinelSessionSink Protocol.

    Buffers sessions flagged by axor-probe as having behavioral drift
    (had_taint=True, taint_source="behavioral_drift").  The caller drains
    the buffer before each SentinelCycle.run_once() call:

        probe_sessions = bridge.drain_pending()
        all_sessions   = core_sessions + probe_sessions
        cycle.run_once(all_sessions, resource_scores, container_members)

    axor-sentinel never imports axor-probe — the bridge implements axor-probe's
    SentinelSessionSink Protocol by structural compatibility only.

    Thread safety: not thread-safe. Use one bridge per sentinel cycle runner.
    """

    def __init__(self) -> None:
        self._pending: list[SessionSummary] = []

    async def mark_session_tainted(self, session_id: str, agent_id: str) -> None:
        """
        Buffer a SessionSummary with had_taint=True for the next audit cycle.

        accessed_resources is empty — probe does not have resource-level signal.
        SentinelCycle processes this session for fanout detection and
        had_taint=True contribution but no hot weights (no resource accesses).
        """
        self._pending.append(SessionSummary(
            session_id=session_id,
            agent_id=agent_id,
            started_at=time.time(),
            had_taint=True,
            had_export_attempt=False,
            had_failed_export=False,
            had_escalation=False,
            accessed_resources=[],
            taint_source=_PROBE_TAINT_SOURCE,
        ))
        log.debug(
            "probe drift: session=%s agent=%s buffered as had_taint=True",
            session_id, agent_id,
        )

    def drain_pending(self) -> list[SessionSummary]:
        """
        Return and clear the buffer of probe-flagged sessions.

        Call once before each SentinelCycle.run_once() and merge the result
        with the core-derived session list.
        """
        sessions = list(self._pending)
        self._pending.clear()
        return sessions

    def pending_count(self) -> int:
        return len(self._pending)

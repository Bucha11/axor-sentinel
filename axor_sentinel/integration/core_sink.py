from __future__ import annotations

import logging
from typing import Mapping, Protocol, Sequence, runtime_checkable

from axor_sentinel.graph.model import SignalType
from axor_sentinel.graph.normalizer import normalize_resource_id
from axor_sentinel.integration.intent_enricher import (
    _derive_container_id,
    derive_resource_info,
)
from axor_sentinel.sentinel.cycle import ResourceAccess, SessionSummary


# Structural contract for a closed-session record from core. axor-core does NOT yet
# emit such a record (there is no SessionSink / SessionAuditRecord in core today),
# so this is a FORWARD integration: sentinel defines the shape it consumes here,
# and a host adapter (or a future core observation contract) can satisfy it by
# duck typing. Defining it sentinel-side keeps the attachment structural (invariant
# P-34) and self-contained — no import edge into a core module that may not exist.
@runtime_checkable
class ToolInvocationRecord(Protocol):
    tool: str
    args: Mapping[str, object]
    executed: bool


@runtime_checkable
class CoreSessionRecord(Protocol):
    session_id: str
    agent_id: str
    started_at: float
    taint_active: bool
    taint_sources: Sequence[str]
    event_kinds: Sequence[str]
    tool_invocations: Sequence[ToolInvocationRecord]


log = logging.getLogger("axor.sentinel.core_sink")

# Tool-name substrings that indicate an export / exfiltration-shaped action.
# Coarse, name-only heuristic — the audit cycle (not the hot path) consumes this,
# so a best-effort signal is acceptable.
_EXPORT_TOOL_TOKENS: tuple[str, ...] = (
    "export", "send", "upload", "email", "post",
    "write", "commit", "push", "share",
)


def _is_export_tool(tool: str) -> bool:
    """True if the tool name looks like an export / outbound action."""
    t = (tool or "").lower()
    return any(token in t for token in _EXPORT_TOOL_TOKENS)


class CoreSessionSink:
    """
    Buckets a closed core session (``CoreSessionRecord``) into a ``SessionSummary``.

    FORWARD INTEGRATION: axor-core does not yet emit a per-session record or call an
    ``on_session_closed`` sink — there is no such contract in core today. This sink
    is ready for when a host adapter (or a future core observation contract) hands
    over closed sessions shaped like ``CoreSessionRecord`` (defined above,
    sentinel-side). Until then it has no producer and is exercised only by tests.

    The counterpart to ``ProbeTaintBridge``, but for *full* core sessions rather
    than just probe-flagged ones. Given a record's raw facts (``taint_active`` /
    ``taint_sources`` / ``event_kinds`` / ``tool_invocations``) this sink derives
    had_taint, had_export_attempt, had_failed_export, had_escalation and graded
    ``ResourceAccess`` entries, and buffers a ``SessionSummary`` for the next cycle.

    The caller drains both buffers before each cycle::

        cycle.run_once(
            core_sink.drain_pending() + probe_bridge.drain_pending(),
            resource_scores,
            container_members,
        )

    axor-sentinel never imports axor-probe or axor-core here — the consumed shape is
    a sentinel-defined structural Protocol (invariant P-34).

    Fail-safe: ``on_session_closed`` must never raise. On a mapping error it logs
    and still buffers a minimal ``SessionSummary`` (had_taint from the record's
    ``taint_active``, empty accessed_resources) so a session is never silently
    dropped.

    Thread safety: not thread-safe. Use one sink per sentinel cycle runner.
    """

    def __init__(self) -> None:
        self._pending: list[SessionSummary] = []

    async def on_session_closed(self, record: "CoreSessionRecord") -> None:
        """
        Buffer a sentinel ``SessionSummary`` derived from a core audit record.

        Must not raise. On any mapping failure, a minimal summary is buffered
        instead so the session still reaches the audit cycle.
        """
        try:
            summary = self._map_record(record)
        except Exception as exc:
            log.warning(
                "core_sink: failed to map session=%s — buffering minimal summary: %s",
                getattr(record, "session_id", "<unknown>"),
                exc,
            )
            summary = self._minimal_summary(record)
        self._pending.append(summary)

    def _map_record(self, record: "CoreSessionRecord") -> SessionSummary:
        """Translate a raw core ``SessionAuditRecord`` into a ``SessionSummary``."""
        event_kinds = tuple(record.event_kinds)
        taint_sources = tuple(record.taint_sources)

        # had_taint: final taint state OR any in-session propagation event.
        had_taint = bool(record.taint_active) or ("taint_propagated" in event_kinds)

        taint_source = taint_sources[0] if taint_sources else "unknown_external"

        had_escalation = (
            "escalation_granted" in event_kinds
            or "escalation_denied" in event_kinds
        )

        invocations = tuple(record.tool_invocations)
        had_export_attempt = any(_is_export_tool(inv.tool) for inv in invocations)
        # Coarse approximation: core gives a session-level "intent_denied" kind, not
        # a per-invocation export-denied flag. We treat an export attempt in a
        # session that also saw a denial as a failed export. This can over-attribute
        # when a denial is for a non-export intent — documented best-effort signal.
        had_failed_export = had_export_attempt and ("intent_denied" in event_kinds)

        accessed_resources = [
            self._map_access(inv, event_kinds) for inv in invocations
        ]

        return SessionSummary(
            session_id=record.session_id,
            agent_id=record.agent_id,
            started_at=record.started_at,
            had_taint=had_taint,
            had_export_attempt=had_export_attempt,
            had_failed_export=had_failed_export,
            had_escalation=had_escalation,
            accessed_resources=accessed_resources,
            taint_source=taint_source,
        )

    def _map_access(self, inv, event_kinds: tuple[str, ...]) -> ResourceAccess:
        """
        Map one ``ToolInvocationRecord`` to a graded ``ResourceAccess``.

        Resource identity is derived with the same arg-key extraction the
        SnapshotIntentEnricher uses (``derive_resource_info`` →
        ``normalize_resource_id``) so IDs are consistent across the hot path and
        the audit path.

        SignalType grading (using real members of graph.model.SignalType):
          * non-export tool                      -> READ                  (0.4)
          * export-shaped tool, executed         -> READ_EXPORT_ADJACENT  (0.8)
          * export-shaped tool, not executed,
            or session saw a denial              -> READ_EXPORT_FAILED    (1.0)
        """
        resource_info = derive_resource_info(inv.tool, inv.args)
        resource_id, _method, confidence = normalize_resource_id(resource_info)
        container_id = _derive_container_id(
            resource_info.get("path", ""),
            resource_info.get("service", ""),
        )

        if _is_export_tool(inv.tool):
            if not inv.executed or ("intent_denied" in event_kinds):
                signal_type = SignalType.READ_EXPORT_FAILED
            else:
                signal_type = SignalType.READ_EXPORT_ADJACENT
        else:
            signal_type = SignalType.READ

        return ResourceAccess(
            resource_id=resource_id,
            container_id=container_id,
            canonical_confidence=confidence,
            signal_type=signal_type,
        )

    @staticmethod
    def _minimal_summary(record: "CoreSessionRecord") -> SessionSummary:
        """
        Build a degraded-but-non-empty summary when full mapping fails.

        Pulls only the cheapest fields defensively so this itself cannot raise.
        """
        return SessionSummary(
            session_id=str(getattr(record, "session_id", "")),
            agent_id=str(getattr(record, "agent_id", "")),
            started_at=float(getattr(record, "started_at", 0.0) or 0.0),
            had_taint=bool(getattr(record, "taint_active", False)),
            had_export_attempt=False,
            had_failed_export=False,
            had_escalation=False,
            accessed_resources=[],
            taint_source="unknown_external",
        )

    def drain_pending(self) -> list[SessionSummary]:
        """
        Return and clear the buffer of core-derived sessions.

        Call once before each ``SentinelCycle.run_once()`` and merge the result
        with the probe-flagged session list.
        """
        sessions = list(self._pending)
        self._pending.clear()
        return sessions

    def pending_count(self) -> int:
        return len(self._pending)

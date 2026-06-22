from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from axor_sentinel.graph.model import SignalType
from axor_sentinel.graph.normalizer import normalize_resource_id
from axor_sentinel.integration.intent_enricher import (
    _derive_container_id,
    derive_resource_info,
)
from axor_sentinel.sentinel.cycle import ResourceAccess, SessionSummary

if TYPE_CHECKING:
    # Lazy / type-only reference to axor-core's closed-session audit contract.
    # axor-sentinel attaches to core only by *structural* Protocol compatibility
    # (invariant P-34): we implement SessionSink's `on_session_closed` signature
    # without a hard runtime import edge. This mirrors the TYPE_CHECKING-only
    # core import already used in integration/intent_enricher.py.
    from axor_core.contracts.session import (  # type: ignore[import-untyped]
        SessionAuditRecord,
    )

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
    Implements axor-core's ``SessionSink`` Protocol structurally.

    The counterpart to ``ProbeTaintBridge``, but for *full* core sessions rather
    than just probe-flagged ones. Core emits a neutral, raw-facts
    ``SessionAuditRecord`` per closed session via ``on_session_closed``; this sink
    buckets those raw facts into a sentinel ``SessionSummary`` (deriving had_taint,
    had_export_attempt, had_failed_export, had_escalation and graded
    ``ResourceAccess`` entries) and buffers it for the next audit cycle.

    This is the "consumer does the bucketing" side: core hands over only raw
    ``taint_active`` / ``taint_sources`` / ``event_kinds`` / ``tool_invocations``,
    and the sink translates them into sentinel vocabulary. It finally produces the
    ``core_sessions`` list that ``ProbeTaintBridge``'s docstring references.

    The caller drains both buffers before each cycle::

        cycle.run_once(
            core_sink.drain_pending() + probe_bridge.drain_pending(),
            resource_scores,
            container_members,
        )

    axor-sentinel never imports axor-probe, and references the core observation
    types under TYPE_CHECKING only — attachment is structural (invariant P-34).

    Fail-safe: ``on_session_closed`` must never raise. On a mapping error it logs
    and still buffers a minimal ``SessionSummary`` (had_taint from the record's
    ``taint_active``, empty accessed_resources) so a session is never silently
    dropped.

    Thread safety: not thread-safe. Use one sink per sentinel cycle runner.
    """

    def __init__(self) -> None:
        self._pending: list[SessionSummary] = []

    async def on_session_closed(self, record: "SessionAuditRecord") -> None:
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

    def _map_record(self, record: "SessionAuditRecord") -> SessionSummary:
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
    def _minimal_summary(record: "SessionAuditRecord") -> SessionSummary:
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

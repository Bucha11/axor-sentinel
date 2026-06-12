"""
Integration tests for the observe-only reputation → degradation coupling.

Current axor-core treats reputation as TELEMETRY: the enricher populates
target_resource_reputation / target_container_reputation on NormalizedIntent, and
the ONLY enforcement coupling is the opt-in DegradationEngine(detection_floor=...)
"tighten" path. Core never denies on reputation (no Phase-1 deny, no anomaly
detector). These tests pin:
  - the suspicion→reputation polarity conversion at the enricher boundary,
  - that a suspicious resource TIGHTENS degradation while the call still executes
    (observe-only — never a deny), and a benign one does not,
  - the long-standing A-7 invariant (no 'flagged' feature on NormalizedIntent).
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import pytest

# Need axor-core on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "axor-core"))

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import (
    ContextFragment, ContextView, LineageSummary,
)
from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.degradation.engine import DegradationEngine
from axor_core.node.intent_loop import IntentLoop
from axor_core.policy.presets import standard as standard_policy
from axor_sentinel.graph.derive import derive_resource_info
from axor_sentinel.graph.normalizer import normalize_resource_id
from axor_sentinel.integration.intent_enricher import (
    SnapshotIntentEnricher,
    _suspicion_to_reputation,
)
from axor_sentinel.sentinel.snapshot import ReputationSnapshot, atomic_swap
from axor_sentinel.sentinel.weight import FLAG_THRESHOLD

# Operator wiring: core tightens when reputation <= detection_floor; a sentinel-
# flagged resource (suspicion >= FLAG_THRESHOLD) converts to 1 - suspicion, so the
# matching floor is 1 - FLAG_THRESHOLD.
_FLOOR = 1.0 - FLAG_THRESHOLD


def _make_envelope(allowed_tools: frozenset[str] = frozenset({"read"})) -> ExecutionEnvelope:
    policy = standard_policy()
    lineage = LineageSummary(
        node_id="n1", parent_id=None, depth=0,
        ancestry_ids=(), inherited_restrictions=(),
    )
    ctx = ContextView(
        node_id="n1",
        working_summary="test",
        visible_fragments=[
            ContextFragment(kind="fact", content="test", token_estimate=5, source="test")
        ],
        active_constraints=[],
        lineage=lineage,
        token_count=5,
        compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=allowed_tools,
        allow_children=False,
        allow_nested_children=False,
        allow_context_expansion=False,
        allow_export=False,
        allow_mutation=False,
        max_child_depth=0,
    )
    export = ExportContract(mode="deny", allowed_fields=frozenset(), max_export_tokens=None)
    return ExecutionEnvelope(
        node_id="n1", task="test", context=ctx, policy=policy, capabilities=caps,
        export_contract=export, lineage=lineage, cancel_token=make_token(),
    )


def _make_normalized(after_external_read: bool = False, resource_rep: float = 0.0) -> NormalizedIntent:
    return NormalizedIntent(
        tool="read", operation="file_read", target_kind="workdir",
        destination_kind="none", provenance="repo",
        reads_secret_like_data=False, writes_outside_workdir=False,
        executes_generated_code=False, after_external_read=after_external_read,
        after_secret_access=False, data_flow="local_to_local",
        target_resource_reputation=resource_rep, target_container_reputation=0.0,
    )


def _make_intent(path: str = "/data/r.txt") -> Intent:
    return Intent(
        kind=IntentKind.TOOL_CALL,
        payload={"tool": "read", "args": {"path": path}},
        node_id="n1",
    )


class _ReadHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args) -> str:
        return "content"


class _FixedReputationEnricher:
    """Stamps a fixed (already core-polarity) reputation, to drive the loop wiring."""
    def __init__(self, reputation: float):
        self._rep = reputation

    def enrich(self, normalized: NormalizedIntent, intent: Intent) -> NormalizedIntent:
        return dataclasses.replace(normalized, target_resource_reputation=self._rep)


# ── polarity conversion ───────────────────────────────────────────────────────

class TestPolarityConversion:
    def test_zero_suspicion_is_unknown(self):
        # 0.0 suspicion → 0.0 reputation = core "unknown" (never crosses).
        assert _suspicion_to_reputation(0.0) == 0.0

    def test_high_suspicion_maps_to_low_reputation(self):
        # high suspicion = bad → LOW trust reputation (crosses the floor).
        assert _suspicion_to_reputation(0.9) == pytest.approx(0.1)
        assert _suspicion_to_reputation(FLAG_THRESHOLD) == pytest.approx(1 - FLAG_THRESHOLD)

    def test_low_suspicion_maps_to_high_reputation(self):
        # benign-but-known → HIGH trust (does not cross a sane floor).
        assert _suspicion_to_reputation(0.1) == pytest.approx(0.9)

    def test_max_suspicion_does_not_collapse_to_unknown(self):
        # suspicion 1.0 must NOT become 0.0 ("unknown", never crosses) — fail-open.
        r = _suspicion_to_reputation(1.0)
        assert 0.0 < r <= _FLOOR

    def test_enricher_converts_suspicion_to_reputation(self, tmp_path):
        # The snapshot stores suspicion; the enricher hands core the converted trust.
        info = derive_resource_info("read", {"path": "/data/r.txt"})
        rid, _, _ = normalize_resource_id(info)
        snap = ReputationSnapshot(
            version=1, generated_at=time.time(),
            resource_reputation={rid: 0.9},   # suspicion 0.9
            container_reputation={},
        ).with_checksum()
        atomic_swap(tmp_path, snap)
        enricher = SnapshotIntentEnricher.from_dir(tmp_path)
        result = enricher.enrich(_make_normalized(after_external_read=True), _make_intent("/data/r.txt"))
        assert result.target_resource_reputation == pytest.approx(0.1)  # 1 - 0.9


# ── observe-only: tighten degradation, never deny ─────────────────────────────

class TestObserveOnlyDegradation:
    async def _run(self, enricher, floor):
        ex = CapabilityExecutor()
        ex.register(_ReadHandler())
        deg = DegradationEngine(detection_floor=floor)
        loop = IntentLoop(
            capability_executor=ex, trace_events=[],
            degradation_engine=deg, reputation_enricher=enricher,
        )
        env = _make_envelope(frozenset({"read"}))

        async def stream():
            yield ExecutorEvent(
                kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": "read", "tool_use_id": "tu1", "args": {"path": "/data/r.txt"}},
                node_id="n1",
            )
            yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id="n1")

        events = [e async for e in loop.run(stream(), env)]
        return loop, events

    @pytest.mark.asyncio
    async def test_suspicious_resource_tightens_but_does_not_deny(self):
        # suspicion 0.9 → reputation 0.1 <= floor 0.3 → core TIGHTENS to RESTRICTED.
        loop, events = await self._run(_FixedReputationEnricher(0.1), _FLOOR)
        assert loop._degradation_engine.state.level >= DegradationLevel.RESTRICTED
        # But the call still executed — reputation is observe-only, never a deny.
        approved = [e for e in events if e.payload.get("approved") is True]
        assert approved, "the read must still execute; reputation never denies"

    @pytest.mark.asyncio
    async def test_benign_resource_does_not_tighten(self):
        # suspicion 0.1 → reputation 0.9 > floor 0.3 → NO crossing, stays NORMAL.
        loop, _ = await self._run(_FixedReputationEnricher(0.9), _FLOOR)
        assert loop._degradation_engine.state.level == DegradationLevel.NORMAL


# ── flagged never a feature — invariant A-7 ───────────────────────────────────

class TestFlaggedNotAFeature:
    def test_normalized_intent_has_no_flagged_field(self):
        field_names = {f.name for f in dataclasses.fields(NormalizedIntent)}
        assert "flagged" not in field_names

    def test_enricher_does_not_add_flagged_field(self, tmp_path):
        snap = ReputationSnapshot(
            version=1, generated_at=time.time(),
            resource_reputation={"/data/r.txt": 0.9},
            container_reputation={},
        ).with_checksum()
        atomic_swap(tmp_path, snap)
        enricher = SnapshotIntentEnricher.from_dir(tmp_path)
        result = enricher.enrich(_make_normalized(after_external_read=True), _make_intent("/data/r.txt"))
        field_names = {f.name for f in dataclasses.fields(result)}
        assert "flagged" not in field_names

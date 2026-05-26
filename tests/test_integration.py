"""
Integration tests for Phase 1 enforcement and Phase 2 gate.

Covers invariants A-6, A-7, A-11, A-12 and spec adversarial test matrix:
  - Phase 1: reputation >= 0.8 + external read → deterministic deny before ML (2 variants)
  - Phase 2 gate: reputation floats not passed to Layer 2 until N_MIN_REAL_ATTACKS met (2 variants)
  - flagged never passed as feature (A-7)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Need axor-core on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "axor-core"))

from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult, NormalizedIntent
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import (
    ContextFragment, ContextView, LineageSummary,
)
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.capability.executor import CapabilityExecutor
from axor_core.policy.presets import standard as standard_policy
from axor_sentinel.integration.intent_enricher import SnapshotIntentEnricher
from axor_sentinel.integration.phase import (
    N_MIN_REAL_ATTACKS,
    MAX_FP_RATE,
    is_phase2_active,
)
from axor_sentinel.sentinel.snapshot import ReputationSnapshot


def _make_envelope(allowed_tools: frozenset[str] = frozenset({"read"})) -> ExecutionEnvelope:
    """Build a minimal test envelope."""
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
        node_id="n1",
        task="test",
        context=ctx,
        policy=policy,
        capabilities=caps,
        export_contract=export,
        lineage=lineage,
        cancel_token=make_token(),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_normalized(
    after_external_read: bool = False,
    resource_rep: float = 0.0,
) -> NormalizedIntent:
    return NormalizedIntent(
        tool="read",
        operation="file_read",
        target_kind="workdir",
        destination_kind="none",
        provenance="repo",
        reads_secret_like_data=False,
        writes_outside_workdir=False,
        executes_generated_code=False,
        after_external_read=after_external_read,
        after_secret_access=False,
        data_flow="local_to_local",
        target_resource_reputation=resource_rep,
        target_container_reputation=0.0,
    )


def _make_intent(path: str = "/data/secret.txt") -> Intent:
    return Intent(
        kind=IntentKind.TOOL_CALL,
        payload={"tool": "read", "args": {"path": path}},
        node_id="n1",
    )


class _StubEnricher:
    """Stub enricher that injects a fixed reputation score."""
    def __init__(self, reputation: float):
        self._rep = reputation

    def enrich(self, normalized: NormalizedIntent, intent: Intent) -> NormalizedIntent:
        import dataclasses
        return dataclasses.replace(normalized, target_resource_reputation=self._rep)


# ── Phase 1 enforcement — 2 variants ──────────────────────────────────────────

class TestPhase1Enforcement:
    """
    Invariant A-11: reputation >= 0.8 AND after_external_read → deterministic deny
    before anomaly detector (Layer 2) is invoked.
    """

    @pytest.mark.asyncio
    async def test_phase1_deny_fires_before_anomaly_detector(self, tmp_path):
        """
        When resource reputation >= 0.8 and after_external_read=True,
        the Phase 1 rule must deny before the anomaly detector is called.
        Mock anomaly detector call count must be 0.
        """
        from axor_core.node.intent_loop import IntentLoop

        # Mock anomaly detector — should NOT be called
        mock_detector = MagicMock()
        mock_detector.score = AsyncMock(
            return_value=AnomalyResult(score=0.0, cls=AnomalyClass.NORMAL)
        )

        # Stub enricher injects high reputation
        enricher = _StubEnricher(reputation=0.85)

        # Mock capability executor — should NOT be reached
        mock_executor = MagicMock(spec=CapabilityExecutor)
        mock_executor.execute = AsyncMock(return_value={"result": "content"})

        loop = IntentLoop(
            capability_executor=mock_executor,
            trace_events=[],
            anomaly_detector=mock_detector,
            reputation_enricher=enricher,
        )
        envelope = _make_envelope(frozenset({"read"}))

        async def fake_stream():
            yield ExecutorEvent(
                kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": "read", "tool_use_id": "tu1", "args": {"path": "/data/secret.txt"}},
                node_id="n1",
            )
            yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id="n1")

        # Patch normalizer to return after_external_read=True; enricher sets rep to 0.85
        with patch.object(loop._normalizer, "normalize") as mock_normalize:
            mock_normalize.return_value = _make_normalized(after_external_read=True, resource_rep=0.0)
            [e async for e in loop.run(fake_stream(), envelope)]

        # Anomaly detector must NOT have been called (Phase 1 fired first, A-11)
        assert mock_detector.score.call_count == 0
        # Executor must NOT have been called (denied before execution)
        assert mock_executor.execute.call_count == 0

    @pytest.mark.asyncio
    async def test_phase1_no_trigger_below_threshold(self, tmp_path):
        """
        reputation=0.7 (< 0.8) with after_external_read=True → Phase 1 does NOT fire.
        Anomaly detector IS called normally.
        """
        from axor_core.node.intent_loop import IntentLoop

        mock_detector = MagicMock()
        mock_detector.score = AsyncMock(
            return_value=AnomalyResult(score=0.0, cls=AnomalyClass.NORMAL)
        )

        enricher = _StubEnricher(reputation=0.7)  # below threshold

        mock_executor = MagicMock(spec=CapabilityExecutor)
        mock_executor.execute = AsyncMock(return_value={"result": "ok"})

        loop = IntentLoop(
            capability_executor=mock_executor,
            trace_events=[],
            anomaly_detector=mock_detector,
            reputation_enricher=enricher,
        )
        envelope = _make_envelope(frozenset({"read"}))

        async def fake_stream():
            yield ExecutorEvent(
                kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": "read", "tool_use_id": "tu1", "args": {"path": "/data/r.txt"}},
                node_id="n1",
            )
            yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id="n1")

        with patch.object(loop._normalizer, "normalize") as mock_normalize:
            mock_normalize.return_value = _make_normalized(after_external_read=True)
            [e async for e in loop.run(fake_stream(), envelope)]

        # Anomaly detector SHOULD have been called (Phase 1 did not fire)
        assert mock_detector.score.call_count >= 1


# ── Phase 2 gate — 2 variants ─────────────────────────────────────────────────

class TestPhase2Gate:
    def test_phase2_not_active_below_min_attacks(self):
        """Invariant A-12: N < 50 confirmed incidents → Phase 2 inactive."""
        assert not is_phase2_active(49, 0.01)
        assert not is_phase2_active(0, 0.0)

    def test_phase2_not_active_high_fp_rate(self):
        """Phase 2 inactive if fp_rate > MAX_FP_RATE even with enough incidents."""
        assert not is_phase2_active(N_MIN_REAL_ATTACKS, MAX_FP_RATE + 0.001)

    def test_phase2_activates_when_conditions_met(self):
        """Phase 2 active when both conditions satisfied."""
        assert is_phase2_active(N_MIN_REAL_ATTACKS, MAX_FP_RATE)
        assert is_phase2_active(100, 0.01)


# ── flagged never a Layer 2 feature — invariant A-7 ───────────────────────────

class TestFlaggedNotAFeature:
    def test_normalized_intent_has_no_flagged_field(self):
        """
        Invariant A-7: NormalizedIntent has no 'flagged' field.
        The anomaly detector (Layer 2) cannot receive flagged as a feature because
        the field simply doesn't exist on NormalizedIntent.
        """
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(NormalizedIntent)}
        assert "flagged" not in field_names

    def test_enricher_does_not_add_flagged_field(self, tmp_path):
        """SnapshotIntentEnricher only populates the two reputation float fields."""
        import dataclasses
        import time
        from axor_sentinel.sentinel.snapshot import atomic_swap

        snap = ReputationSnapshot(
            version=1, generated_at=time.time(),
            resource_reputation={"/data/r.txt": 0.9},
            container_reputation={},
        ).with_checksum()
        atomic_swap(tmp_path, snap)

        enricher = SnapshotIntentEnricher.from_dir(tmp_path)
        ni = _make_normalized(after_external_read=True)
        intent = _make_intent("/data/r.txt")
        result = enricher.enrich(ni, intent)

        field_names = {f.name for f in dataclasses.fields(result)}
        assert "flagged" not in field_names
        # Only the two reputation fields should differ
        assert result.target_resource_reputation >= 0.0
        assert result.target_container_reputation >= 0.0

"""Attestation folded into the audit cycle: the exported snapshot descends over
an attested branch — one LEVEL in the deterministic codomain, effective_score
in the scalar telemetry — Neo4j is untouched, and a re-triggering value
re-heats (levels re-derive from evidence every cycle)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.attestation import AttestationRecord
from axor_sentinel.sentinel.cycle import SentinelCycle
from axor_sentinel.sentinel.evidence import Evidence
from axor_sentinel.sentinel.predicates import LEVEL_SUSPICION, ReputationLevel


class _FakeResult:
    def __init__(self, rows=None):
        self._rows = rows or []

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None

    def data(self):
        return self._rows


class _FakeNeo4j:
    """Returns fixed resource scores on read-back; swallows writes."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def run(self, query: str, **params):
        if "AS score" in query:  # RESOURCE_SCORES_QUERY read-back
            return _FakeResult([
                {"id": rid, "score": s} for rid, s in self._scores.items()
            ])
        return _FakeResult([])


@pytest.fixture
def cycle(tmp_path: Path) -> SentinelCycle:
    return SentinelCycle(
        neo4j_session=_FakeNeo4j({"res_hot": 0.86, "res_cool": 0.1}),
        snapshot_dir=tmp_path,
        agent_baselines={}, signal_history={}, prior_counts={},
    )


def _attest(rid: str, prior: float, aid="a1", revokes=None) -> AttestationRecord:
    return AttestationRecord(
        attestation_id=aid, operator="op_d", reason="investigated; ours",
        causal_root="root_1", prior_heat=prior, revokes=revokes, resource_id=rid,
    )


def test_reason_required(cycle: SentinelCycle) -> None:
    from axor_sentinel.sentinel.attestation import AttestationError

    bad = AttestationRecord(
        attestation_id="a", operator="op", reason="  ", causal_root="r",
        prior_heat=0.8, resource_id="res_hot",
    )
    with pytest.raises(AttestationError):
        cycle.attest(bad)


def _watch_evidence(now: float) -> Evidence:
    """One tainted staging fact — enough for a deterministic WATCH (W1)."""
    return Evidence(
        origin="agent_x", session_id="s1", rank=SignalType.READ_SUMMARIZE,
        tainted=True, observed_at=now, resolution="provider_id",
    )


def test_snapshot_without_attestation_exports_raw(cycle: SentinelCycle) -> None:
    # The wire codomain is deterministic levels now: with no evidence the
    # branch is CLEAN (absent from resource_reputation); the raw Neo4j
    # read-back survives untouched in the score telemetry.
    snap = cycle.run_once(sessions=[])
    assert "res_hot" not in snap.resource_reputation
    assert snap.resource_score_telemetry["res_hot"] == pytest.approx(0.86)


def test_attested_branch_descends_in_snapshot(cycle: SentinelCycle) -> None:
    cycle._evidence.add("res_hot", _watch_evidence(time.time()))
    before = cycle.run_once(sessions=[])
    assert before.resource_reputation["res_hot"] == pytest.approx(
        LEVEL_SUSPICION[ReputationLevel.WATCH]
    )

    cycle.attest(_attest("res_hot", prior=0.86))
    after = cycle.run_once(sessions=[])
    # WATCH descends to CLEAN on the wire; the event stays visible in the
    # facts (history, not laundering), and the scalar telemetry reads the
    # post-attestation residue. Untouched branches are unaffected.
    assert "res_hot" not in after.resource_reputation
    assert any(f == "A2:attested:a1" for f in after.verdict_facts["res_hot"])
    assert after.resource_score_telemetry["res_hot"] < 0.86
    assert "res_cool" not in after.resource_reputation


def test_revocation_restores_exported_score(cycle: SentinelCycle) -> None:
    cycle._evidence.add("res_hot", _watch_evidence(time.time()))
    cycle.attest(_attest("res_hot", prior=0.86, aid="a1"))
    down = cycle.run_once(sessions=[])
    assert "res_hot" not in down.resource_reputation

    # Revocation (same keyset) is itself a new event: the exported level and
    # the raw telemetry both come back — full history, both directions.
    cycle.attest(_attest("res_hot", prior=0.0, aid="a2", revokes="a1"))
    restored = cycle.run_once(sessions=[])
    assert restored.resource_reputation["res_hot"] == pytest.approx(
        LEVEL_SUSPICION[ReputationLevel.WATCH]
    )
    assert restored.resource_score_telemetry["res_hot"] == pytest.approx(0.86)

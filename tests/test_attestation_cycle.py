"""Attestation folded into the audit cycle: the exported snapshot descends over
an attested branch, Neo4j is untouched, and a re-triggering value re-heats."""
from __future__ import annotations

from pathlib import Path

import pytest

from axor_sentinel.sentinel.attestation import AttestationRecord
from axor_sentinel.sentinel.cycle import SentinelCycle


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


def test_snapshot_without_attestation_exports_raw(cycle: SentinelCycle) -> None:
    snap = cycle.run_once(sessions=[])
    assert snap.resource_reputation["res_hot"] == pytest.approx(0.86)


def test_attested_branch_descends_in_snapshot(cycle: SentinelCycle) -> None:
    cycle.attest(_attest("res_hot", prior=0.86))
    snap = cycle.run_once(sessions=[])
    assert snap.resource_reputation["res_hot"] == pytest.approx(0.0)
    # untouched branch unaffected
    assert snap.resource_reputation["res_cool"] == pytest.approx(0.1)


def test_revocation_restores_exported_score(cycle: SentinelCycle) -> None:
    cycle.attest(_attest("res_hot", prior=0.86, aid="a1"))
    cycle.attest(_attest("res_hot", prior=0.0, aid="a2", revokes="a1"))
    snap = cycle.run_once(sessions=[])
    assert snap.resource_reputation["res_hot"] == pytest.approx(0.86)

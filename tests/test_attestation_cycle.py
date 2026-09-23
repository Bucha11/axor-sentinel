"""Attestation folded into the audit cycle: the exported snapshot descends over
an attested branch — one LEVEL in the deterministic codomain, effective_score
in the scalar telemetry — Neo4j is untouched, and a re-triggering value
re-heats (levels re-derive from evidence every cycle)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest

from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.attestation import AttestationRecord
from axor_sentinel.sentinel.cycle import SentinelCycle
from axor_sentinel.sentinel.evidence import Evidence
from axor_sentinel.sentinel.predicates import LEVEL_SUSPICION, ReputationLevel
from axor_sentinel.sentinel.snapshot import SNAPSHOT_KEY_ENV
from tests.test_cycle import _MockNeo4j, _session


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


# ── Supersession, fact naming, persistence ──────────────────────────────────
#
# These drive real verdicts through run_once with _MockNeo4j (test_cycle.py):
# sessions whose accesses are READ_EXPORT_FAILED fire P1 → FLAGGED, so an
# attestation's one-level descent and its lapse are both observable on the
# wire. Session start times are set explicitly; supersession compares an
# attestation against Evidence.known_at = max(session start, INGEST time), and
# the ingest time is the cycle's clock. So an attestation meant to cover the
# evidence already ingested is stamped by attest() itself (created_at=0.0 →
# the cycle clock, which is never earlier than a previous cycle's), not
# back-dated: back-dating before an ingest now (by design) makes it lapse.

_RID = "res_x"


def _export_denied(agent: str, started_at: float):
    s = _session(
        agent, [(_RID, "c1", 1.0, SignalType.READ_EXPORT_FAILED)], had_failed=True,
    )
    return dataclasses.replace(s, session_id=f"s_{agent}", started_at=started_at)


def _rec(aid: str, created_at: float, revokes=None, org="") -> AttestationRecord:
    return AttestationRecord(
        attestation_id=aid, operator="op_d", reason="investigated; ours",
        causal_root="root_1", prior_heat=0.0 if revokes else 0.9,
        revokes=revokes, resource_id=_RID, org=org, created_at=created_at,
    )


def _mock_cycle(tmp_path: Path, explicit: bool = True) -> SentinelCycle:
    if explicit:
        return SentinelCycle(
            _MockNeo4j(), tmp_path,
            agent_baselines={}, signal_history={}, prior_counts={},
        )
    return SentinelCycle(_MockNeo4j(), tmp_path)  # restores from disk


def test_newer_evidence_supersedes_attestation(tmp_path: Path) -> None:
    # The reproduced bug: FLAGGED → attest → WATCH, then a fresh export-denied
    # session from a DIFFERENT origin (P1 + P2 fire) used to stay WATCH (0.4,
    # below core's 0.3 floor) forever. Newer evidence now lifts the discount.
    now = time.time()
    c = _mock_cycle(tmp_path)
    flagged = c.run_once(sessions=[_export_denied("agent_a", now - 100)])
    assert flagged.resource_level[_RID] == "FLAGGED"

    c.attest(_rec("a1", created_at=0.0))
    attested = c.run_once(sessions=[])
    assert attested.resource_level[_RID] == "WATCH"
    assert "A2:attested:a1" in attested.verdict_facts[_RID]

    # Started after the attestation (fact time alone would supersede).
    refired = c.run_once(sessions=[_export_denied("agent_b", time.time() + 1)])
    assert refired.resource_level[_RID] == "FLAGGED"
    assert refired.resource_reputation[_RID] == pytest.approx(
        LEVEL_SUSPICION[ReputationLevel.FLAGGED]
    )
    facts = refired.verdict_facts[_RID]
    assert any(f.startswith("P2:") for f in facts)
    assert "A2:attested:a1" not in facts
    assert "A2:attestation_superseded_by_newer_evidence:a1" in facts
    # History stays: the superseded attestation is still on record.
    assert [r.attestation_id for r in c.attestations_for(_RID)] == ["a1"]


def test_evidence_at_attestation_instant_does_not_supersede(tmp_path: Path) -> None:
    # Strictly newer: a fact stamped at the attestation instant was visible
    # to the operator.
    # The instant compared is the fact's known_at (max of session start and
    # ingest time), so the attestation is stamped exactly at that.
    now = time.time()
    c = _mock_cycle(tmp_path)
    c.run_once(sessions=[_export_denied("agent_a", now - 10)])
    (ev,) = c._evidence.evidence_for(_RID)
    c.attest(_rec("a1", created_at=ev.known_at))
    assert c.run_once(sessions=[]).resource_level[_RID] == "WATCH"


def test_attest_stamps_created_at_and_refuses_future(tmp_path: Path) -> None:
    c = _mock_cycle(tmp_path)
    before = time.time()
    c.attest(_rec("unset", created_at=0.0))
    # A future timestamp would make every later fact look "older" and pin the
    # discount on forever — it is replaced by the cycle clock.
    c.attest(_rec("future", created_at=before + 10 * 86400))
    c.attest(_rec("past", created_at=before - 5))
    after = time.time()
    by_id = {r.attestation_id: r for r in c.attestations_for(_RID)}
    assert before <= by_id["unset"].created_at <= after
    assert before <= by_id["future"].created_at <= after
    assert by_id["past"].created_at == before - 5


def test_fact_names_applied_attestation_not_revocation(tmp_path: Path) -> None:
    # a1, a2 attest; a3 revokes a2 → the newest record (records[0]) is the
    # revocation a3, but the attestation actually applied is a1.
    now = time.time()
    c = _mock_cycle(tmp_path)
    c.run_once(sessions=[_export_denied("agent_a", now - 100)])
    c.attest(_rec("a1", created_at=0.0))
    c.attest(_rec("a2", created_at=0.0))
    c.attest(_rec("a3", created_at=0.0, revokes="a2"))
    assert c.attestations_for(_RID)[0].attestation_id == "a3"
    snap = c.run_once(sessions=[])
    assert snap.resource_level[_RID] == "WATCH"
    facts = snap.verdict_facts[_RID]
    assert "A2:attested:a1" in facts
    assert not any(f in facts for f in ("A2:attested:a3", "A2:attested:a2"))


def test_attestations_survive_restart_signed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "k")
    now = time.time()
    c = _mock_cycle(tmp_path, explicit=False)
    c.run_once(sessions=[_export_denied("agent_a", now - 100)])
    c.attest(_rec("a1", created_at=0.0, org="acme"))
    c.attest(_rec("a2", created_at=0.0, revokes="a1", org="rogue"))
    assert c.run_once(sessions=[]).resource_level[_RID] == "WATCH"

    # The signed envelope carries them (and the payload is HMAC-covered).
    raw = json.loads((tmp_path / "sentinel_state.json").read_text())
    assert raw["_signed"] is True
    assert "attestations" in json.loads(raw["payload"])

    restarted = _mock_cycle(tmp_path, explicit=False)
    assert restarted.attestations_for(_RID) == c.attestations_for(_RID)
    snap = restarted.run_once(sessions=[])
    # Before the fix the restart dropped a1 and the level jumped to FLAGGED.
    assert snap.resource_level[_RID] == "WATCH"
    assert "A2:attested:a1" in snap.verdict_facts[_RID]


def test_tampered_state_restores_no_attestations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "k")
    c = _mock_cycle(tmp_path, explicit=False)
    c.attest(_rec("a1", created_at=time.time() - 5))
    c.save_state()
    p = tmp_path / "sentinel_state.json"
    env = json.loads(p.read_text())
    env["sig"] = "deadbeef"
    p.write_text(json.dumps(env))
    assert SentinelCycle.load_attestations(p) == {}
    assert _mock_cycle(tmp_path, explicit=False).attestations_for(_RID) == []


def test_state_without_attestations_is_backward_compatible(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
    monkeypatch.delenv("AXOR_ENV", raising=False)
    monkeypatch.delenv("AXOR_SNAPSHOT_REQUIRE_SIGNATURE", raising=False)
    p = tmp_path / "sentinel_state.json"
    p.write_text(json.dumps({  # a pre-attestation state file
        "version": 4, "signal_history": {}, "prior_counts": {}, "baselines": {},
    }))
    assert SentinelCycle.load_attestations(p) == {}
    restored = _mock_cycle(tmp_path, explicit=False)
    assert restored.attestations_for(_RID) == []
    assert restored.run_once(sessions=[]).version == 5


def test_invalid_persisted_attestation_is_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
    monkeypatch.delenv("AXOR_ENV", raising=False)
    monkeypatch.delenv("AXOR_SNAPSHOT_REQUIRE_SIGNATURE", raising=False)
    good = _rec("a1", created_at=1.0).to_json()
    bad = dict(good, attestation_id="a0", reason="   ")
    p = tmp_path / "sentinel_state.json"
    p.write_text(json.dumps({"version": 1, "attestations": {_RID: [good, bad]}}))
    loaded = SentinelCycle.load_attestations(p)
    assert [r.attestation_id for r in loaded[_RID]] == ["a1"]


# ── Supersession by INGEST time, not only session start ─────────────────────

def test_late_reported_session_supersedes_attestation(tmp_path: Path) -> None:
    # A session that STARTED before the attestation but was only reported
    # (ingested) after it is evidence the operator never saw. Compared by
    # session start alone it left the discount in place.
    now = time.time()
    c = _mock_cycle(tmp_path)
    c.run_once(sessions=[_export_denied("agent_a", now - 100)])
    c.attest(_rec("a1", created_at=0.0))            # stamped now
    assert c.run_once(sessions=[]).resource_level[_RID] == "WATCH"

    late = _export_denied("agent_b", now - 50)      # started BEFORE a1
    snap = c.run_once(sessions=[late])
    assert snap.resource_level[_RID] == "FLAGGED"
    assert "A2:attestation_superseded_by_newer_evidence:a1" in snap.verdict_facts[_RID]


def test_replayed_session_does_not_supersede(tmp_path: Path) -> None:
    # Re-reporting an ALREADY-ingested session is no new knowledge: the store
    # keeps the first record (and its ingest time), so the discount stands.
    now = time.time()
    c = _mock_cycle(tmp_path)
    sess = _export_denied("agent_a", now - 100)
    c.run_once(sessions=[sess])
    c.attest(_rec("a1", created_at=0.0))
    assert c.run_once(sessions=[sess]).resource_level[_RID] == "WATCH"


def test_window_still_uses_session_time(tmp_path: Path) -> None:
    # Ingest time does not extend a fact's life: a session that started
    # outside the window is pruned even though it was ingested just now.
    c = _mock_cycle(tmp_path)
    ancient = _export_denied("agent_a", time.time() - 40 * 86400)
    snap = c.run_once(sessions=[ancient])
    assert _RID not in snap.resource_level


def test_evidence_ingest_time_is_backward_compatible() -> None:
    legacy = {  # persisted before ingested_at existed
        "origin": "o", "session_id": "s", "rank": SignalType.READ.value,
        "tainted": True, "observed_at": 123.0, "resolution": "path",
    }
    ev = Evidence.from_json(legacy)
    assert ev.ingested_at == 0.0 and ev.known_at == 123.0
    fresh = dataclasses.replace(ev, ingested_at=456.0)
    assert Evidence.from_json(fresh.to_json()) == fresh
    assert fresh.known_at == 456.0

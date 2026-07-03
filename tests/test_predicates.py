"""Truth tables for the deterministic verdict layer (evidence + predicates)."""
from __future__ import annotations

from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.evidence import (
    Evidence,
    EvidenceStore,
    resolution_from_confidence,
)
from axor_sentinel.sentinel.predicates import (
    LEVEL_SUSPICION,
    ReputationLevel,
    SentinelPolicy,
    Verdict,
    evaluate_container,
    evaluate_resource,
    fanout_exceeded,
)

_POLICY = SentinelPolicy()
_NOW = 1_000_000.0
_DAY = 86400.0


def _ev(
    origin: str = "agent_a",
    session: str = "s1",
    rank: SignalType = SignalType.READ_SUMMARIZE,
    tainted: bool = True,
    age_days: float = 1.0,
    resolution: str = "provider_id",
) -> Evidence:
    return Evidence(
        origin=origin, session_id=session, rank=rank, tainted=tainted,
        observed_at=_NOW - age_days * _DAY, resolution=resolution,
    )


# ── P1: export denied ──────────────────────────────────────────────────────────

def test_p1_single_export_failed_flags():
    v = evaluate_resource([_ev(rank=SignalType.READ_EXPORT_FAILED)], _POLICY, _NOW)
    assert v.level == ReputationLevel.FLAGGED
    assert any(f.startswith("P1:") for f in v.facts)


# ── P2: distinct-origin export adjacency ───────────────────────────────────────

def test_p2_two_origins_flag_one_origin_does_not():
    one = [
        _ev(origin="a", session="s1", rank=SignalType.READ_EXPORT_ADJACENT),
        _ev(origin="a", session="s2", rank=SignalType.READ_EXPORT_ADJACENT),
    ]
    assert evaluate_resource(one, _POLICY, _NOW).level == ReputationLevel.WATCH

    two = [
        _ev(origin="a", session="s1", rank=SignalType.READ_EXPORT_ADJACENT),
        _ev(origin="b", session="s2", rank=SignalType.READ_EXPORT_ADJACENT),
    ]
    v = evaluate_resource(two, _POLICY, _NOW)
    assert v.level == ReputationLevel.FLAGGED
    assert any(f.startswith("P2:") for f in v.facts)


# ── P3: staging session count (distinctness = anti-poisoning) ──────────────────

def test_p3_three_distinct_tainted_sessions_flag():
    ev = [_ev(session=f"s{i}") for i in range(3)]
    v = evaluate_resource(ev, _POLICY, _NOW)
    assert v.level == ReputationLevel.FLAGGED
    assert any(f.startswith("P3:") for f in v.facts)


def test_p3_repeats_from_same_session_do_not_flag():
    # The set-semantics version of origin_dampening: replays collapse.
    ev = [_ev(session="s1")] * 10
    assert evaluate_resource(ev, _POLICY, _NOW).level == ReputationLevel.WATCH


def test_p3_untainted_sessions_do_not_count():
    ev = [_ev(session=f"s{i}", tainted=False) for i in range(5)]
    assert evaluate_resource(ev, _POLICY, _NOW).level == ReputationLevel.CLEAN


def test_p3_plain_reads_do_not_count():
    ev = [_ev(session=f"s{i}", rank=SignalType.READ) for i in range(5)]
    assert evaluate_resource(ev, _POLICY, _NOW).level == ReputationLevel.CLEAN


# ── P4: staged-then-export sequence ────────────────────────────────────────────

def test_p4_read_then_later_export_flags():
    ev = [
        _ev(session="s1", rank=SignalType.READ, age_days=5.0),
        _ev(session="s2", rank=SignalType.READ_EXPORT_ADJACENT, tainted=False, age_days=1.0),
    ]
    v = evaluate_resource(ev, _POLICY, _NOW)
    assert v.level == ReputationLevel.FLAGGED
    assert any(f.startswith("P4:") for f in v.facts)


def test_p4_export_before_read_does_not_fire():
    ev = [
        _ev(session="s1", rank=SignalType.READ, age_days=1.0),
        _ev(session="s2", rank=SignalType.READ_EXPORT_ADJACENT, tainted=False, age_days=5.0),
    ]
    v = evaluate_resource(ev, _POLICY, _NOW)
    assert not any(f.startswith("P4:") for f in v.facts)


def test_p4_same_session_does_not_fire():
    ev = [
        _ev(session="s1", rank=SignalType.READ, age_days=5.0),
        Evidence(origin="agent_a", session_id="s1",
                 rank=SignalType.READ_EXPORT_ADJACENT, tainted=True,
                 observed_at=_NOW - 1.0 * _DAY, resolution="provider_id"),
    ]
    v = evaluate_resource(ev, _POLICY, _NOW)
    assert not any(f.startswith("P4:") for f in v.facts)


# ── Window: exact TTL replaces decay ───────────────────────────────────────────

def test_expired_evidence_is_ignored():
    ev = [_ev(session=f"s{i}", age_days=31.0) for i in range(5)]
    assert evaluate_resource(ev, _POLICY, _NOW).level == ReputationLevel.CLEAN


def test_evidence_inside_window_is_monotone():
    inside = [_ev(session=f"s{i}", age_days=29.0) for i in range(3)]
    assert evaluate_resource(inside, _POLICY, _NOW).level == ReputationLevel.FLAGGED


# ── Resolution ceiling ─────────────────────────────────────────────────────────

def test_heuristic_only_evidence_caps_at_watch():
    ev = [_ev(session=f"s{i}", resolution="heuristic") for i in range(3)]
    v = evaluate_resource(ev, _POLICY, _NOW)
    assert v.level == ReputationLevel.WATCH
    assert any(f.startswith("heuristic_ceiling:") for f in v.facts)


def test_resolution_tiers_recovered_from_confidence():
    assert resolution_from_confidence(1.0) == "provider_id"
    assert resolution_from_confidence(0.7) == "path"
    assert resolution_from_confidence(0.4) == "heuristic"


# ── WATCH tier & CLEAN ─────────────────────────────────────────────────────────

def test_single_tainted_staging_is_watch():
    v = evaluate_resource([_ev()], _POLICY, _NOW)
    assert v == Verdict(ReputationLevel.WATCH, ("W1:tainted_staging_sessions=['s1']",))


def test_no_evidence_is_clean():
    assert evaluate_resource([], _POLICY, _NOW).level == ReputationLevel.CLEAN


# ── Containers: counting replaces mean-over-threshold ──────────────────────────

def test_container_two_flagged_members_flag():
    lv = [ReputationLevel.FLAGGED, ReputationLevel.FLAGGED, ReputationLevel.CLEAN]
    assert evaluate_container(lv, _POLICY).level == ReputationLevel.FLAGGED


def test_container_one_flagged_member_is_watch():
    lv = [ReputationLevel.FLAGGED, ReputationLevel.CLEAN]
    assert evaluate_container(lv, _POLICY).level == ReputationLevel.WATCH


def test_container_all_clean_is_clean():
    assert evaluate_container(
        [ReputationLevel.CLEAN, ReputationLevel.CLEAN], _POLICY
    ).level == ReputationLevel.CLEAN


# ── Fanout quota (replaces the z-score baseline) ───────────────────────────────

def test_fanout_quota_exceeded():
    containers = [f"c{i}" for i in range(8)]
    assert fanout_exceeded(True, containers, SignalType.READ_SUMMARIZE, _POLICY)


def test_fanout_within_quota_or_untainted_or_low_rank():
    many = [f"c{i}" for i in range(8)]
    few = [f"c{i}" for i in range(7)]
    assert not fanout_exceeded(True, few, SignalType.READ_SUMMARIZE, _POLICY)
    assert not fanout_exceeded(False, many, SignalType.READ_SUMMARIZE, _POLICY)
    assert not fanout_exceeded(True, many, SignalType.READ, _POLICY)


def test_fanout_duplicate_containers_collapse():
    assert not fanout_exceeded(True, ["c1"] * 100, SignalType.READ_SUMMARIZE, _POLICY)


# ── EvidenceStore: dedup, prune, round-trip ────────────────────────────────────

def test_store_dedupes_session_rank_replays():
    store = EvidenceStore()
    for _ in range(5):
        store.add("r1", _ev(session="s1"))
    assert len(store.evidence_for("r1")) == 1


def test_store_prune_drops_expired_and_empty_resources():
    store = EvidenceStore()
    store.add("r1", _ev(session="s1", age_days=40.0))
    store.add("r2", _ev(session="s2", age_days=1.0))
    store.prune(_NOW, _POLICY.window_days)
    assert store.resource_ids() == ("r2",)


def test_store_json_round_trip():
    store = EvidenceStore()
    store.add("r1", _ev(session="s1"))
    store.add("r1", _ev(session="s2", rank=SignalType.READ_EXPORT_FAILED))
    restored = EvidenceStore.from_json(store.to_json())
    assert set(restored.evidence_for("r1")) == set(store.evidence_for("r1"))


def test_store_corrupt_json_cold_starts():
    assert EvidenceStore.from_json({"r1": [{"bad": True}]}).resource_ids() == ()


# ── Wire codomain: decidable end-to-end through core's floor ───────────────────

def test_level_suspicion_codomain_is_finite_and_ordered():
    assert LEVEL_SUSPICION[ReputationLevel.CLEAN] == 0.0
    assert LEVEL_SUSPICION[ReputationLevel.WATCH] < LEVEL_SUSPICION[ReputationLevel.FLAGGED]


def test_polarity_round_trip_through_enricher_conversion():
    from axor_sentinel.integration.intent_enricher import _suspicion_to_reputation

    floor = 0.3  # the existing operator default keeps its exact meaning
    rep_flagged = _suspicion_to_reputation(LEVEL_SUSPICION[ReputationLevel.FLAGGED])
    rep_watch = _suspicion_to_reputation(LEVEL_SUSPICION[ReputationLevel.WATCH])
    rep_clean = _suspicion_to_reputation(LEVEL_SUSPICION[ReputationLevel.CLEAN])
    assert 0.0 < rep_flagged <= floor          # FLAGGED crosses → tightens
    assert rep_watch > floor                   # WATCH does not cross at 0.3
    assert rep_clean == 0.0                    # CLEAN stays "unknown"
    assert 0.0 < rep_flagged <= 0.6 and rep_watch <= 0.6  # floor 0.6 → WATCH also crosses

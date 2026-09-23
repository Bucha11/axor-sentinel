"""Per-peer reputation (spec v2 Ch.1 §2): heat accrues to the authenticated
peer identity — attribution is what L1 buys; L0 names are spoofable and never
accrue. Scoring reuses the sentinel invariants (A-1 log accumulation, A-3
half-life decay)."""
from __future__ import annotations

from axor_sentinel.sentinel.peer_reputation import PEER_SIGNAL_WEIGHTS, PeerReputation


def test_forged_assertion_heats_the_peer() -> None:
    rep = PeerReputation()
    s = rep.record_signal("partner", "assertion_forged", identity_verified=True)
    assert s == PEER_SIGNAL_WEIGHTS["assertion_forged"]


def test_unverified_identity_never_accrues() -> None:
    """An L0 peer id is attacker-chosen — heat on it would be heat the
    attacker controls (poison a name, then rotate)."""
    rep = PeerReputation()
    assert rep.record_signal("spoofed", "assertion_forged", identity_verified=False) == 0.0
    assert rep.score("spoofed") == 0.0
    assert rep.snapshot() == {}


def test_accumulation_is_logarithmic_and_bounded() -> None:
    rep = PeerReputation()
    for _ in range(50):
        s = rep.record_signal("partner", "assertion_forged", identity_verified=True)
    assert s <= 1.0  # invariant A-1
    # two weak probes < one forged assertion
    rep2 = PeerReputation()
    rep2.record_signal("p2", "class_probe", identity_verified=True)
    weak2 = rep2.record_signal("p2", "class_probe", identity_verified=True)
    assert weak2 < PEER_SIGNAL_WEIGHTS["assertion_forged"]


def test_half_life_decay() -> None:
    rep = PeerReputation()
    rep.record_signal("partner", "assertion_forged", identity_verified=True, at_days=0.0)
    assert abs(rep.score("partner", at_days=30.0) - 0.25) < 1e-9  # 0.5 * 0.5


def test_unknown_signal_kind_is_ignored() -> None:
    rep = PeerReputation()
    assert rep.record_signal("partner", "mystery", identity_verified=True) == 0.0
    assert rep.snapshot() == {}


def test_snapshot_is_quiet_until_wrong() -> None:
    rep = PeerReputation()
    rep.record_signal("noisy", "send_denied", identity_verified=True)
    snap = rep.snapshot()
    assert list(snap) == ["noisy"] and snap["noisy"]["signals"] == ["send_denied"]


def test_out_of_order_signal_does_not_double_decay() -> None:
    # In order: A@0, B@30. Delivered B then A. The record's clock must not move
    # back to day 0 — that decayed the whole score again across days 0..30.
    in_order = PeerReputation()
    in_order.record_signal("p", "assertion_forged", identity_verified=True, at_days=0.0)
    in_order.record_signal("p", "send_denied", identity_verified=True, at_days=30.0)

    late = PeerReputation()
    late.record_signal("p", "send_denied", identity_verified=True, at_days=30.0)
    late.record_signal("p", "assertion_forged", identity_verified=True, at_days=0.0)

    expected = in_order.score("p", at_days=60.0)
    assert abs(expected - 0.4375 * 0.5) < 1e-9
    assert abs(late.score("p", at_days=60.0) - expected) < 1e-9


def test_signal_log_is_windowed_and_capped() -> None:
    from axor_sentinel.sentinel.peer_reputation import (
        PEER_SIGNAL_CAP,
        PEER_SIGNAL_WINDOW_DAYS,
    )

    rep = PeerReputation()
    for i in range(PEER_SIGNAL_CAP * 3):
        rep.record_signal("p", "class_probe", identity_verified=True, at_days=i * 0.001)
    assert len(rep.snapshot()["p"]["signals"]) == PEER_SIGNAL_CAP

    rep.record_signal(
        "p", "send_denied", identity_verified=True,
        at_days=PEER_SIGNAL_WINDOW_DAYS + 10.0,
    )
    assert rep.snapshot()["p"]["signals"] == ["send_denied"]


def test_snapshot_decays_to_now_when_given() -> None:
    rep = PeerReputation()
    rep.record_signal("p", "assertion_forged", identity_verified=True, at_days=0.0)
    assert rep.snapshot()["p"]["score"] == 0.5          # as of the last signal
    assert abs(rep.snapshot(at_days=30.0)["p"]["score"] - 0.25) < 1e-9
    assert rep.snapshot(at_days=30.0)["p"]["score"] == rep.score("p", at_days=30.0)

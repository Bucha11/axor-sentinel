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

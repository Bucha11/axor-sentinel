"""Per-peer reputation — the inter-federation Sentinel scope (spec v2 Ch.1 §2).

Across a federation boundary Sentinel sees ONE stable thing about a foreign
agent: its authenticated identity. Heat therefore accrues to the peer identity
— this is precisely what L1 buys ("attribution, not trust"). Foreign nodes
stay opaque; we never score a foreign agent's *integrity* (decision v2-9), we
score the observable behavior of OUR edge to it: forged label assertions,
denied sends, discount-class probing.

Reuses the sentinel scoring invariants: logarithmic accumulation (A-1, two
weak signals never equal one strong one) and 30-day half-life decay (A-3).
Unverified identities (L0) accrue nothing — an unauthenticated peer id is
attacker-chosen, and heat on a spoofable name is heat the attacker controls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from axor_sentinel.sentinel.weight import accumulate, apply_decay_to_score

# Signal weights, strongest first. A forged assertion is an *active* attempt
# to launder provenance through our trust ladder; a denied send is our gate
# holding; a class probe is an L2 peer nosing outside its discount scope.
PEER_SIGNAL_WEIGHTS: dict[str, float] = {
    "assertion_forged": 0.5,
    "send_denied": 0.25,
    "class_probe": 0.1,
}


@dataclass
class _PeerRecord:
    score: float = 0.0
    last_decay_at: float = 0.0  # epoch days granularity is the caller's choice
    signals: list[str] = field(default_factory=list)


class PeerReputation:
    """Heat per authenticated peer identity. Observe-only — feeds the topology
    badge and (via the platform) operator attention; never gates directly."""

    def __init__(self) -> None:
        self._peers: dict[str, _PeerRecord] = {}

    def record_signal(
        self,
        peer_id: str,
        kind: str,
        *,
        identity_verified: bool,
        at_days: float = 0.0,
    ) -> float:
        """Fold one signal; returns the peer's new score. Unknown signal kinds
        are ignored (weight 0) rather than guessed — fail quiet, not loud, on
        telemetry. L0 (unverified) identities never accrue."""
        if not identity_verified:
            return 0.0
        weight = PEER_SIGNAL_WEIGHTS.get(kind, 0.0)
        rec = self._peers.setdefault(peer_id, _PeerRecord(last_decay_at=at_days))
        rec.score = apply_decay_to_score(rec.score, max(0.0, at_days - rec.last_decay_at))
        rec.last_decay_at = at_days
        if weight > 0.0:
            rec.score = accumulate(rec.score, weight)
            rec.signals.append(kind)
        return rec.score

    def score(self, peer_id: str, *, at_days: float | None = None) -> float:
        rec = self._peers.get(peer_id)
        if rec is None:
            return 0.0
        if at_days is None:
            return rec.score
        return apply_decay_to_score(rec.score, max(0.0, at_days - rec.last_decay_at))

    def snapshot(self) -> dict[str, dict]:
        """Peers with non-zero heat only (quiet-until-wrong applied to the
        reputation surface)."""
        return {
            pid: {"score": rec.score, "signals": list(rec.signals)}
            for pid, rec in sorted(self._peers.items())
            if rec.score > 0.0
        }

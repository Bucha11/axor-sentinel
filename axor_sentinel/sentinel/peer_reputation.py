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

from axor_sentinel.sentinel.weight import accumulate, apply_decay_to_score, time_decay

# Signal weights, strongest first. A forged assertion is an *active* attempt
# to launder provenance through our trust ladder; a denied send is our gate
# holding; a class probe is an L2 peer nosing outside its discount scope.
PEER_SIGNAL_WEIGHTS: dict[str, float] = {
    "assertion_forged": 0.5,
    "send_denied": 0.25,
    "class_probe": 0.1,
}


# Retention of the per-peer signal log (the ``signals`` list the snapshot shows).
# The score itself is a decayed scalar and never grows; the log did, one entry
# per signal forever. It is kept for the window in which a signal still carries
# material heat — three half-lives, ≤ 12.5% of its weight left — and hard-capped
# so a burst inside that window cannot grow it without bound either.
PEER_SIGNAL_WINDOW_DAYS: float = 90.0
PEER_SIGNAL_CAP: int = 256


@dataclass
class _PeerRecord:
    score: float = 0.0
    # Time the score is expressed at. Monotone: it never moves backwards (see
    # record_signal), so decay is applied to each interval exactly once.
    last_decay_at: float = 0.0  # epoch days granularity is the caller's choice
    signals: list[str] = field(default_factory=list)
    signal_at: list[float] = field(default_factory=list)  # parallel to signals

    def prune_signals(self) -> None:
        """Drop log entries older than the window (relative to the newest time
        the record has seen) and keep at most PEER_SIGNAL_CAP, newest last."""
        horizon = self.last_decay_at - PEER_SIGNAL_WINDOW_DAYS
        kept = [
            (k, t) for k, t in zip(self.signals, self.signal_at, strict=True)
            if t >= horizon
        ][-PEER_SIGNAL_CAP:]
        self.signals = [k for k, _ in kept]
        self.signal_at = [t for _, t in kept]


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
        telemetry. L0 (unverified) identities never accrue.

        Signals may arrive out of order. The record's clock (last_decay_at)
        only moves FORWARD: an in-order signal decays the score up to
        ``at_days`` and then accumulates; a LATE signal (``at_days`` before the
        record's clock) leaves the clock where it is and accumulates its weight
        pre-aged by its own age, ``weight × time_decay(last_decay_at −
        at_days)``. Moving the clock back to the late signal's time (the old
        behaviour) made the next in-order signal decay the whole score across
        an interval it had already been decayed over — double decay. (Folding a
        pre-aged weight is exact for a single signal; with accumulate's
        saturation it is the close, monotone approximation.) The returned score
        is expressed at the record's clock, i.e. max(at_days seen so far).
        """
        if not identity_verified:
            return 0.0
        weight = PEER_SIGNAL_WEIGHTS.get(kind, 0.0)
        rec = self._peers.setdefault(peer_id, _PeerRecord(last_decay_at=at_days))
        if at_days >= rec.last_decay_at:
            rec.score = apply_decay_to_score(rec.score, at_days - rec.last_decay_at)
            rec.last_decay_at = at_days
            effective = weight
        else:
            effective = weight * time_decay(rec.last_decay_at - at_days)
        if weight > 0.0:
            rec.score = accumulate(rec.score, effective)
            rec.signals.append(kind)
            rec.signal_at.append(at_days)
            rec.prune_signals()
        return rec.score

    def score(self, peer_id: str, *, at_days: float | None = None) -> float:
        rec = self._peers.get(peer_id)
        if rec is None:
            return 0.0
        if at_days is None:
            return rec.score
        return apply_decay_to_score(rec.score, max(0.0, at_days - rec.last_decay_at))

    def snapshot(self, *, at_days: float | None = None) -> dict[str, dict]:
        """Peers with non-zero heat only (quiet-until-wrong applied to the
        reputation surface).

        Pass ``at_days`` (the caller's "now", same clock as record_signal) to
        get every score decayed to that moment, consistent with
        ``score(peer, at_days=...)``. Omitted, each score is as of that peer's
        LAST signal (undecayed since) — two peers' numbers are then taken at
        different times and are not comparable; kept as the default only for
        backward compatibility. ``signals`` is the retained log (window
        PEER_SIGNAL_WINDOW_DAYS, at most PEER_SIGNAL_CAP entries), not the full
        history.
        """
        out: dict[str, dict] = {}
        for pid, rec in sorted(self._peers.items()):
            score = rec.score
            if at_days is not None:
                score = apply_decay_to_score(score, max(0.0, at_days - rec.last_decay_at))
            if score > 0.0:
                out[pid] = {"score": score, "signals": list(rec.signals)}
        return out

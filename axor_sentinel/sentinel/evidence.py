"""Typed evidence sets — the deterministic substrate replacing score arithmetic.

Instead of folding accesses into a calibrated scalar (accumulate → decay →
FLAG_THRESHOLD), each resource carries a windowed SET of typed facts. A fact is
either inside the sliding window or expired — exact semantics, no half-life
fuzz. Verdicts are decidable predicates over these sets (see predicates.py);
nothing here is tuned to data.

Anti-poisoning is set semantics instead of multipliers: predicates count
DISTINCT origins / sessions, so N repeats from one actor collapse to one
element — the exact-arithmetic-free version of source_diversity_factor and
origin_dampening. The origin key is SessionSummary.mitigation_origin (the
authenticated source_class when attested, else agent_id), same as the F1 fix.

The cycle's taint gate is preserved: evidence is collected from tainted
sessions only, so the dual-run comparison against the score path stays
apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from axor_sentinel.graph.model import SignalType

# canonical_confidence is a three-valued resolution label encoded as a float
# (normalizer tiers: provider_id 1.0 / path 0.7 / heuristic 0.4). Recover the
# label deterministically; predicates treat "heuristic" as a level ceiling
# instead of multiplying weight by 0.4.
_PATH_TIER: float = 0.7


def resolution_from_confidence(canonical_confidence: float) -> str:
    """Map the normalizer's confidence tier back to its resolution label."""
    if canonical_confidence >= 1.0:
        return "provider_id"
    if canonical_confidence >= _PATH_TIER:
        return "path"
    return "heuristic"


@dataclass(frozen=True)
class Evidence:
    """One deterministic fact about a resource, contributed by one session.

    rank is the graded involvement depth (SignalType — an ORDER, not a weight);
    tainted is core's session taint fact; resolution is the resource-id
    normalization tier ("heuristic" caps the reachable level, see predicates).
    """
    origin: str          # mitigation_origin: source_class | agent_id (F1 keying)
    session_id: str
    rank: SignalType
    tainted: bool
    observed_at: float
    resolution: str      # "provider_id" | "path" | "heuristic"

    def to_json(self) -> dict:
        return {
            "origin": self.origin,
            "session_id": self.session_id,
            "rank": self.rank.value,
            "tainted": self.tainted,
            "observed_at": self.observed_at,
            "resolution": self.resolution,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Evidence":
        return cls(
            origin=str(obj["origin"]),
            session_id=str(obj["session_id"]),
            rank=SignalType(obj["rank"]),
            tainted=bool(obj["tainted"]),
            observed_at=float(obj["observed_at"]),
            resolution=str(obj.get("resolution", "path")),
        )


class EvidenceStore:
    """Windowed per-resource evidence sets.

    Deduplicates by (session_id, rank) per resource — a session contributes one
    fact per involvement depth, so replaying the same record (e.g. a merged
    probe-bridge duplicate) cannot inflate counts. prune() drops facts older
    than the window: monotone inside the window, exact expiry at its edge.
    """

    def __init__(self) -> None:
        self._by_resource: dict[str, dict[tuple[str, str], Evidence]] = {}

    def add(self, resource_id: str, evidence: Evidence) -> None:
        bucket = self._by_resource.setdefault(resource_id, {})
        key = (evidence.session_id, evidence.rank.value)
        # First occurrence wins — timestamps of a duplicate replay are ignored.
        bucket.setdefault(key, evidence)

    def evidence_for(self, resource_id: str) -> tuple[Evidence, ...]:
        return tuple(self._by_resource.get(resource_id, {}).values())

    def resource_ids(self) -> tuple[str, ...]:
        return tuple(self._by_resource)

    def prune(self, now: float, window_days: float) -> None:
        """Drop evidence outside the sliding window (exact TTL, replaces decay)."""
        horizon = now - window_days * 86400.0
        for rid in list(self._by_resource):
            kept = {
                k: e for k, e in self._by_resource[rid].items()
                if e.observed_at >= horizon
            }
            if kept:
                self._by_resource[rid] = kept
            else:
                del self._by_resource[rid]

    # ── persistence (rides inside the signed sentinel_state envelope) ─────────

    def to_json(self) -> dict[str, list[dict]]:
        return {
            rid: [e.to_json() for e in bucket.values()]
            for rid, bucket in self._by_resource.items()
        }

    @classmethod
    def from_json(cls, obj: dict) -> "EvidenceStore":
        store = cls()
        try:
            for rid, items in obj.items():
                for item in items:
                    store.add(str(rid), Evidence.from_json(item))
        except Exception:
            # Corrupt state is recoverable — cold-start semantics, same policy
            # as the rest of load_state.
            return cls()
        return store


def evidence_from_session(
    origin: str,
    session_id: str,
    started_at: float,
    tainted: bool,
    accesses: Iterable,  # Iterable[cycle.ResourceAccess] — no import cycle
) -> list[tuple[str, Evidence]]:
    """Map one session's accesses to (resource_id, Evidence) pairs."""
    out: list[tuple[str, Evidence]] = []
    for access in accesses:
        out.append((
            access.resource_id,
            Evidence(
                origin=origin,
                session_id=session_id,
                rank=access.signal_type,
                tainted=tainted,
                observed_at=started_at,
                resolution=resolution_from_confidence(access.canonical_confidence),
            ),
        ))
    return out

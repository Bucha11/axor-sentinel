from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from axor_sentinel.graph.derive import derive_container_id, derive_resource_info
from axor_sentinel.graph.normalizer import normalize_resource_id
from axor_sentinel.sentinel.snapshot import ReputationSnapshot, load_snapshot

if TYPE_CHECKING:
    from axor_core.contracts.anomaly import NormalizedIntent  # type: ignore[import-untyped]
    from axor_core.contracts.intent import Intent  # type: ignore[import-untyped]

log = logging.getLogger("axor.sentinel.enricher")

# Smallest reputation a positively-suspicious resource maps to. Core treats a 0.0
# reading as "unknown" (never crosses the detection floor); a maximally-suspicious
# resource (suspicion 1.0) must still produce a positive, floor-crossing reading
# rather than collapse to that "unknown" — so the conversion clamps above 0.
_MIN_CROSSING_REP: float = 1e-3


def _suspicion_to_reputation(suspicion: float) -> float:
    """Convert sentinel's *suspicion* score (high = bad) to the value core's
    ``target_resource_reputation`` field expects.

    Core's degradation floor is a TRUST reading: ``record_detection`` tightens when
    ``0.0 < reputation <= detection_floor`` and treats ``0.0`` as "unknown" (never
    crosses). Sentinel scores the opposite polarity (``suspicion_score``, high = bad),
    so handing the suspicion through unconverted would tighten core on the *trusted*
    resources and ignore the suspicious ones. We invert: ``reputation = 1 - suspicion``.

    A 0.0 suspicion maps to 0.0 (core "unknown", no crossing); any positive suspicion
    maps to ``1 - suspicion`` clamped above 0 so suspicion 1.0 still crosses. Operator
    wiring: set ``detection_floor = 1 - FLAG_THRESHOLD`` (default 0.3) so a
    sentinel-flagged resource (suspicion >= FLAG_THRESHOLD) crosses and tightens.
    """
    if suspicion <= 0.0:
        return 0.0
    return max(_MIN_CROSSING_REP, 1.0 - suspicion)


class SnapshotIntentEnricher:
    """
    Implements axor-core's ReputationEnricher protocol.

    Reads the current reputation snapshot (pre-loaded — no Neo4j on hot path, A-6)
    and populates target_resource_reputation and target_container_reputation on
    NormalizedIntent via dataclasses.replace().

    Polarity: the snapshot holds sentinel's *suspicion* score (high = bad); core's
    reputation field is *trust* (a positive reading <= detection_floor crosses and
    tightens degradation, 0.0 = unknown). The fields are converted at this boundary
    by _suspicion_to_reputation, so core tightens on suspicious resources, not benign
    ones. Reputation is observe-only in core: it never denies, only feeds the opt-in
    degradation floor.

    Unknown resources return the original NormalizedIntent unchanged (score stays 0.0).
    Never raises — failures are logged and original intent returned (fail-safe).

    Usage:
        enricher = SnapshotIntentEnricher.from_dir(Path("~/.axor/sentinel/snapshots"))
        # or with a pre-loaded snapshot:
        enricher = SnapshotIntentEnricher(snapshot)
    """

    def __init__(self, snapshot: ReputationSnapshot | None = None) -> None:
        self._snapshot = snapshot

    @classmethod
    def from_dir(cls, snapshot_dir: Path) -> "SnapshotIntentEnricher":
        """Load snapshot from directory and return an enricher instance."""
        snapshot = load_snapshot(Path(snapshot_dir))
        return cls(snapshot)

    def reload(self, snapshot_dir: Path) -> None:
        """Reload the snapshot from disk. Call after each audit cycle."""
        self._snapshot = load_snapshot(Path(snapshot_dir))

    def enrich(
        self,
        normalized: "NormalizedIntent",
        intent: "Intent",
    ) -> "NormalizedIntent":
        """
        Return normalized with reputation fields populated from snapshot.

        Derives resource_id from intent args using graph/normalizer.py.
        Falls back to original NormalizedIntent if resource is unknown or
        if snapshot is not loaded.

        Must not query Neo4j — reads pre-loaded snapshot only (invariant A-6).
        Must not raise (invariant: fail-safe on hot path).
        """
        if self._snapshot is None:
            return normalized

        try:
            resource_id, container_id = self._derive_ids(intent)
            # The snapshot stores SUSPICION (high = bad); core's reputation field is
            # TRUST (low-positive crosses the floor). Convert at this boundary.
            resource_susp = self._snapshot.resource_reputation.get(resource_id, 0.0)
            container_susp = self._snapshot.container_reputation.get(container_id, 0.0)

            if resource_susp == 0.0 and container_susp == 0.0:
                return normalized

            return dataclasses.replace(
                normalized,
                target_resource_reputation=_suspicion_to_reputation(resource_susp),
                target_container_reputation=_suspicion_to_reputation(container_susp),
            )
        except Exception as exc:
            log.debug("enricher failed (returning original): %s", exc)
            return normalized

    def _derive_ids(self, intent: "Intent") -> tuple[str, str]:
        """
        Derive (resource_id, container_id) from intent args.

        Resource ID derived via graph/normalizer.py priority order.
        Container ID derived from service + directory extracted from args.
        """
        args = intent.payload.get("args", {})
        tool = intent.payload.get("tool", "")

        resource_info = derive_resource_info(tool, args)
        resource_id, _, _ = normalize_resource_id(resource_info)

        # Container ID: service + directory of the resource
        container_id = derive_container_id(
            resource_info.get("path", ""),
            resource_info.get("service", ""),
        )
        return resource_id, container_id

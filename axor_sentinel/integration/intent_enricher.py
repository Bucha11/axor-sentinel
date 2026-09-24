from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from axor_sentinel.graph.derive import derive_identity
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
    maps to ``1 - suspicion`` clamped above 0 so suspicion 1.0 still crosses.

    The snapshot codomain is FINITE (LEVEL_SUSPICION: 0.0 clean / 0.4 watch /
    1.0 flagged), so the conversion emits exactly {0.0, 0.6, 0.001}. Operator
    wiring: ``detection_floor`` in (0.001, 0.6) tightens on FLAGGED only — the
    long-standing 0.3 default keeps its exact meaning; a floor >= 0.6 also
    tightens on WATCH. Decidable end-to-end: fact → predicate → level → finite
    value → floor comparison, no calibrated threshold anywhere on the path.
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
    def from_dir(cls, snapshot_dir: Path) -> SnapshotIntentEnricher:
        """Load snapshot from directory and return an enricher instance."""
        snapshot = load_snapshot(Path(snapshot_dir))
        return cls(snapshot)

    def reload(self, snapshot_dir: Path) -> None:
        """Reload the snapshot from disk. Call after each audit cycle.

        The held snapshot is only ever REPLACED by a newer one:

        - load fails (checksum / signature / parse / level-binding failure, a
          missing or dangling link) → keep the previous snapshot and warn.
          load_snapshot returns None for all of these, and assigning that None
          switched reputation off for the whole node until the next good cycle —
          a corrupted or tampered file is exactly when it must stay on.
        - loaded version < held version → a ROLLBACK (e.g. snapshot_current
          re-linked to an older, validly-signed snapshot_vN.json); keep the
          previous snapshot and warn. load_snapshot has no memory and cannot see
          this — this is the stateful reader, so the check lives here.
        - loaded version == held version → no-op; the cycle never publishes two
          snapshots under one version, so there is nothing newer to take.
        """
        loaded = load_snapshot(Path(snapshot_dir))
        held = self._snapshot
        if loaded is None:
            if held is not None:
                log.warning(
                    "enricher: snapshot reload from %s failed — keeping version %d",
                    snapshot_dir, held.version,
                )
            return
        if held is not None and loaded.version < held.version:
            log.warning(
                "enricher: refusing snapshot version %d from %s — lower than the "
                "held version %d (rollback); keeping version %d",
                loaded.version, snapshot_dir, held.version, held.version,
            )
            return
        if held is not None and loaded.version == held.version:
            return
        self._snapshot = loaded

    def enrich(
        self,
        normalized: NormalizedIntent,
        intent: Intent,
    ) -> NormalizedIntent:
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
            ids = self._derive_ids(intent)
            if ids is None:
                # No resource named (bash, send_email …): nothing to look up. Never
                # look up "" — that would be one reputation for every such call.
                return normalized
            resource_id, container_id = ids
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

    def _derive_ids(self, intent: Intent) -> tuple[str, str] | None:
        """
        Derive (resource_id, container_id) from intent args, or ``None`` when the
        call names no resource.

        Uses ``graph.derive.derive_identity`` — the same function CoreSessionSink
        uses on the audit path — so a resource the cycle scored is found here under
        the identical id. The container is derived from the NORMALISED resource
        locator, never from the raw path.
        """
        args = intent.payload.get("args", {})
        tool = intent.payload.get("tool", "")

        identity = derive_identity(tool, args)
        if identity is None:
            return None
        return identity.resource_id, identity.container_id

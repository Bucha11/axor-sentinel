"""Branch attestation — heat reset done right (UI spec 8.1.1).

A reset implemented as deletion or zeroing would be an operator-side
reputation-laundering channel, so reset does not exist. An attestation is an
append-only event: who, when, reason (required), scope (causal_root branch),
prior heat. Reputation is *recomputed downward over* the attestation; revoking
is itself a new event — full history, both directions.

Score semantics under the logarithmic accumulator (weight.accumulate):
scores compose as ``s = 1 - (1-a)(1-b)``. An unrevoked attestation with
``prior_heat = a`` discharges exactly the evidence it vouched for, so the
effective score is the *post-attestation residue*::

    effective = 1 - (1 - raw) / (1 - prior_heat)

At the moment of attestation (raw == prior_heat) the branch reads 0.0; signals
recorded afterwards re-heat it from that baseline — "I checked, resume
watching", never "trust this forever". Values' integrity taint is untouched:
attestation lowers reputation, endorsement of values stays the kernel's
bounded-codomain mechanism only.

Level semantics (the deterministic codomain the cycle exports): an attestation
vouches for the evidence that existed WHEN it was made, nothing later. It
descends the verdict one level only while no evidence fact for the resource is
newer than ``created_at`` (``is_superseded``); once a newer fact arrives the
attestation stops applying and the full verdict is exported — without this a
single attestation discounted every future cycle, so a fresh export-denied
flag read WATCH (0.4, under core's 0.3 floor) forever. The record itself is
never dropped: it stays in history and the cycle names it as superseded.

Authentication is out of scope here: ``operator`` and ``org`` are
self-declared strings. Proving that the caller really is that operator from
that keyset is the job of whatever front-end calls ``SentinelCycle.attest``;
this module only enforces the structural rules (reason required, same-keyset
revocation) over whatever identity it is handed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

# Append an attestation event node and link it to the attested branch.
# Append-only: MERGE is deliberately NOT used for the event node — a duplicate
# attestation_id is a caller bug and should violate the unique constraint.
ATTEST_BRANCH_QUERY = """
MATCH (r:Resource {id: $resource_id})
CREATE (a:Attestation {
    attestation_id: $attestation_id,
    operator: $operator,
    org: $org,
    reason: $reason,
    causal_root: $causal_root,
    prior_heat: r.suspicion_score,
    revokes: $revokes,
    created_at: timestamp()
})
CREATE (a)-[:ATTESTS]->(r)
RETURN a.prior_heat AS prior_heat
"""

# All attestation events for a branch, newest first (history, both directions).
BRANCH_ATTESTATIONS_QUERY = """
MATCH (a:Attestation {causal_root: $causal_root})-[:ATTESTS]->(r:Resource)
RETURN a.attestation_id AS attestation_id, a.operator AS operator,
       a.org AS org, a.reason AS reason, a.prior_heat AS prior_heat,
       a.revokes AS revokes, a.created_at AS created_at
ORDER BY a.created_at DESC
"""


class AttestationError(ValueError):
    """Refused attestation (missing reason — decision 8 makes it required — or
    a missing operator / resource_id, or an out-of-range prior_heat)."""


@dataclass(frozen=True)
class AttestationRecord:
    attestation_id: str
    operator: str
    reason: str
    causal_root: str
    prior_heat: float
    revokes: str | None = None
    # The Sentinel resource whose branch this attestation covers. The cycle keys
    # attestations by it; the taint-graph scope is `causal_root` (spec 8.1.1).
    resource_id: str = ""
    # Operator keyset / organisation. A revocation only takes effect when its org
    # matches the org of the attestation it revokes (see effective_revocations):
    # revoking an attestation RAISES the branch score back toward raw, so a
    # cross-org operator honouring their own revocation would be a griefing /
    # laundering-reversal channel. Empty org = unspecified: with no keyset model
    # configured on EITHER side the check is a no-op (legacy behaviour), so
    # single-org deployments are unaffected; an empty org never matches a
    # non-empty one (see _same_keyset). Self-declared — see module docstring.
    org: str = ""
    # When the attestation was made, seconds since the epoch on the cycle's
    # clock (same clock and unit as Evidence.observed_at, which is what it is
    # compared against). 0.0 = unset; SentinelCycle.attest stamps it. NOTE:
    # the Neo4j ATTEST_BRANCH_QUERY's created_at is timestamp() milliseconds —
    # divide by 1000 when building a record from BRANCH_ATTESTATIONS_QUERY.
    created_at: float = 0.0

    def to_json(self) -> dict:
        return {
            "attestation_id": self.attestation_id,
            "operator": self.operator,
            "reason": self.reason,
            "causal_root": self.causal_root,
            "prior_heat": self.prior_heat,
            "revokes": self.revokes,
            "resource_id": self.resource_id,
            "org": self.org,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, obj: dict) -> AttestationRecord:
        revokes = obj.get("revokes")
        return cls(
            attestation_id=str(obj["attestation_id"]),
            operator=str(obj["operator"]),
            reason=str(obj["reason"]),
            causal_root=str(obj.get("causal_root", "")),
            prior_heat=float(obj["prior_heat"]),
            revokes=None if revokes is None else str(revokes),
            resource_id=str(obj["resource_id"]),
            org=str(obj.get("org", "")),
            created_at=float(obj.get("created_at", 0.0)),
        )


def validate(record: AttestationRecord) -> None:
    """Structural gate for an attestation (or revocation) record.

    Whitespace-only operator / resource_id are refused like a whitespace-only
    reason: a blank identity is no identity, and a record with no resource_id
    was silently keyed under "" by the cycle and never covered anything.
    prior_heat must be a finite score in [0, 1] — it enters the log-space
    discharge in effective_score, where NaN or an out-of-range value yields
    nonsense. This does NOT authenticate the operator (module docstring).
    """
    if not record.reason.strip():
        raise AttestationError("attestation requires a reason (decision 8)")
    if not record.operator.strip():
        raise AttestationError("attestation requires an operator identity")
    if not record.resource_id.strip():
        raise AttestationError("attestation requires a resource_id")
    heat = record.prior_heat
    if not isinstance(heat, (int, float)) or not math.isfinite(heat) or not 0.0 <= heat <= 1.0:
        raise AttestationError(
            f"attestation prior_heat must be a finite score in [0, 1], got {heat!r}"
        )


def _same_keyset(revoker: AttestationRecord, target: AttestationRecord) -> bool:
    """A revocation is authorised only from the attesting keyset. Orgs must be
    equal; the legacy no-keyset case (BOTH empty) is simply that equality, so
    single-org deployments keep working. An empty org never matches a set one:
    treating "unspecified" as a wildcard let any operator lift an org's
    attestation just by leaving ``org`` blank — the check was bypassable by
    omission. (Orgs are self-declared; authenticating them is the caller's
    job, see the module docstring.)"""
    return revoker.org == target.org


def effective_revocations(records: list[AttestationRecord]) -> set[str]:
    """attestation_ids that are validly revoked — a revocation whose org matches
    its target's. Cross-org revocations stay in history (append-only, nothing is
    deleted) but do not change coverage."""
    by_id = {r.attestation_id: r for r in records}
    revoked: set[str] = set()
    for r in records:
        if r.revokes is None:
            continue
        target = by_id.get(r.revokes)
        if target is not None and _same_keyset(r, target):
            revoked.add(r.revokes)
    return revoked


def active_attestation(records: list[AttestationRecord]) -> AttestationRecord | None:
    """The newest unrevoked attestation (never a revocation record), or None.

    ``records`` newest-first (as BRANCH_ATTESTATIONS_QUERY returns them).
    Revocations are attestation events whose ``revokes`` names an earlier
    attestation_id — nothing is deleted, coverage just changes — and only a
    same-keyset revocation is honoured (effective_revocations). Callers that
    name the attestation in facts use THIS record's id: ``records[0]`` may
    well be the revocation that just lifted a different one.
    """
    revoked = effective_revocations(records)
    for record in records:
        if record.revokes is None and record.attestation_id not in revoked:
            return record
    return None


def active_prior_heat(records: list[AttestationRecord]) -> float | None:
    """The prior_heat of the newest unrevoked attestation, or None."""
    record = active_attestation(records)
    return None if record is None else record.prior_heat


def is_superseded(record: AttestationRecord, observed_at: Iterable[float]) -> bool:
    """True when any evidence timestamp is strictly newer than the attestation.

    The operator vouched for what existed at ``created_at``; a fact observed
    afterwards is evidence the attestation never saw, so it stops discounting
    (the full verdict stands). Strictly newer: a fact stamped at the same
    instant was, by construction, visible to the operator. A record with no
    timestamp (created_at <= 0, never stamped) is treated as older than any
    evidence — an undated attestation must not outlive new evidence.

    The cycle passes Evidence.known_at — max(session start, ingest time) — not
    the bare session start: a session that STARTED before the attestation but
    was only reported (ingested) after it is still evidence the operator never
    saw, and comparing by fact time alone let such a late report leave the
    discount in place. Evidence persisted before ingest times existed has
    known_at == observed_at, i.e. the old comparison.
    """
    return any(t > record.created_at for t in observed_at)


def _clamp_heat(value: float) -> float:
    """prior_heat as a usable score: NaN / inf / non-numbers -> 0.0 (discharge
    nothing — the conservative reading of a corrupt record), else clamped to
    [0, 1]. Records that went through validate() are already in range; this
    guards ones built directly or restored from an older store."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return min(1.0, max(0.0, v))


def effective_score(raw_score: float, records: list[AttestationRecord]) -> float:
    """Recompute the branch score over its attestation history.

    No unrevoked attestation -> the raw score stands. Otherwise the attested
    portion is discharged in log-space (see module docstring); a branch whose
    values re-trigger denials heats right back up from the attested baseline.

    ``prior_heat`` is clamped by _clamp_heat first: a NaN prior used to
    propagate NaN, a negative one inflated the score above raw, and one above
    1 flipped the sign of the residue. A non-finite prior now discharges
    nothing (raw stands); out-of-range values are clamped to [0, 1].
    """
    record = active_attestation(records)
    if record is None:
        return raw_score
    prior = _clamp_heat(record.prior_heat)
    if prior >= 1.0:
        return 0.0 if raw_score <= prior else raw_score
    residue = 1.0 - (1.0 - raw_score) / (1.0 - prior)
    return min(1.0, max(0.0, residue))

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
"""
from __future__ import annotations

from dataclasses import dataclass

# Append an attestation event node and link it to the attested branch.
# Append-only: MERGE is deliberately NOT used for the event node — a duplicate
# attestation_id is a caller bug and should violate the unique constraint.
ATTEST_BRANCH_QUERY = """
MATCH (r:Resource {id: $resource_id})
CREATE (a:Attestation {
    attestation_id: $attestation_id,
    operator: $operator,
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
       a.reason AS reason, a.prior_heat AS prior_heat,
       a.revokes AS revokes, a.created_at AS created_at
ORDER BY a.created_at DESC
"""


class AttestationError(ValueError):
    """Refused attestation (missing reason — decision 8 makes it required)."""


@dataclass(frozen=True)
class AttestationRecord:
    attestation_id: str
    operator: str
    reason: str
    causal_root: str
    prior_heat: float
    revokes: str | None = None


def validate(record: AttestationRecord) -> None:
    if not record.reason.strip():
        raise AttestationError("attestation requires a reason (decision 8)")
    if not record.operator:
        raise AttestationError("attestation requires an operator identity")


def active_prior_heat(records: list[AttestationRecord]) -> float | None:
    """The prior_heat of the newest unrevoked attestation, or None.

    ``records`` newest-first (as BRANCH_ATTESTATIONS_QUERY returns them).
    Revocations are attestation events whose ``revokes`` names an earlier
    attestation_id — nothing is deleted, coverage just changes.
    """
    revoked = {r.revokes for r in records if r.revokes is not None}
    for record in records:
        if record.revokes is None and record.attestation_id not in revoked:
            return record.prior_heat
    return None


def effective_score(raw_score: float, records: list[AttestationRecord]) -> float:
    """Recompute the branch score over its attestation history.

    No unrevoked attestation -> the raw score stands. Otherwise the attested
    portion is discharged in log-space (see module docstring); a branch whose
    values re-trigger denials heats right back up from the attested baseline.
    """
    prior = active_prior_heat(records)
    if prior is None:
        return raw_score
    if prior >= 1.0:
        return 0.0 if raw_score <= prior else raw_score
    residue = 1.0 - (1.0 - raw_score) / (1.0 - prior)
    return min(1.0, max(0.0, residue))

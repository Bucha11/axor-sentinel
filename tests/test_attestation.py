"""Branch attestation: append-only coverage, downward recompute, re-heating."""
from __future__ import annotations

import pytest

from axor_sentinel.sentinel.attestation import (
    AttestationError,
    AttestationRecord,
    active_attestation,
    active_prior_heat,
    effective_revocations,
    effective_score,
    is_superseded,
    validate,
)
from axor_sentinel.sentinel.weight import accumulate


def _rec(
    aid: str, prior: float, revokes: str | None = None, org: str = ""
) -> AttestationRecord:
    return AttestationRecord(
        attestation_id=aid, operator="op_d", reason="investigated",
        causal_root="root_1", prior_heat=prior, revokes=revokes, org=org,
    )


def test_reason_is_required() -> None:
    bad = AttestationRecord(
        attestation_id="a1", operator="op", reason="  ",
        causal_root="r", prior_heat=0.8,
    )
    with pytest.raises(AttestationError):
        validate(bad)


def test_no_attestation_score_stands() -> None:
    assert effective_score(0.86, []) == 0.86


def test_attestation_zeroes_branch_at_attestation_time() -> None:
    assert effective_score(0.86, [_rec("a1", 0.86)]) == pytest.approx(0.0)


def test_post_attestation_signals_reheat_from_baseline() -> None:
    # heat 0.6 attested; then a new 0.5-weight signal accumulates on the raw
    # score. The effective score must equal what that signal ALONE contributes.
    prior = 0.6
    raw_after = accumulate(prior, 0.5)
    assert effective_score(raw_after, [_rec("a1", prior)]) == pytest.approx(0.5)


def test_revocation_restores_raw_score() -> None:
    records = [  # newest first
        _rec("a2", 0.0, revokes="a1"),
        _rec("a1", 0.86),
    ]
    assert active_prior_heat(records) is None
    assert effective_score(0.86, records) == 0.86


def test_newest_unrevoked_attestation_wins() -> None:
    records = [  # newest first: a2 covers more heat than a1
        _rec("a2", 0.9),
        _rec("a1", 0.5),
    ]
    assert active_prior_heat(records) == 0.9
    assert effective_score(0.9, records) == pytest.approx(0.0)


def test_bounds_are_clamped() -> None:
    assert effective_score(0.2, [_rec("a1", 0.8)]) == 0.0  # decayed below prior
    assert effective_score(1.0, [_rec("a1", 1.0)]) == 0.0


def test_same_org_revocation_is_honoured() -> None:
    records = [  # newest first
        _rec("a2", 0.0, revokes="a1", org="acme"),
        _rec("a1", 0.86, org="acme"),
    ]
    assert effective_revocations(records) == {"a1"}
    assert active_prior_heat(records) is None
    assert effective_score(0.86, records) == 0.86  # revoked → raw stands


def test_cross_org_revocation_is_ignored() -> None:
    # A rogue operator from another keyset cannot lift acme's attestation and
    # re-heat the branch — the revocation stays in history but does not count.
    records = [
        _rec("a2", 0.0, revokes="a1", org="rogue"),
        _rec("a1", 0.86, org="acme"),
    ]
    assert effective_revocations(records) == set()
    assert active_prior_heat(records) == 0.86  # attestation still covers
    assert effective_score(0.86, records) == pytest.approx(0.0)


def test_unspecified_org_keeps_legacy_behaviour() -> None:
    # No keyset model configured (empty orgs) → the guard is a no-op.
    records = [_rec("a2", 0.0, revokes="a1"), _rec("a1", 0.86)]
    assert effective_revocations(records) == {"a1"}
    assert effective_score(0.86, records) == 0.86


# ── Supersession, keyset, validation, prior_heat bounds ─────────────────────

def _full(**kw) -> AttestationRecord:
    base = dict(
        attestation_id="a1", operator="op", reason="checked",
        causal_root="r", prior_heat=0.5, resource_id="res_1",
    )
    base.update(kw)
    return AttestationRecord(**base)


def test_is_superseded_only_by_strictly_newer_evidence() -> None:
    rec = _full(created_at=100.0)
    assert not is_superseded(rec, [])
    assert not is_superseded(rec, [50.0, 100.0])
    assert is_superseded(rec, [50.0, 100.5])
    # An undated (never stamped) record lapses under any evidence.
    assert is_superseded(_full(created_at=0.0), [1.0])


def test_active_attestation_skips_revocation_records() -> None:
    records = [  # newest first: a3 revokes a2 → a1 is the one that applies
        _rec("a3", 0.0, revokes="a2"),
        _rec("a2", 0.9),
        _rec("a1", 0.5),
    ]
    active = active_attestation(records)
    assert active is not None and active.attestation_id == "a1"
    assert active_attestation([_rec("a2", 0.0, revokes="a1"), _rec("a1", 0.5)]) is None


@pytest.mark.parametrize("revoker_org,target_org", [("", "acme"), ("acme", "")])
def test_empty_org_does_not_match_a_set_org(revoker_org: str, target_org: str) -> None:
    # Leaving org blank used to be a wildcard: any operator could lift acme's
    # attestation by omission.
    records = [
        _rec("a2", 0.0, revokes="a1", org=revoker_org),
        _rec("a1", 0.86, org=target_org),
    ]
    assert effective_revocations(records) == set()
    assert active_prior_heat(records) == 0.86


@pytest.mark.parametrize("field,value", [
    ("operator", ""), ("operator", "   "),
    ("resource_id", ""), ("resource_id", " \t"),
    ("reason", "\n"),
    ("prior_heat", float("nan")), ("prior_heat", float("inf")),
    ("prior_heat", -0.1), ("prior_heat", 1.5),
])
def test_validate_rejects_blank_identity_and_bad_heat(field: str, value) -> None:
    with pytest.raises(AttestationError):
        validate(_full(**{field: value}))


def test_validate_accepts_boundary_heat() -> None:
    validate(_full(prior_heat=0.0))
    validate(_full(prior_heat=1.0))


def test_effective_score_clamps_prior_heat() -> None:
    # NaN / inf discharge nothing (raw stands) instead of propagating NaN.
    assert effective_score(0.7, [_rec("a1", float("nan"))]) == 0.7
    assert effective_score(0.7, [_rec("a1", float("inf"))]) == 0.7
    # Negative used to inflate above raw; clamped to 0 → raw stands.
    assert effective_score(0.7, [_rec("a1", -0.5)]) == pytest.approx(0.7)
    # Above 1 used to flip the residue's sign; clamped to 1.
    assert effective_score(0.7, [_rec("a1", 1.5)]) == 0.0
    assert effective_score(1.0, [_rec("a1", 1.5)]) == 0.0


def test_record_json_round_trip() -> None:
    rec = _full(revokes="a0", org="acme", created_at=123.5)
    assert AttestationRecord.from_json(rec.to_json()) == rec

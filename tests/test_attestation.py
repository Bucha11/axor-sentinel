"""Branch attestation: append-only coverage, downward recompute, re-heating."""
from __future__ import annotations

import pytest

from axor_sentinel.sentinel.attestation import (
    AttestationError,
    AttestationRecord,
    active_prior_heat,
    effective_score,
    validate,
)
from axor_sentinel.sentinel.weight import accumulate


def _rec(aid: str, prior: float, revokes: str | None = None) -> AttestationRecord:
    return AttestationRecord(
        attestation_id=aid, operator="op_d", reason="investigated",
        causal_root="root_1", prior_heat=prior, revokes=revokes,
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

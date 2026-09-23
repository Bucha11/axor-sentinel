"""The snapshot over a wire, not a filesystem.

`atomic_swap` / `load_snapshot` hand a snapshot to a reader on the SAME host —
the enricher on the governance hot path, reading a symlink the cycle swapped. A
control plane is not on that host: it renders the reputation a node's sentinel
computed, so the snapshot travels, and what arrives was computed somewhere else
entirely.

That changes what the checksum is for. Between two processes on one host it
catches a lost bit. Over a wire, a payload whose checksum does not match its own
maps was rewritten by somebody, and the plane has no other way to tell.
"""
from __future__ import annotations

import pytest

from axor_sentinel.sentinel.predicates import LEVEL_SUSPICION, ReputationLevel
from axor_sentinel.sentinel.snapshot import (
    ReputationSnapshot,
    SnapshotRejected,
    snapshot_from_payload,
    snapshot_payload,
)


def _snapshot(**over: object) -> ReputationSnapshot:
    base = dict(
        version=7,
        generated_at=1_700_000_000.0,
        resource_reputation={"db:customers": 1.0, "s3:exports": 0.4},
        container_reputation={"svc:billing": 0.4},
        resource_level={"db:customers": "FLAGGED", "s3:exports": "WATCH"},
        container_level={"svc:billing": "WATCH"},
        verdict_facts={"db:customers": ["P3 staging count", "P4 staged-then-export"]},
    )
    base.update(over)
    return ReputationSnapshot(**base).with_checksum()  # type: ignore[arg-type]


def test_a_snapshot_survives_the_round_trip() -> None:
    snapshot = _snapshot()
    assert snapshot_from_payload(snapshot_payload(snapshot)) == snapshot


def test_the_verdict_facts_travel() -> None:
    """The reputation number alone is an accusation. What a consumer renders
    next to a FLAGGED resource is why it is flagged."""
    arrived = snapshot_from_payload(snapshot_payload(_snapshot()))
    assert arrived.verdict_facts["db:customers"] == [
        "P3 staging count", "P4 staged-then-export",
    ]


def test_a_rewritten_map_is_refused() -> None:
    """The one check that matters over a wire: the maps and the checksum arrive
    together from a party the reader is not. Clearing a FLAGGED resource is the
    edit worth making, and it is the edit the checksum catches."""
    payload = snapshot_payload(_snapshot())
    payload["resource_reputation"]["db:customers"] = 0.0
    with pytest.raises(SnapshotRejected, match="checksum"):
        snapshot_from_payload(payload)


def test_a_json_round_trip_does_not_break_the_checksum() -> None:
    """The checksum covers a serialisation in which 1.0 is written "1.0", and a
    JSON round-trip does not preserve that: JSON.parse("1.0") is the number 1,
    JSON.stringify writes "1", and Python then parses an int. The maps are
    numerically identical and the naive comparison fails.

    Verifying against the sender's spelling would make this wire
    Python-to-Python only, and would reject a correct snapshot for the crime of
    passing through a proxy that reformatted its JSON.
    """
    snapshot = _snapshot()
    payload = snapshot_payload(snapshot)
    # exactly what a JS (or any int-collapsing) JSON writer emits
    payload["resource_reputation"] = {
        k: int(v) if float(v).is_integer() else v
        for k, v in payload["resource_reputation"].items()
    }
    assert payload["resource_reputation"]["db:customers"] == 1  # an int, not 1.0

    arrived = snapshot_from_payload(payload)
    assert arrived == snapshot
    assert arrived.resource_reputation["db:customers"] == 1.0


def test_the_round_trip_tolerance_does_not_soften_the_tamper_check() -> None:
    """The reason to be careful: the coercion normalises SPELLING, never value.
    A cleared FLAGGED resource is still a different number and still refused."""
    payload = snapshot_payload(_snapshot())
    payload["resource_reputation"]["db:customers"] = 0  # cleared, int-spelled
    with pytest.raises(SnapshotRejected, match="checksum"):
        snapshot_from_payload(payload)


def test_a_calibrated_float_is_not_a_verdict() -> None:
    """The codomain is finite by construction so core's detection_floor
    comparison is decidable. A consumer that did not compute these numbers has
    to be able to hold that line, or an arbitrary float reintroduces the
    calibrated threshold the deterministic verdict layer exists to remove."""
    payload = snapshot_payload(_snapshot(resource_reputation={"r": 0.73}))
    with pytest.raises(SnapshotRejected, match="finite codomain"):
        snapshot_from_payload(payload)


def test_the_accepted_values_are_the_ones_sentinel_emits() -> None:
    """Not a literal: whatever LEVEL_SUSPICION says, on both sides."""
    for level, suspicion in LEVEL_SUSPICION.items():
        payload = snapshot_payload(
            _snapshot(resource_reputation={"r": suspicion},
                      resource_level={"r": level.name})
        )
        assert snapshot_from_payload(payload).resource_reputation["r"] == suspicion


def test_a_level_this_library_does_not_know_is_refused() -> None:
    payload = snapshot_payload(_snapshot(resource_level={"r": "SPICY"}))
    with pytest.raises(SnapshotRejected, match="CLEAN"):
        snapshot_from_payload(payload)
    assert {level.name for level in ReputationLevel} == {"CLEAN", "WATCH", "FLAGGED"}


@pytest.mark.parametrize("bad", [
    "not-an-object",
    None,
    {"generated_at": 1.0},                       # no version
    {"version": 1},                              # no generated_at
    {"version": True, "generated_at": 1.0},      # a bool is not a version
])
def test_a_payload_that_is_not_a_snapshot_is_refused(bad: object) -> None:
    with pytest.raises(SnapshotRejected):
        snapshot_from_payload(bad)


def test_a_field_from_a_newer_sentinel_is_dropped_not_fatal() -> None:
    """Forward-compatible in the same direction the on-disk loader is: a newer
    field cannot alter the maps the checksum covers, so it is not a reason to
    refuse a node's whole report."""
    payload = snapshot_payload(_snapshot())
    payload["fanout_quota_breaches"] = {"svc:billing": 4}
    assert snapshot_from_payload(payload) == _snapshot()


def test_an_unsigned_snapshot_is_accepted() -> None:
    """The HMAC signature is keyed to the node's own AXOR_SNAPSHOT_KEY, which a
    plane does not hold and must not. Requiring one here would mean handing the
    reputation key to the party the reputation is reported TO."""
    assert snapshot_from_payload(snapshot_payload(_snapshot())).signature == ""


# ── levels are bound to the checksummed suspicions ────────────────────────────


def test_relabelling_a_flagged_resource_clean_is_refused() -> None:
    """The checksum covers the suspicion maps, and a consumer renders and alerts
    on the LEVELS. A payload that keeps `1.0` (so the checksum still matches)
    but says CLEAN would clear a flagged resource on every screen and silence
    its alert; the level must be the one its suspicion was derived from."""
    payload = snapshot_payload(_snapshot())
    payload["resource_level"]["db:customers"] = "CLEAN"
    with pytest.raises(SnapshotRejected, match="contradicts"):
        snapshot_from_payload(payload)


def test_a_suspicion_with_no_level_is_refused() -> None:
    payload = snapshot_payload(_snapshot())
    del payload["resource_level"]["s3:exports"]
    with pytest.raises(SnapshotRejected, match="has no level"):
        snapshot_from_payload(payload)


def test_a_snapshot_without_levels_still_arrives() -> None:
    """A legacy snapshot that carries only suspicions has no level to contradict."""
    legacy = _snapshot(resource_level={}, container_level={})
    assert snapshot_from_payload(snapshot_payload(legacy)).resource_level == {}


def test_lowercase_levels_arrive_canonical() -> None:
    """Sentinel <0.4.2 wrote `flagged`; the wire refused it, so no real cycle's
    snapshot could be reported. Either spelling is accepted and handed back in
    the canonical one, which is what consumers compare against."""
    payload = snapshot_payload(_snapshot(
        resource_level={"db:customers": "flagged", "s3:exports": "watch"},
        container_level={"svc:billing": "Watch"},
    ))
    arrived = snapshot_from_payload(payload)
    assert arrived.resource_level == {"db:customers": "FLAGGED", "s3:exports": "WATCH"}
    assert arrived.container_level == {"svc:billing": "WATCH"}

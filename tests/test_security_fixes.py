"""Regression tests for sentinel security fixes (M-1 resource aliasing, M-2 snapshot auth)."""

from __future__ import annotations

import json
import os

import pytest

from axor_sentinel.graph.normalizer import normalize_resource_id
from axor_sentinel.sentinel.snapshot import (
    SNAPSHOT_KEY_ENV,
    ReputationSnapshot,
    atomic_swap,
    load_snapshot,
)


# ── M-1: symlink aliasing collapses to one canonical id ──────────────────────

def test_symlink_and_target_share_canonical_id(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("x")
    link = tmp_path / "alias.txt"
    os.symlink(target, link)

    rid_target, _, _ = normalize_resource_id({"path": str(target)})
    rid_link, _, _ = normalize_resource_id({"path": str(link)})
    assert rid_target == rid_link


def test_relative_path_not_cwd_joined():
    # Relative paths must stay lexical (no realpath / cwd join) for determinism.
    rid, _, _ = normalize_resource_id({"path": "report.csv"})
    assert os.getcwd() not in rid


# ── M-2: snapshot HMAC authentication ────────────────────────────────────────

def _snapshot() -> ReputationSnapshot:
    return ReputationSnapshot(
        version=1, generated_at=0.0,
        resource_reputation={"r1": 0.9}, container_reputation={},
    ).with_checksum()


def test_unsigned_snapshot_loads_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
    atomic_swap(tmp_path, _snapshot())
    # Loads (checksum-only) but warns loudly that it is unauthenticated (F7).
    with pytest.warns(Warning, match="UNAUTHENTICATED"):
        loaded = load_snapshot(tmp_path)
    assert loaded is not None and loaded.resource_reputation["r1"] == 0.9


def test_snapshot_with_unknown_field_still_loads(tmp_path, monkeypatch):
    # Forward-compat: a snapshot written by a newer sentinel (extra top-level key)
    # must load, not be rejected — integrity is over the reputation maps (F8).
    monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
    snap = _snapshot()
    atomic_swap(tmp_path, snap)
    current = tmp_path / "snapshot_current"
    data = json.loads(current.read_text())
    data["future_field"] = {"unexpected": 123}
    current.write_text(json.dumps(data))
    with pytest.warns(Warning):  # unauthenticated warning
        loaded = load_snapshot(tmp_path)
    assert loaded is not None and loaded.resource_reputation["r1"] == 0.9


def test_signed_snapshot_round_trips_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "super-secret-key")
    atomic_swap(tmp_path, _snapshot())
    loaded = load_snapshot(tmp_path)
    assert loaded is not None and loaded.signature


def test_unsigned_snapshot_refused_in_production(tmp_path, monkeypatch):
    # Production requires authenticated snapshots; unsigned + no key → refuse.
    monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
    atomic_swap(tmp_path, _snapshot())
    monkeypatch.setenv("AXOR_ENV", "production")
    with pytest.warns(Warning):
        assert load_snapshot(tmp_path) is None


def test_tampered_scores_rejected_when_key_set(tmp_path, monkeypatch):
    # Attacker writes a snapshot with a valid checksum but no/invalid HMAC.
    monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
    forged = ReputationSnapshot(
        version=2, generated_at=0.0,
        resource_reputation={"r1": 0.0},  # suspicion downgraded to dodge the floor
        container_reputation={},
    ).with_checksum()  # checksum recomputed by attacker, but no signature
    atomic_swap(tmp_path, forged)

    # Defender loads with the key configured — signature missing → refuse.
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "super-secret-key")
    with pytest.warns(Warning):
        assert load_snapshot(tmp_path) is None

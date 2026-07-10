"""Tests for authenticated sentinel state persistence (M7)."""

from __future__ import annotations

import json

from axor_sentinel.sentinel.cycle import SentinelCycle
from axor_sentinel.sentinel.snapshot import SNAPSHOT_KEY_ENV, sign_blob


def _state_dict() -> dict:
    return {
        "version": 3,
        "signal_history": {"r1": ["web", "file"]},
        "prior_counts": {"r1\x00web": 2},
        "baselines": {},
    }


def _write_signed(path, state, sig_override=None):
    serialized = json.dumps(state, sort_keys=True, separators=(",", ":"))
    sig = sig_override if sig_override is not None else sign_blob(serialized)
    path.write_text(json.dumps({"_signed": True, "payload": serialized, "sig": sig}))


def test_signed_state_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "k")
    p = tmp_path / "sentinel_state.json"
    _write_signed(p, _state_dict())
    sh, pc, bl, ver, _ev = SentinelCycle.load_state(p)
    assert ver == 3
    assert pc[("r1", "web")] == 2
    assert sh["r1"] == ["web", "file"]


def test_tampered_signed_state_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "k")
    p = tmp_path / "sentinel_state.json"
    _write_signed(p, _state_dict(), sig_override="deadbeef")  # wrong signature
    sh, pc, bl, ver, _ev = SentinelCycle.load_state(p)
    assert ver == 0 and pc == {} and sh == {}  # cold start


def test_unsigned_state_rejected_when_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "k")
    p = tmp_path / "sentinel_state.json"
    p.write_text(json.dumps(_state_dict()))  # legacy flat, no signature
    _, _, _, ver, _ev = SentinelCycle.load_state(p)
    assert ver == 0  # cold start — unsigned not trusted under a key


def test_unsigned_state_ok_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
    monkeypatch.delenv("AXOR_ENV", raising=False)
    monkeypatch.delenv("AXOR_SNAPSHOT_REQUIRE_SIGNATURE", raising=False)
    p = tmp_path / "sentinel_state.json"
    p.write_text(json.dumps(_state_dict()))
    _, _, _, ver, _ev = SentinelCycle.load_state(p)
    assert ver == 3  # legacy behaviour preserved when no key configured

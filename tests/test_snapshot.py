"""
Adversarial tests for snapshot integrity.

Covers invariants A-5, A-16, A-17 and spec adversarial test matrix:
  - POSIX symlink swap atomicity (2 variants)
  - Windows os.replace path (2 variants — mocked)
  - Checksum failure retains previous + emits warning (2 variants)
  - Network mount warning at startup (1 variant)
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import os
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from axor_sentinel.sentinel.snapshot import (
    SNAPSHOT_KEY_ENV,
    AuditIntegrityWarning,
    ReputationSnapshot,
    SnapshotVersionRegression,
    _signature_payload,
    atomic_swap,
    latest_snapshot_version,
    load_snapshot,
    snapshot_payload,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_snap(version: int, scores: dict[str, float] | None = None) -> ReputationSnapshot:
    return ReputationSnapshot(
        version=version,
        generated_at=time.time(),
        resource_reputation=scores or {"r1": 0.5},
        container_reputation={"c1": 0.4},
    ).with_checksum()


# ── POSIX symlink swap — 2 variants ───────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestPosixSymlinkSwap:
    def test_symlink_created_and_points_to_version_file(self, tmp_path):
        """atomic_swap creates snapshot_current symlink pointing to versioned file."""
        snap = _make_snap(1)
        atomic_swap(tmp_path, snap)

        current = tmp_path / "snapshot_current"
        assert current.exists()
        assert current.is_symlink()
        # Symlink target should be the versioned file
        target = current.readlink()
        assert "snapshot_v1" in str(target)

    def test_second_swap_replaces_symlink_atomically(self, tmp_path):
        """Second atomic_swap replaces symlink; reader never sees missing link."""
        snap1 = _make_snap(1, {"r1": 0.3})
        snap2 = _make_snap(2, {"r1": 0.7})

        atomic_swap(tmp_path, snap1)
        atomic_swap(tmp_path, snap2)

        loaded = load_snapshot(tmp_path)
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.resource_reputation["r1"] == 0.7


# ── Windows os.replace path — 2 variants ──────────────────────────────────────

class TestWindowsReplace:
    def test_windows_path_uses_os_replace_not_rename(self, tmp_path):
        """On Windows, atomic_swap must use os.replace, not os.rename over a file."""
        snap = _make_snap(1)
        with patch("sys.platform", "win32"), patch("os.replace") as mock_replace:
            # Provide a no-op replace that writes the file
            def fake_replace(src, dst):
                import shutil
                shutil.copy2(str(src), str(dst))
            mock_replace.side_effect = fake_replace
            atomic_swap(tmp_path, snap)
        # os.replace now also lands the version file (temp + rename, never an
        # in-place rewrite), so assert on the call that makes it visible rather
        # than on the call count.
        assert any(
            call.args[1] == tmp_path / "snapshot_current"
            for call in mock_replace.call_args_list
        )

    def test_windows_no_symlink_created(self, tmp_path):
        """On Windows path, current snapshot file is a regular file, not a symlink."""
        snap = _make_snap(1)
        with patch("sys.platform", "win32"):
            atomic_swap(tmp_path, snap)
        current = tmp_path / "snapshot_current"
        assert current.exists()
        # On POSIX machine running the Windows code path: current should NOT be a symlink
        # (the Windows code creates a temp file and calls os.replace)
        if sys.platform != "win32":
            # We patched sys.platform, so the rename branch was skipped;
            # os.replace was called instead — current is a regular copy
            assert not current.is_symlink()


# ── Checksum failure — 2 variants ─────────────────────────────────────────────

class TestChecksumFailure:
    def test_corrupted_checksum_returns_none_and_warns(self, tmp_path):
        """
        Invariant A-5: corrupted snapshot → load returns None and emits AuditIntegrityWarning.
        Previous snapshot (held by caller) is retained.
        """
        snap = _make_snap(1)
        atomic_swap(tmp_path, snap)

        # Corrupt the checksum in the current file
        current = tmp_path / "snapshot_current"
        data = json.loads(current.read_text())
        data["checksum"] = "badhash"
        current.write_text(json.dumps(data))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_snapshot(tmp_path)

        assert result is None
        audit_warnings = [w for w in caught if issubclass(w.category, AuditIntegrityWarning)]
        assert len(audit_warnings) >= 1

    def test_truncated_json_returns_none_and_warns(self, tmp_path):
        """Malformed JSON in snapshot file → load returns None and emits warning."""
        snap = _make_snap(1)
        atomic_swap(tmp_path, snap)

        current = tmp_path / "snapshot_current"
        current.write_text("{broken json")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_snapshot(tmp_path)

        assert result is None
        assert len(caught) >= 1


# ── Network mount warning — 1 variant ─────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX /proc/mounts only")
class TestNetworkMountWarning:
    def test_network_mount_emits_hard_warning(self, tmp_path):
        """
        Invariant A-17: snapshot_dir on network filesystem → hard UserWarning emitted.

        Uses the _mounts_path parameter to inject a fake /proc/mounts file so
        the test doesn't depend on the real system mount table.
        """
        from axor_sentinel.sentinel.snapshot import _warn_if_network_mount

        # Write a fake mounts file that maps tmp_path to an NFS4 mount
        fake_mounts = tmp_path / "fake_proc_mounts"
        fake_mounts.write_text(
            f"server:/export {str(tmp_path)} nfs4 defaults 0 0\n",
            encoding="utf-8",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_network_mount(tmp_path, _mounts_path=fake_mounts)

        network_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "network" in str(w.message).lower()
        ]
        assert len(network_warnings) >= 1, (
            "Expected at least one UserWarning mentioning 'network' "
            f"for a path under an NFS mount, got: {[str(w.message) for w in caught]}"
        )

    def test_local_mount_no_warning(self, tmp_path):
        """Non-network filesystem → no warning emitted."""
        from axor_sentinel.sentinel.snapshot import _warn_if_network_mount

        # Mounts file with only ext4 — no network filesystems
        fake_mounts = tmp_path / "fake_proc_mounts"
        fake_mounts.write_text(
            f"/dev/sda1 {str(tmp_path)} ext4 defaults 0 0\n",
            encoding="utf-8",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_network_mount(tmp_path, _mounts_path=fake_mounts)

        network_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and "network" in str(w.message).lower()
        ]
        assert len(network_warnings) == 0


# ── Signature scope: the HMAC covers the whole snapshot ──────────────────────

_KEY = "snapshot-test-key"


def _leveled(version: int) -> ReputationSnapshot:
    """A snapshot the way a real cycle writes it: levels bound to suspicions."""
    return ReputationSnapshot(
        version=version,
        generated_at=1_700_000_000.0,
        resource_reputation={"db:customers": 1.0},
        container_reputation={"svc:billing": 0.4},
        resource_level={"db:customers": "FLAGGED"},
        container_level={"svc:billing": "WATCH"},
        verdict_facts={"db:customers": ["P4 staged-then-export"]},
    ).with_checksum()


def _rewrite_live(tmp_path, **changes) -> None:
    """Edit the live snapshot file the way a writer to the dir would: the maps
    (and so the checksum) untouched, the signature left as it was."""
    current = tmp_path / "snapshot_current"
    target = tmp_path / current.readlink() if current.is_symlink() else current
    data = json.loads(target.read_text())
    data.update(changes)
    target.unlink()
    target.write_text(json.dumps(data))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestSignatureCoversFullSnapshot:
    @pytest.mark.parametrize("changes", [
        {"version": 999},                                    # fake a newer version
        {"generated_at": 0.0},                               # backdate it
        {"verdict_facts": {"db:customers": ["nothing to see"]}},
        {"resource_score_telemetry": {"db:customers": 0.0}},
    ])
    def test_edits_outside_the_maps_are_refused_under_a_key(
        self, tmp_path, monkeypatch, changes
    ):
        """0.4.2 and earlier signed only the reputation maps, so all of these
        loaded under a still-valid signature."""
        monkeypatch.setenv(SNAPSHOT_KEY_ENV, _KEY)
        atomic_swap(tmp_path, _leveled(1))
        assert load_snapshot(tmp_path) is not None
        _rewrite_live(tmp_path, **changes)
        with pytest.warns(AuditIntegrityWarning, match="signature"):
            assert load_snapshot(tmp_path) is None

    def test_the_checksum_scope_is_unchanged(self):
        """The checksum is wire format (snapshot_from_payload in the control
        plane): still the two maps, so editing a level does not move it."""
        snap = _leveled(1)
        relabelled = dataclasses.replace(snap, resource_level={"db:customers": "WATCH"})
        assert relabelled.compute_checksum() == snap.checksum

    def test_a_maps_only_signature_no_longer_verifies(self, tmp_path, monkeypatch):
        """A snapshot signed by 0.4.2 or earlier (HMAC over the maps alone) is
        refused under a key until the next cycle rewrites it — there is no
        legacy fallback, because accepting one would reopen the relabel."""
        monkeypatch.setenv(SNAPSHOT_KEY_ENV, _KEY)
        atomic_swap(tmp_path, _leveled(1))
        legacy_sig = hmac.new(
            _KEY.encode(), _leveled(1)._canonical_payload(), hashlib.sha256
        ).hexdigest()
        _rewrite_live(tmp_path, signature=legacy_sig)
        with pytest.warns(AuditIntegrityWarning, match="0.4.2"):
            assert load_snapshot(tmp_path) is None

    def test_a_field_from_a_newer_sentinel_still_verifies(self, tmp_path, monkeypatch):
        """The signature is checked over the raw stored object, so a field this
        version does not know (and drops) is still covered — a newer writer's
        signed snapshot loads here instead of failing verification."""
        monkeypatch.setenv(SNAPSHOT_KEY_ENV, _KEY)
        data = snapshot_payload(_leveled(1))
        data["future_field"] = {"x": 1}
        data["signature"] = hmac.new(
            _KEY.encode(), _signature_payload(data), hashlib.sha256
        ).hexdigest()
        (tmp_path / "snapshot_v1.json").write_text(json.dumps(data))
        (tmp_path / "snapshot_current").symlink_to("snapshot_v1.json")
        loaded = load_snapshot(tmp_path)
        assert loaded is not None and loaded.version == 1

        # ...and the unknown field is covered: editing it breaks the signature.
        _rewrite_live(tmp_path, future_field={"x": 2})
        with pytest.warns(AuditIntegrityWarning):
            assert load_snapshot(tmp_path) is None

    def test_a_stale_caller_signature_is_replaced_on_write(self, tmp_path, monkeypatch):
        """with_checksum() then replace(version=...) leaves a signature over the
        OLD version; the writer holding the key re-signs what it publishes."""
        monkeypatch.setenv(SNAPSHOT_KEY_ENV, _KEY)
        stale = dataclasses.replace(_leveled(1), version=2)
        atomic_swap(tmp_path, stale)
        loaded = load_snapshot(tmp_path)
        assert loaded is not None and loaded.version == 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestLoadBindsLevels:
    """load_snapshot runs the same level/suspicion binding the wire reader does."""

    def test_relabelling_flagged_clean_is_refused_even_unkeyed(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
        atomic_swap(tmp_path, _leveled(1))
        _rewrite_live(tmp_path, resource_level={"db:customers": "CLEAN"})
        with pytest.warns(AuditIntegrityWarning, match="contradicts"):
            assert load_snapshot(tmp_path) is None

    def test_a_suspicion_without_a_level_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
        atomic_swap(tmp_path, _leveled(1))
        # A non-empty level map that simply omits the FLAGGED resource (an
        # empty map is a legacy snapshot and has nothing to contradict).
        _rewrite_live(tmp_path, resource_level={"other": "CLEAN"})
        with pytest.warns(AuditIntegrityWarning, match="has no level"):
            assert load_snapshot(tmp_path) is None

    def test_lowercase_levels_from_older_cycles_still_load(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
        snap = dataclasses.replace(
            _leveled(1),
            resource_level={"db:customers": "flagged"},
            container_level={"svc:billing": "watch"},
        )
        atomic_swap(tmp_path, snap)
        with pytest.warns(AuditIntegrityWarning, match="UNAUTHENTICATED"):
            assert load_snapshot(tmp_path) is not None

    def test_a_negative_version_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv(SNAPSHOT_KEY_ENV, raising=False)
        atomic_swap(tmp_path, _leveled(1))
        _rewrite_live(tmp_path, version=-5)
        with pytest.warns(AuditIntegrityWarning, match="non-negative"):
            assert load_snapshot(tmp_path) is None


# ── Writer: versions only move forward, files are never rewritten in place ───

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestWriterMonotonicity:
    @pytest.mark.parametrize("version", [1, 2])
    def test_version_at_or_below_live_is_refused(self, tmp_path, version):
        atomic_swap(tmp_path, _make_snap(2))
        before = (tmp_path / "snapshot_v2.json").read_bytes()
        with pytest.raises(SnapshotVersionRegression):
            atomic_swap(tmp_path, _make_snap(version, {"r1": 0.9}))
        # the live file and link are untouched
        assert (tmp_path / "snapshot_v2.json").read_bytes() == before
        assert (tmp_path / "snapshot_current").readlink().name == "snapshot_v2.json"

    def test_version_file_is_written_by_rename_not_in_place(self, tmp_path):
        """The version file lands via temp + os.replace, so a reader holding the
        old inode (or a crash mid-write) never sees a partial file."""
        real_replace = os.replace
        replaced: list[str] = []

        def _spy(src, dst):
            replaced.append(Path(dst).name)
            return real_replace(src, dst)

        with patch("axor_sentinel.sentinel.snapshot.os.replace", side_effect=_spy):
            atomic_swap(tmp_path, _make_snap(1))
        assert "snapshot_v1.json" in replaced
        assert not list(tmp_path.glob("*.tmp"))

    def test_stale_links_from_a_crash_are_pruned(self, tmp_path):
        """A crash between symlink_to and rename leaves snapshot_link_vN behind;
        a same-version one used to make the next symlink_to raise."""
        atomic_swap(tmp_path, _make_snap(1))
        (tmp_path / "snapshot_link_v1").symlink_to("snapshot_v1.json")
        (tmp_path / "snapshot_link_v2").symlink_to("snapshot_v2.json")  # same version
        atomic_swap(tmp_path, _make_snap(2))
        assert not list(tmp_path.glob("snapshot_link_v*"))
        assert (tmp_path / "snapshot_current").is_symlink()
        assert (tmp_path / "snapshot_current").readlink().name == "snapshot_v2.json"

    def test_stale_temp_files_are_pruned(self, tmp_path):
        (tmp_path / "snapshot_v1.json.4242.tmp").write_text("{partial")
        (tmp_path / "snapshot_v9.json.4242.tmp").write_text("{in flight, newer")
        atomic_swap(tmp_path, _make_snap(2))
        assert not (tmp_path / "snapshot_v1.json.4242.tmp").exists()
        assert (tmp_path / "snapshot_v9.json.4242.tmp").exists()  # newer: left alone

    def test_directory_is_fsynced_after_the_swap(self, tmp_path):
        with patch("axor_sentinel.sentinel.snapshot._fsync_dir") as fsync_dir:
            atomic_swap(tmp_path, _make_snap(1))
        assert any(c.args[0] == tmp_path for c in fsync_dir.call_args_list)

    def test_latest_version_sees_retained_files_and_the_live_link(self, tmp_path):
        assert latest_snapshot_version(tmp_path) == 0
        for v in range(1, 6):
            atomic_swap(tmp_path, _make_snap(v))
        assert latest_snapshot_version(tmp_path) == 5
        # the live link alone is enough (all version files gone)
        for f in tmp_path.glob("snapshot_v*.json"):
            f.unlink()
        assert latest_snapshot_version(tmp_path) == 5

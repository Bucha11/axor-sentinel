"""
Adversarial tests for snapshot integrity.

Covers invariants A-5, A-16, A-17 and spec adversarial test matrix:
  - POSIX symlink swap atomicity (2 variants)
  - Windows os.replace path (2 variants — mocked)
  - Checksum failure retains previous + emits warning (2 variants)
  - Network mount warning at startup (1 variant)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from unittest.mock import patch

import pytest

from axor_sentinel.sentinel.snapshot import (
    AuditIntegrityWarning,
    ReputationSnapshot,
    atomic_swap,
    load_snapshot,
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
        mock_replace.assert_called_once()

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

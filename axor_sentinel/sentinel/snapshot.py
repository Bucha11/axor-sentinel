from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

log = logging.getLogger("axor.sentinel.snapshot")

# Number of old snapshot version files to keep alongside the current symlink.
SNAPSHOT_RETAIN_VERSIONS: int = 3


class AuditIntegrityWarning(UserWarning):
    """Emitted when snapshot checksum verification fails."""


@dataclass(frozen=True)
class ReputationSnapshot:
    """
    Versioned, checksummed reputation snapshot delivered atomically to axor-core.

    version:              monotonically increasing integer
    generated_at:         unix timestamp when the snapshot was written
    resource_reputation:  resource_id → suspicion_score mapping
    container_reputation: container_id → suspicion_score mapping
    checksum:             SHA-256 of the serialized resource/container maps
    """
    version: int
    generated_at: float
    resource_reputation: dict[str, float] = field(default_factory=dict)
    container_reputation: dict[str, float] = field(default_factory=dict)
    checksum: str = ""

    def compute_checksum(self) -> str:
        """SHA-256 of the reputation maps, deterministically serialized."""
        payload = json.dumps(
            {
                "resource_reputation": self.resource_reputation,
                "container_reputation": self.container_reputation,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def with_checksum(self) -> "ReputationSnapshot":
        """Return a copy of this snapshot with the checksum field populated."""
        return replace(self, checksum=self.compute_checksum())


def _serialize(snapshot: ReputationSnapshot) -> str:
    return json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"))


def _deserialize(text: str) -> ReputationSnapshot:
    data = json.loads(text)
    return ReputationSnapshot(**data)


def atomic_swap(snapshot_dir: Path, new_snapshot: ReputationSnapshot) -> None:
    """
    Write snapshot atomically so axor-core never reads a partial write.

    Protocol (invariant A-5, A-16):
      1. Write new version file and fsync.
      2. Verify checksum before making it visible.
      3. Create a new symlink alongside current (POSIX) or use os.replace (Windows).
      4. Atomic rename/replace of symlink.
      5. Prune old version files (keep SNAPSHOT_RETAIN_VERSIONS).

    On POSIX: os.rename over an existing symlink is atomic.
    On Windows: os.replace is used — atomic on same-volume (invariant A-16).
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    version_file = snapshot_dir / f"snapshot_v{new_snapshot.version}.json"
    serialized = _serialize(new_snapshot)

    # 1. Write and fsync new version file
    version_file.write_text(serialized, encoding="utf-8")
    with version_file.open("rb") as fh:
        os.fsync(fh.fileno())

    # 2. Verify checksum from in-memory bytes before making visible (invariant A-5).
    # Hashing the serialized bytes avoids a second file read (which could return stale
    # data on NFS) and ensures exactly what was written is what was verified.
    _verify_checksum_bytes(serialized.encode(), new_snapshot.checksum)

    current_link = snapshot_dir / "snapshot_current"

    if sys.platform == "win32":
        # Windows: os.replace is atomic on same-volume (invariant A-16)
        # Write to a temp file then replace — Windows cannot rename over a symlink
        temp_file = snapshot_dir / f"snapshot_v{new_snapshot.version}_current.json"
        temp_file.write_text(serialized, encoding="utf-8")
        os.replace(temp_file, current_link)
    else:
        # POSIX: symlink rename is atomic even over an existing symlink
        new_link = snapshot_dir / f"snapshot_link_v{new_snapshot.version}"
        new_link.symlink_to(version_file.name)
        os.rename(new_link, current_link)

    # 5. Prune old version files — keep last SNAPSHOT_RETAIN_VERSIONS
    _prune_old_versions(snapshot_dir, new_snapshot.version, keep=SNAPSHOT_RETAIN_VERSIONS)


def load_snapshot(snapshot_dir: Path) -> ReputationSnapshot | None:
    """
    Load and verify the current snapshot from snapshot_dir.

    Returns None if no snapshot exists yet.
    On checksum failure: emits AuditIntegrityWarning and returns None
    (caller retains its previous snapshot — invariant A-5).
    """
    snapshot_dir = Path(snapshot_dir)
    current_link = snapshot_dir / "snapshot_current"
    if not current_link.exists():
        return None
    try:
        text = current_link.read_text(encoding="utf-8")
        snapshot = _deserialize(text)
        expected = snapshot.compute_checksum()
        if snapshot.checksum != expected:
            warnings.warn(
                f"snapshot checksum mismatch: stored={snapshot.checksum!r} "
                f"computed={expected!r} — retaining previous snapshot",
                AuditIntegrityWarning,
                stacklevel=2,
            )
            return None
        return snapshot
    except Exception as exc:
        warnings.warn(
            f"failed to load snapshot: {exc} — retaining previous snapshot",
            AuditIntegrityWarning,
            stacklevel=2,
        )
        return None


def validate_snapshot_dir(
    snapshot_dir: Path,
    _mounts_path: Path | None = None,
) -> None:
    """
    Emit a hard warning if snapshot_dir appears to be a network mount.

    Network filesystems break the atomicity guarantees of os.rename/os.replace
    regardless of OS. This is a deployment constraint — not enforced in code
    (invariant A-17).

    Args:
        snapshot_dir:  directory to validate.
        _mounts_path:  override the mounts file path (default ``/proc/mounts``).
                       Intended for testing only.
    """
    snapshot_dir = Path(snapshot_dir)
    try:
        os.statvfs(snapshot_dir)  # POSIX only — validates accessible
        # f_flag bit 1 (ST_RDONLY=1) is not NFS-specific; check mount type via /proc/mounts
        _warn_if_network_mount(snapshot_dir, _mounts_path=_mounts_path)
    except AttributeError:
        # Windows: os.statvfs not available; skip detection
        pass
    except Exception:
        pass


def _warn_if_network_mount(
    path: Path,
    _mounts_path: Path | None = None,
) -> None:
    """
    Check /proc/mounts for NFS/CIFS/SMB entries that include path.

    Args:
        path:          path to check.
        _mounts_path:  override the mounts file (default ``/proc/mounts``).
                       Intended for testing only.
    """
    mounts_file = _mounts_path or Path("/proc/mounts")
    if not mounts_file.exists():
        return
    try:
        path_str = str(path.resolve())
        for line in mounts_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mount_point, fs_type = parts[1], parts[2]
            if fs_type.lower() in ("nfs", "nfs4", "cifs", "smbfs", "fuse.sshfs"):
                # Check if our path is under this mount point
                try:
                    Path(path_str).relative_to(mount_point)
                    warnings.warn(
                        f"snapshot_dir '{path}' appears to be on a network filesystem "
                        f"({fs_type} at {mount_point}). Atomic swap guarantees are broken "
                        "on network mounts — use a local volume (invariant A-17).",
                        UserWarning,
                        stacklevel=4,
                    )
                    return
                except ValueError:
                    pass
    except Exception:
        pass


# ── Internal helpers ───────────────────────────────────────────────────────────

def _verify_checksum_bytes(serialized_bytes: bytes, expected: str) -> None:
    """Verify checksum against already-serialized bytes (not a file re-read)."""
    data = json.loads(serialized_bytes)
    payload = json.dumps(
        {
            "resource_reputation": data.get("resource_reputation", {}),
            "container_reputation": data.get("container_reputation", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(
            f"snapshot checksum verification failed: expected={expected!r} actual={actual!r}"
        )


def _prune_old_versions(snapshot_dir: Path, current_version: int, keep: int) -> None:
    """Remove version files older than (current_version - keep)."""
    cutoff = current_version - keep
    for f in snapshot_dir.glob("snapshot_v*.json"):
        try:
            # Extract version number from filename
            version_str = f.stem.replace("snapshot_v", "")
            version = int(version_str)
            if version <= cutoff:
                f.unlink(missing_ok=True)
        except (ValueError, OSError):
            pass

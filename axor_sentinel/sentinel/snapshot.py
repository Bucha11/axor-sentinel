from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

log = logging.getLogger("axor.sentinel.snapshot")

# Number of old snapshot version files to keep alongside the current symlink.
SNAPSHOT_RETAIN_VERSIONS: int = 3

# Env var holding the HMAC key used to authenticate snapshots. When set, the
# checksum alone is no longer trusted — a valid HMAC signature is required on
# load. The key must live outside the snapshot directory and be unreachable by
# the governed agent. When unset, behaviour falls back to checksum-only
# integrity (corruption detection, NOT tamper protection).
SNAPSHOT_KEY_ENV: str = "AXOR_SNAPSHOT_KEY"


def _snapshot_key() -> bytes | None:
    raw = os.environ.get(SNAPSHOT_KEY_ENV, "")
    return raw.encode() if raw else None


def _signature_required() -> bool:
    """True when an authenticated signature is mandatory to load a snapshot.

    Enabled explicitly via AXOR_SNAPSHOT_REQUIRE_SIGNATURE, or implicitly when
    running in a production environment (AXOR_ENV=production). In that mode an
    unsigned snapshot — or any snapshot when no key is configured — is refused
    (fail-closed): the reputation gate must not run on unauthenticated data.
    """
    if os.environ.get("AXOR_SNAPSHOT_REQUIRE_SIGNATURE", "").lower() in ("1", "true", "yes"):
        return True
    return os.environ.get("AXOR_ENV", "").lower() == "production"


def sign_blob(serialized: str) -> str | None:
    """HMAC-sign an arbitrary serialized blob with the snapshot key.

    Returns the hex signature, or None when no key is configured. Reused for any
    locally-persisted state that must be tamper-evident (e.g. sentinel_state.json).
    """
    key = _snapshot_key()
    if key is None:
        return None
    return hmac.new(key, serialized.encode(), hashlib.sha256).hexdigest()


def verify_blob(serialized: str, signature: str | None) -> bool:
    """Return True if a persisted blob is safe to load.

    - key configured  → require a matching HMAC signature.
    - no key, prod / require-signature → refuse (fail-closed).
    - no key, otherwise → accept (integrity not enforced).
    """
    key = _snapshot_key()
    if key is not None:
        if not signature:
            return False
        expected = hmac.new(key, serialized.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)
    if _signature_required():
        return False
    return True


class AuditIntegrityWarning(UserWarning):
    """Emitted when snapshot checksum or signature verification fails."""


@dataclass(frozen=True)
class ReputationSnapshot:
    """
    Versioned, checksummed reputation snapshot delivered atomically to axor-core.

    version:              monotonically increasing integer
    generated_at:         unix timestamp when the snapshot was written
    resource_reputation:  resource_id → suspicion, FINITE codomain — the values
                          are LEVEL_SUSPICION[level] from the deterministic
                          verdict layer (0.0 clean / 0.4 watch / 1.0 flagged),
                          so core's detection_floor comparison is decidable
                          end-to-end
    container_reputation: container_id → suspicion, same finite codomain
    checksum:             SHA-256 of the serialized resource/container maps
    """
    version: int
    generated_at: float
    resource_reputation: dict[str, float] = field(default_factory=dict)
    container_reputation: dict[str, float] = field(default_factory=dict)
    checksum: str = ""
    signature: str = ""
    # Deterministic verdicts (predicates.py): id → level name, plus the facts
    # behind each non-clean resource verdict. Forward-compatible: an older
    # loader drops unknown keys. Once the deterministic codomain is
    # authoritative, the reputation floats above are DERIVED from these levels,
    # so integrity transfers through the checksummed maps.
    resource_level: dict[str, str] = field(default_factory=dict)
    container_level: dict[str, str] = field(default_factory=dict)
    verdict_facts: dict[str, list[str]] = field(default_factory=dict)
    # Demoted scalar scores (accumulate/decay path) — non-load-bearing
    # telemetry, kept for observability while the deterministic levels are
    # authoritative for the reputation maps above.
    resource_score_telemetry: dict[str, float] = field(default_factory=dict)
    container_score_telemetry: dict[str, float] = field(default_factory=dict)

    def _canonical_payload(self) -> bytes:
        return json.dumps(
            {
                "resource_reputation": self.resource_reputation,
                "container_reputation": self.container_reputation,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def compute_checksum(self) -> str:
        """SHA-256 of the reputation maps, deterministically serialized."""
        return hashlib.sha256(self._canonical_payload()).hexdigest()

    def compute_signature(self, key: bytes) -> str:
        """HMAC-SHA256 of the reputation maps under the given key."""
        return hmac.new(key, self._canonical_payload(), hashlib.sha256).hexdigest()

    def with_checksum(self) -> "ReputationSnapshot":
        """Return a copy with checksum populated, and signature if a key is set."""
        updated = replace(self, checksum=self.compute_checksum())
        key = _snapshot_key()
        if key is not None:
            updated = replace(updated, signature=updated.compute_signature(key))
        return updated


def _serialize(snapshot: ReputationSnapshot) -> str:
    return json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"))


def _deserialize(text: str) -> ReputationSnapshot:
    data = json.loads(text)
    # Forward-compatible: ignore unknown top-level keys rather than raising on a
    # snapshot written by a newer sentinel that added a field. Integrity is still
    # enforced by checksum/signature over the reputation maps, which a stray field
    # cannot alter.
    known = {f.name for f in fields(ReputationSnapshot)}
    return ReputationSnapshot(**{k: v for k, v in data.items() if k in known})


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

    # Attach an HMAC signature if a key is configured and one is not already set,
    # so snapshots written by this process are tamper-evident on load (M-2).
    key = _snapshot_key()
    if key is not None and not new_snapshot.signature:
        new_snapshot = replace(
            new_snapshot, signature=new_snapshot.compute_signature(key)
        )

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
        if not hmac.compare_digest(snapshot.checksum, expected):
            warnings.warn(
                f"snapshot checksum mismatch: stored={snapshot.checksum!r} "
                f"computed={expected!r} — retaining previous snapshot",
                AuditIntegrityWarning,
                stacklevel=2,
            )
            return None
        # Authenticate the snapshot when a key is configured. A correct checksum
        # is not sufficient — an attacker with write access can recompute it.
        key = _snapshot_key()
        if key is not None:
            expected_sig = snapshot.compute_signature(key)
            if not snapshot.signature or not hmac.compare_digest(
                snapshot.signature, expected_sig
            ):
                warnings.warn(
                    "snapshot signature invalid or missing while "
                    f"{SNAPSHOT_KEY_ENV} is set — refusing to load (possible "
                    "tampering); retaining previous snapshot",
                    AuditIntegrityWarning,
                    stacklevel=2,
                )
                return None
        elif _signature_required():
            # Production requires authenticated snapshots but no key is configured,
            # so the signature cannot be verified — fail closed.
            warnings.warn(
                "snapshot signature is required (production / "
                "AXOR_SNAPSHOT_REQUIRE_SIGNATURE) but no "
                f"{SNAPSHOT_KEY_ENV} is configured — refusing to load; "
                "retaining previous snapshot",
                AuditIntegrityWarning,
                stacklevel=2,
            )
            return None
        else:
            # No key and not required: the snapshot is protected by checksum only,
            # which any process with write access to the snapshot dir can recompute
            # after tampering. Warn loudly so an operator running unauthenticated is
            # aware — set AXOR_SNAPSHOT_KEY (and AXOR_SNAPSHOT_REQUIRE_SIGNATURE /
            # AXOR_ENV=production) to fail closed on tampering.
            warnings.warn(
                "loading an UNAUTHENTICATED snapshot (checksum only, no HMAC): a "
                f"writer to the snapshot dir can forge scores. Set {SNAPSHOT_KEY_ENV} "
                "to authenticate.",
                AuditIntegrityWarning,
                stacklevel=2,
            )
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

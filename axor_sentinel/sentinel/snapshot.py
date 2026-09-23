from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from axor_sentinel.sentinel.predicates import LEVEL_SUSPICION, ReputationLevel

log = logging.getLogger("axor.sentinel.snapshot")

# The two vocabularies a snapshot is written in, read from the module that
# DEFINES them rather than spelled again: the finite suspicion codomain and
# the level names beside it.
_SUSPICION_VALUES: frozenset[float] = frozenset(LEVEL_SUSPICION.values())
_LEVEL_NAMES: frozenset[str] = frozenset(level.name for level in ReputationLevel)

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
    return not _signature_required()


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
    checksum:             SHA-256 of the serialized resource/container maps.
                          UNKEYED, so it is corruption detection only: anyone
                          who can rewrite the maps can recompute it. Its scope
                          is part of the wire format (snapshot_from_payload,
                          consumed by axor-control-plane) and stays the two maps.
    signature:            HMAC-SHA256 under AXOR_SNAPSHOT_KEY over the WHOLE
                          canonical snapshot — every field except checksum and
                          signature themselves (see _signature_payload). This is
                          the tamper protection; the checksum is not.
    """
    version: int
    generated_at: float
    resource_reputation: dict[str, float] = field(default_factory=dict)
    container_reputation: dict[str, float] = field(default_factory=dict)
    checksum: str = ""
    signature: str = ""
    # Deterministic verdicts (predicates.py): id → level name, plus the facts
    # behind each non-clean resource verdict. Forward-compatible: an older
    # loader drops unknown keys. The reputation floats above are DERIVED from
    # these levels, and both readers (load_snapshot, snapshot_from_payload)
    # refuse a snapshot whose levels contradict its suspicions — so the levels
    # are bound to the checksummed maps. Authenticity of all of it (levels,
    # facts, version, generated_at) comes only from the HMAC signature.
    resource_level: dict[str, str] = field(default_factory=dict)
    container_level: dict[str, str] = field(default_factory=dict)
    verdict_facts: dict[str, list[str]] = field(default_factory=dict)
    # Demoted scalar scores (accumulate/decay path) — non-load-bearing
    # telemetry, kept for observability while the deterministic levels are
    # authoritative for the reputation maps above.
    resource_score_telemetry: dict[str, float] = field(default_factory=dict)
    container_score_telemetry: dict[str, float] = field(default_factory=dict)

    def _canonical_payload(self) -> bytes:
        """The bytes the CHECKSUM covers: the two reputation maps only.

        Deliberately narrow and frozen — it is the wire format a control plane
        verifies (snapshot_from_payload), and widening it would make every
        snapshot from a newer sentinel fail an older plane's check. The
        signature does NOT use this; it covers the full snapshot."""
        return json.dumps(
            {
                "resource_reputation": self.resource_reputation,
                "container_reputation": self.container_reputation,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def compute_checksum(self) -> str:
        """SHA-256 of the reputation maps, deterministically serialized.

        Corruption detection only — unkeyed, so recomputable by any writer."""
        return hashlib.sha256(self._canonical_payload()).hexdigest()

    def compute_signature(self, key: bytes) -> str:
        """HMAC-SHA256 of the full canonical snapshot under the given key.

        Covers version, generated_at, both reputation maps, both level maps,
        verdict_facts and the telemetry maps — everything but checksum and
        signature. Signing only the maps (as 0.4.2 and earlier did) let a
        writer to the snapshot dir relabel levels, fake the version, or edit
        facts under a still-valid signature."""
        return _hmac_hex(key, _signature_payload(asdict(self)))

    def with_checksum(self) -> ReputationSnapshot:
        """Return a copy with checksum populated, and signature if a key is set."""
        updated = replace(self, checksum=self.compute_checksum())
        key = _snapshot_key()
        if key is not None:
            updated = replace(updated, signature=updated.compute_signature(key))
        return updated


# Domain separation for the snapshot HMAC. The same AXOR_SNAPSHOT_KEY also signs
# sentinel_state.json (sign_blob); the prefix guarantees a signature minted for
# one kind of blob can never verify as the other, and it versions the scope, so
# a maps-only signature from 0.4.2 and earlier can never verify under the full scope.
_SIGNATURE_DOMAIN: bytes = b"axor-sentinel/snapshot/v2\x00"
_UNSIGNED_FIELDS: frozenset[str] = frozenset({"checksum", "signature"})


def _signature_payload(data: dict) -> bytes:
    """Canonical bytes the snapshot HMAC covers: every top-level field of the
    snapshot as stored except checksum and signature.

    Takes the raw mapping, not the dataclass, so load_snapshot can verify over
    exactly what is on disk — including a field a NEWER sentinel added that this
    version's dataclass does not know. Verifying over the dataclass would drop
    that field and fail every snapshot a newer writer signed (breaking the
    forward-compatibility _deserialize promises); verifying over the raw object
    keeps it and still covers every field this version reads."""
    body = {k: v for k, v in data.items() if k not in _UNSIGNED_FIELDS}
    return _SIGNATURE_DOMAIN + json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()


def _hmac_hex(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


class SnapshotRejected(ValueError):
    """A payload that is not a usable ReputationSnapshot."""


class SnapshotVersionRegression(ValueError):
    """atomic_swap was asked to publish a version <= the one already live.

    Versions are the only ordering a reader has: two different snapshots at
    one version, or a lower version after a higher one, are indistinguishable
    from a rollback. The writer refuses rather than emit either."""


# The snapshot over a wire, not a filesystem.
#
# `atomic_swap` / `load_snapshot` deliver a snapshot to a reader on the SAME
# host — the enricher on the governance hot path, reading a symlink the cycle
# swapped. A control plane is not on that host: it renders the reputation a
# node's sentinel computed, so the snapshot has to travel, and the shape it
# travels in belongs here beside the dataclass rather than in whatever consumer
# happens to need it first.
#
# The checksum comes along and is CHECKED on arrival — but it is an UNKEYED
# SHA-256, so what it proves is narrow: the maps were not corrupted or naively
# edited between the node and the plane. Anyone who rewrites the maps can
# recompute it, so it is NOT tamper protection; authenticity over a wire has to
# come from the transport (TLS / an authenticated channel to the node). The
# HMAC signature is the stronger claim and stays optional here: it is keyed to
# the node's own AXOR_SNAPSHOT_KEY, which a plane does not hold and must not.


def snapshot_payload(snapshot: ReputationSnapshot) -> dict:
    """The snapshot as JSON-ready data, checksum included."""
    return asdict(snapshot)


def _finite_number(value: object, what: str) -> float:
    """``value`` as a finite float, or SnapshotRejected.

    A JSON number is unbounded: ``float(10**400)`` raises OverflowError, and
    NaN / Infinity parse from Python's json. None of them is a value a snapshot
    carries, and each would otherwise escape as the wrong exception type or
    poison a comparison downstream (NaN compares unequal to everything)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotRejected(f"{what} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise SnapshotRejected(f"{what} is out of range") from exc
    if not math.isfinite(number):
        raise SnapshotRejected(f"{what} must be finite")
    return number


def _check_levels_bound(snapshot: ReputationSnapshot) -> None:
    """Refuse a snapshot whose level maps contradict its suspicion maps.

    Shared by both readers — snapshot_from_payload (wire) and load_snapshot
    (disk). The checksum covers the suspicion maps, not the level maps beside
    them — and the levels are what a consumer renders and alerts on. Integrity
    only transfers from one to the other if the levels are what the suspicions
    were derived FROM (LEVEL_SUSPICION): a snapshot that relabels a FLAGGED
    resource CLEAN while its checksummed suspicion still says 1.0 is a rewrite,
    and is refused like one. A legacy snapshot carrying no levels at all has
    nothing to contradict and passes. Level names are matched
    case-insensitively (the cycle wrote them lower-cased until 0.4.2).
    """
    for levels_name, values_name in (("resource_level", "resource_reputation"),
                                     ("container_level", "container_reputation")):
        levels = getattr(snapshot, levels_name)
        values = getattr(snapshot, values_name)
        if not isinstance(levels, dict) or not isinstance(values, dict):
            raise SnapshotRejected(f"`{levels_name}` / `{values_name}` must be objects")
        if not levels:
            continue
        for key, level in levels.items():
            if not isinstance(level, str) or level.upper() not in _LEVEL_NAMES:
                raise SnapshotRejected(
                    f"`{levels_name}[{key}]` = {level!r} is not a level in "
                    f"{sorted(_LEVEL_NAMES)}"
                )
            expected = LEVEL_SUSPICION[ReputationLevel[level.upper()]]
            value = values.get(key, 0.0)
            if _finite_number(value, f"`{values_name}[{key}]`") != expected:
                raise SnapshotRejected(
                    f"`{levels_name}[{key}]` = {level} contradicts "
                    f"`{values_name}[{key}]` = {value} "
                    f"(a {level} resource carries {expected})"
                )
        for key in values:
            if key not in levels:
                raise SnapshotRejected(
                    f"`{values_name}[{key}]` has no level in `{levels_name}`"
                )


def snapshot_from_payload(payload: object) -> ReputationSnapshot:
    """Rebuild a ReputationSnapshot that arrived over a wire.

    Forward-compatible in the same direction `_deserialize` is: a field added by
    a newer sentinel is dropped rather than raising — this reader never reads
    it, so it cannot change what the reader acts on.

    Raises SnapshotRejected — never another exception type — on anything that
    is not a snapshot: a bad shape, a negative or non-integer version, a
    non-finite generated_at, a reputation value outside the finite codomain a
    deterministic sentinel emits, a telemetry value that is not a finite
    number, verdict_facts that is not id → list of strings, a level name this
    library does not know, levels that contradict the suspicions, or a checksum
    that does not match the maps in the payload. The checksum is unkeyed: a
    match proves the maps were not corrupted, not who wrote them.
    """
    if not isinstance(payload, dict):
        raise SnapshotRejected("snapshot must be an object")
    known = {f.name for f in fields(ReputationSnapshot)}
    fetched = {k: v for k, v in payload.items() if k in known}
    # Coerce the number maps to float BEFORE anything reads them, the checksum
    # included. The checksum covers a canonical serialisation in which 1.0 is
    # written "1.0" — and a JSON round-trip does not preserve that.
    # JSON.parse("1.0") is the number 1, JSON.stringify writes "1", and Python
    # then parses an int; the maps are numerically identical and the checksum
    # does not match. Verifying against the sender's spelling would have made
    # this wire Python-to-Python only, and would have rejected a correct
    # snapshot for passing through a proxy that reformatted its JSON. The
    # coercion goes through _finite_number so an unbounded int (10**400) or a
    # NaN/Infinity is refused here as SnapshotRejected rather than escaping as
    # OverflowError or slipping past the codomain check as a NaN.
    for name in ("resource_reputation", "container_reputation",
                 "resource_score_telemetry", "container_score_telemetry"):
        got = fetched.get(name, {})
        if not isinstance(got, dict) or not all(isinstance(k, str) for k in got):
            raise SnapshotRejected(f"`{name}` must map string ids to numbers")
        fetched[name] = {
            k: _finite_number(v, f"`{name}[{k}]`") for k, v in got.items()
        }
    if "generated_at" in fetched:
        fetched["generated_at"] = _finite_number(fetched["generated_at"], "`generated_at`")
    try:
        snapshot = ReputationSnapshot(**fetched)
    except TypeError as exc:  # missing version / generated_at
        raise SnapshotRejected(f"not a snapshot: {exc}") from exc

    if not isinstance(snapshot.version, int) or isinstance(snapshot.version, bool):
        raise SnapshotRejected("`version` must be an integer")
    if snapshot.version < 0:
        raise SnapshotRejected("`version` must not be negative")

    for name in ("resource_reputation", "container_reputation"):
        for key, value in getattr(snapshot, name).items():
            # The codomain is finite by construction (predicates.LEVEL_SUSPICION).
            # Checking it here is what keeps core's detection_floor comparison
            # decidable for a consumer that did not compute these numbers: an
            # arbitrary float would reintroduce exactly the calibrated threshold
            # the deterministic verdict layer exists to remove.
            if value not in _SUSPICION_VALUES:
                raise SnapshotRejected(
                    f"`{name}[{key}]` = {value} is not one of "
                    f"{sorted(_SUSPICION_VALUES)} — a deterministic sentinel "
                    f"emits a finite codomain"
                )

    facts = snapshot.verdict_facts
    if not isinstance(facts, dict) or not all(
        isinstance(k, str) and isinstance(v, list) and all(isinstance(f, str) for f in v)
        for k, v in facts.items()
    ):
        raise SnapshotRejected("`verdict_facts` maps ids to a list of strings")

    # Level names are canonical UPPERCASE on the wire. The cycle wrote them
    # lower-cased (`lvl.name.lower()`) until 0.4.2, so a real cycle's snapshot
    # was refused here as "not a level" and no node could ever report one.
    # Accept either spelling from a sender, hand back the canonical one.
    normalised: dict[str, dict[str, str]] = {}
    for name in ("resource_level", "container_level"):
        got = getattr(snapshot, name)
        if not isinstance(got, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v.upper() in _LEVEL_NAMES
            for k, v in got.items()
        ):
            raise SnapshotRejected(
                f"`{name}` maps ids to a level in {sorted(_LEVEL_NAMES)}"
            )
        normalised[name] = {k: v.upper() for k, v in got.items()}
    snapshot = replace(
        snapshot,
        resource_level=normalised["resource_level"],
        container_level=normalised["container_level"],
    )

    if snapshot.checksum != snapshot.compute_checksum():
        raise SnapshotRejected(
            "checksum does not match the reputation maps in this payload"
        )
    _check_levels_bound(snapshot)
    return snapshot


def _serialize(snapshot: ReputationSnapshot) -> str:
    return json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"))


def _deserialize(text: str) -> tuple[ReputationSnapshot, dict]:
    """Parse a snapshot file into ``(snapshot, raw_mapping)``.

    Forward-compatible: unknown top-level keys are ignored rather than raising
    on a snapshot written by a newer sentinel that added a field. The raw
    mapping is returned alongside so the signature is verified over what is
    actually on disk (see _signature_payload) — the unknown field included."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("snapshot file is not a JSON object")
    known = {f.name for f in fields(ReputationSnapshot)}
    return ReputationSnapshot(**{k: v for k, v in data.items() if k in known}), data


def write_file_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so a reader or a crash never sees a partial file.

    Temp file in the SAME directory (os.replace is only atomic within one
    filesystem) → write → fsync → os.replace → fsync the directory. A crash at
    any point leaves either the old file intact or the new one complete, never a
    truncated one. The temp name carries the pid so two processes cannot collide
    on it; a temp left behind by a crash is harmless (never read) and is swept
    by atomic_swap's pruning for snapshot files.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    _fsync_dir(path.parent)


def _version_from_name(name: str) -> int | None:
    """N from ``snapshot_vN.json``, else None."""
    if not (name.startswith("snapshot_v") and name.endswith(".json")):
        return None
    try:
        return int(name[len("snapshot_v"):-len(".json")])
    except ValueError:
        return None


def live_snapshot_version(snapshot_dir: Path) -> int:
    """Version of the snapshot ``snapshot_current`` makes visible, 0 if none.

    POSIX: read from the symlink's TARGET NAME (snapshot_vN.json), so it works
    even when the target file is unreadable. Windows (or any regular file at
    that path): the ``version`` field of its content. Best effort — anything
    unparseable counts as 0, and callers combine this with the retained
    version files (latest_snapshot_version) rather than trusting it alone.
    """
    current = Path(snapshot_dir) / "snapshot_current"
    try:
        if current.is_symlink():
            version = _version_from_name(Path(os.readlink(current)).name)
            return version if version is not None else 0
        if current.is_file():
            version = json.loads(current.read_text(encoding="utf-8")).get("version", 0)
            if isinstance(version, int) and not isinstance(version, bool):
                return max(version, 0)
    except (OSError, ValueError, AttributeError):
        pass
    return 0


def latest_snapshot_version(snapshot_dir: Path) -> int:
    """Highest version this directory has ever made visible or retained.

    max(the live snapshot_current version, every snapshot_vN.json on disk). A
    writer that lost its own record of the version (state file missing,
    truncated, or rejected) resumes ABOVE this, so it never rewrites a retained
    snapshot_vN.json with different content under the same number. File names
    are trusted without reading the files: an attacker able to drop a
    high-numbered file can only make the writer SKIP versions, never repeat one.
    """
    snapshot_dir = Path(snapshot_dir)
    best = live_snapshot_version(snapshot_dir)
    try:
        for f in snapshot_dir.glob("snapshot_v*.json"):
            version = _version_from_name(f.name)
            if version is not None and version > best:
                best = version
    except OSError:
        pass
    return best


def atomic_swap(snapshot_dir: Path, new_snapshot: ReputationSnapshot) -> None:
    """
    Write snapshot atomically so axor-core never reads a partial write.

    Protocol (invariant A-5, A-16):
      0. Refuse a version <= the live one (SnapshotVersionRegression).
      1. Write the new version file via temp + fsync + rename — a live file is
         never rewritten in place.
      2. Verify checksum before making it visible.
      3. Create a new symlink alongside current (POSIX) or use os.replace (Windows).
      4. Atomic rename/replace of symlink, then fsync the directory (POSIX, best
         effort) so the swap itself survives a power loss.
      5. Prune old version files (keep SNAPSHOT_RETAIN_VERSIONS), stale
         snapshot_link_v* symlinks and temp files a crash left behind.

    On POSIX: os.rename over an existing symlink is atomic.
    On Windows: os.replace is used — atomic on same-volume (invariant A-16).
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 0. Versions only move forward. A reader orders snapshots by version and
    # nothing else, so publishing a lower (or equal) number is a rollback from
    # its point of view — and rewriting snapshot_vN.json with new content would
    # make one version name two different snapshots.
    live = live_snapshot_version(snapshot_dir)
    if new_snapshot.version <= live:
        raise SnapshotVersionRegression(
            f"refusing to publish snapshot version {new_snapshot.version}: "
            f"version {live} is already live"
        )

    # Sign whenever a key is configured, replacing any signature the caller
    # brought: the signature covers the full snapshot, so one computed before a
    # later dataclasses.replace() would be stale, and the writer holding the key
    # is the authority on what it publishes (M-2).
    key = _snapshot_key()
    if key is not None:
        new_snapshot = replace(
            new_snapshot, signature=new_snapshot.compute_signature(key)
        )

    version_file = snapshot_dir / f"snapshot_v{new_snapshot.version}.json"
    serialized = _serialize(new_snapshot)

    # 1. Write the version file atomically (temp + fsync + rename). A leftover
    # snapshot_vN.json with N > live (a crash before the link swap) is not live,
    # so replacing it is safe; the live one is protected by step 0.
    write_file_atomic(version_file, serialized)

    # 2. Verify checksum from in-memory bytes before making visible (invariant A-5).
    # Hashing the serialized bytes avoids a second file read (which could return stale
    # data on NFS) and ensures exactly what was written is what was verified.
    _verify_checksum_bytes(serialized.encode(), new_snapshot.checksum)

    current_link = snapshot_dir / "snapshot_current"

    if sys.platform == "win32":
        # Windows: os.replace is atomic on same-volume (invariant A-16)
        # Write to a temp file then replace — Windows cannot rename over a symlink
        temp_file = snapshot_dir / f"snapshot_v{new_snapshot.version}_current.json"
        with temp_file.open("w", encoding="utf-8") as fh:
            fh.write(serialized)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_file, current_link)
    else:
        # POSIX: symlink rename is atomic even over an existing symlink. A
        # same-named link left by a crash between symlink_to and rename would
        # make symlink_to raise FileExistsError; it was never live, so drop it.
        new_link = snapshot_dir / f"snapshot_link_v{new_snapshot.version}"
        if new_link.is_symlink():
            new_link.unlink()
        new_link.symlink_to(version_file.name)
        os.rename(new_link, current_link)
        _fsync_dir(snapshot_dir)

    # 5. Prune old version files — keep last SNAPSHOT_RETAIN_VERSIONS
    _prune_old_versions(snapshot_dir, new_snapshot.version, keep=SNAPSHOT_RETAIN_VERSIONS)


def load_snapshot(snapshot_dir: Path) -> ReputationSnapshot | None:
    """
    Load and verify the current snapshot from snapshot_dir.

    Returns None if no snapshot exists yet.
    On checksum failure, signature failure, a malformed file, or levels that
    contradict the suspicion maps: emits AuditIntegrityWarning and returns None
    (caller retains its previous snapshot — invariant A-5).

    With AXOR_SNAPSHOT_KEY set, the HMAC is verified over the full snapshot as
    stored (every field but checksum/signature), so version, generated_at,
    levels and facts are authenticated along with the maps. Without a key only
    the unkeyed checksum is checked — corruption detection, not tamper
    protection.

    ROLLBACK is not detectable here: this function has no memory, and an older
    snapshot_vN.json re-linked as current carries a perfectly valid signature.
    Rollback protection belongs to the stateful reader — see
    SnapshotIntentEnricher.reload, which refuses a version lower than the one it
    holds.
    """
    snapshot_dir = Path(snapshot_dir)
    current_link = snapshot_dir / "snapshot_current"
    if not current_link.exists():
        return None
    try:
        text = current_link.read_text(encoding="utf-8")
        snapshot, raw = _deserialize(text)
        if not isinstance(snapshot.version, int) or isinstance(snapshot.version, bool) \
                or snapshot.version < 0:
            raise SnapshotRejected("`version` must be a non-negative integer")
        expected = snapshot.compute_checksum()
        if not isinstance(snapshot.checksum, str) or not hmac.compare_digest(
            snapshot.checksum, expected
        ):
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
            expected_sig = _hmac_hex(key, _signature_payload(raw))
            if not isinstance(snapshot.signature, str) or not snapshot.signature \
                    or not hmac.compare_digest(snapshot.signature, expected_sig):
                warnings.warn(
                    "snapshot signature invalid or missing while "
                    f"{SNAPSHOT_KEY_ENV} is set — refusing to load (possible "
                    "tampering, or a snapshot signed by sentinel 0.4.2 or "
                    "earlier, whose signature covered only the reputation "
                    "maps); retaining previous snapshot",
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
        # The same binding the wire reader enforces: levels must be the ones
        # the (checksummed, and when keyed signed) suspicions derive from. Even
        # unkeyed this stops a relabel that leaves the suspicion maps alone —
        # the edit that would clear a FLAGGED resource on every screen while
        # the checksum still matched.
        _check_levels_bound(snapshot)
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


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename/replace inside it survives a power loss.

    POSIX only and best effort: the rename is already atomic for readers; this
    only makes it durable. Windows cannot open a directory for fsync, and some
    filesystems refuse it — neither is a reason to fail the write."""
    if sys.platform == "win32":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _prune_old_versions(snapshot_dir: Path, current_version: int, keep: int) -> None:
    """Remove version files older than (current_version - keep), plus debris a
    crash can leave: stale ``snapshot_link_v*`` symlinks (a crash between
    symlink_to and rename) and ``snapshot_v*.tmp`` temp files for versions at or
    below the one just published (a crash inside write_file_atomic).

    Never touches the live link — it is named ``snapshot_current``, never
    ``snapshot_link_v*`` — and never the file it points to (its version is
    ``current_version``, above the cutoff)."""
    cutoff = current_version - keep
    for f in snapshot_dir.glob("snapshot_v*.json"):
        version = _version_from_name(f.name)
        if version is not None and version <= cutoff:
            with contextlib.suppress(OSError):
                f.unlink(missing_ok=True)
    current_link = snapshot_dir / "snapshot_current"
    for f in snapshot_dir.glob("snapshot_link_v*"):
        if f == current_link or not f.is_symlink():
            continue
        with contextlib.suppress(OSError):
            f.unlink(missing_ok=True)
    for f in snapshot_dir.glob("snapshot_v*.tmp"):
        # snapshot_v{N}.json.{pid}.tmp — only a version already superseded, so
        # a concurrent writer's in-flight temp for a NEWER version is left alone.
        version = _version_from_name(f.name.split(".json.", 1)[0] + ".json")
        if version is not None and version <= current_version:
            with contextlib.suppress(OSError):
                f.unlink(missing_ok=True)

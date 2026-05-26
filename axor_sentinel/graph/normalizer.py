from __future__ import annotations

import os
import re
from urllib.parse import urlparse


# Confidence tiers
_CONFIDENCE_PROVIDER_ID = 1.0
_CONFIDENCE_PATH = 0.7
_CONFIDENCE_HEURISTIC = 0.4

# Normalization method names
METHOD_PROVIDER_ID = "provider_id"
METHOD_PATH = "path"
METHOD_HEURISTIC = "heuristic"

# Auth tokens and fragments to strip from paths/URLs
_STRIP_PARAMS = re.compile(r"[?#].*$")
_AUTH_PREFIX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://[^@]*/")  # strip scheme+host


def normalize_resource_id(resource_info: dict) -> tuple[str, str, float]:
    """
    Derive a canonical resource ID using best-effort three-tier normalization.

    Priority order (per spec §Resource ID normalization):
      1. Provider object ID (SharePoint driveItemId, OneDrive itemId, …)
         → normalization_method="provider_id", canonical_confidence=1.0
      2. Normalized path — strip protocol/host/auth tokens, resolve .., lowercase,
         strip trailing slash, prefix with service
         → normalization_method="path", canonical_confidence=0.7
      3. Heuristic — filename + size + last-modified fingerprint
         → normalization_method="heuristic", canonical_confidence=0.4

    Args:
        resource_info: dict with any combination of keys:
            provider_id  — authoritative object ID from the storage provider
            path         — file path or URL
            service      — datasource/service name (used as prefix for path tier)
            filename     — bare filename for heuristic tier
            size         — file size in bytes for heuristic tier
            last_modified — unix timestamp for heuristic tier

    Returns:
        (canonical_id, normalization_method, canonical_confidence)
    """
    # Tier 1: provider object ID
    provider_id = resource_info.get("provider_id", "")
    if provider_id:
        return str(provider_id), METHOD_PROVIDER_ID, _CONFIDENCE_PROVIDER_ID

    # Tier 2: normalized path
    path = resource_info.get("path", "")
    service = str(resource_info.get("service", "")).lower().strip()
    if path:
        normalized = _normalize_path(str(path))
        canonical = f"{service}:{normalized}" if service else normalized
        return canonical, METHOD_PATH, _CONFIDENCE_PATH

    # Tier 3: heuristic fingerprint
    filename = str(resource_info.get("filename", "")).strip()
    size = resource_info.get("size", "")
    last_modified = resource_info.get("last_modified", "")
    if filename:
        parts = [filename]
        if size != "" and size is not None:
            parts.append(str(size))
        if last_modified != "" and last_modified is not None:
            parts.append(str(last_modified))
        fingerprint = "|".join(parts)
        canonical = f"{service}:heuristic:{fingerprint}" if service else f"heuristic:{fingerprint}"
        return canonical, METHOD_HEURISTIC, _CONFIDENCE_HEURISTIC

    # Fallback: empty resource info — return empty string with lowest confidence
    return "", METHOD_HEURISTIC, _CONFIDENCE_HEURISTIC


def _normalize_path(raw: str) -> str:
    """
    Normalize a file path or URL to a stable canonical form.

    - Strip protocol, hostname, auth tokens
    - Resolve .. segments
    - Lowercase
    - Strip trailing slash
    """
    raw = raw.strip()

    # URL: strip scheme + host, keep path
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", raw):
        parsed = urlparse(raw)
        # Use path component; discard query, fragment, credentials
        raw = parsed.path or "/"

    # Strip query string / fragment from bare paths too
    raw = _STRIP_PARAMS.sub("", raw)

    # Resolve .. segments using os.path.normpath (no filesystem access)
    try:
        resolved = os.path.normpath(raw).replace("\\", "/")
    except Exception:
        resolved = raw

    # Lowercase and strip trailing slash (except root)
    resolved = resolved.lower()
    if len(resolved) > 1:
        resolved = resolved.rstrip("/")

    return resolved

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from axor_sentinel.graph.normalizer import normalize_resource_id
from axor_sentinel.sentinel.snapshot import ReputationSnapshot, load_snapshot

if TYPE_CHECKING:
    from axor_core.contracts.anomaly import NormalizedIntent  # type: ignore[import-untyped]
    from axor_core.contracts.intent import Intent  # type: ignore[import-untyped]

log = logging.getLogger("axor.sentinel.enricher")


class SnapshotIntentEnricher:
    """
    Implements axor-core's ReputationEnricher protocol.

    Reads the current reputation snapshot (pre-loaded — no Neo4j on hot path, A-6)
    and populates target_resource_reputation and target_container_reputation on
    NormalizedIntent via dataclasses.replace().

    Unknown resources return the original NormalizedIntent unchanged (score stays 0.0).
    Never raises — failures are logged and original intent returned (fail-safe).

    Usage:
        enricher = SnapshotIntentEnricher.from_dir(Path("~/.axor/sentinel/snapshots"))
        # or with a pre-loaded snapshot:
        enricher = SnapshotIntentEnricher(snapshot)
    """

    def __init__(self, snapshot: ReputationSnapshot | None = None) -> None:
        self._snapshot = snapshot

    @classmethod
    def from_dir(cls, snapshot_dir: Path) -> "SnapshotIntentEnricher":
        """Load snapshot from directory and return an enricher instance."""
        snapshot = load_snapshot(Path(snapshot_dir))
        return cls(snapshot)

    def reload(self, snapshot_dir: Path) -> None:
        """Reload the snapshot from disk. Call after each audit cycle."""
        self._snapshot = load_snapshot(Path(snapshot_dir))

    def enrich(
        self,
        normalized: "NormalizedIntent",
        intent: "Intent",
    ) -> "NormalizedIntent":
        """
        Return normalized with reputation fields populated from snapshot.

        Derives resource_id from intent args using graph/normalizer.py.
        Falls back to original NormalizedIntent if resource is unknown or
        if snapshot is not loaded.

        Must not query Neo4j — reads pre-loaded snapshot only (invariant A-6).
        Must not raise (invariant: fail-safe on hot path).
        """
        if self._snapshot is None:
            return normalized

        try:
            resource_id, container_id = self._derive_ids(intent)
            resource_rep = self._snapshot.resource_reputation.get(resource_id, 0.0)
            container_rep = self._snapshot.container_reputation.get(container_id, 0.0)

            if resource_rep == 0.0 and container_rep == 0.0:
                return normalized

            return dataclasses.replace(
                normalized,
                target_resource_reputation=resource_rep,
                target_container_reputation=container_rep,
            )
        except Exception as exc:
            log.debug("enricher failed (returning original): %s", exc)
            return normalized

    def _derive_ids(self, intent: "Intent") -> tuple[str, str]:
        """
        Derive (resource_id, container_id) from intent args.

        Resource ID derived via graph/normalizer.py priority order.
        Container ID derived from service + directory extracted from args.
        """
        args = intent.payload.get("args", {})
        tool = intent.payload.get("tool", "")

        resource_info = derive_resource_info(tool, args)
        resource_id, _, _ = normalize_resource_id(resource_info)

        # Container ID: service + directory of the resource
        container_id = _derive_container_id(
            resource_info.get("path", ""),
            resource_info.get("service", ""),
        )
        return resource_id, container_id


def derive_resource_info(tool: str, args: "Mapping[str, object]") -> dict:
    """
    Extract a normalizer-ready ``resource_info`` dict from a tool name + args.

    Shared by SnapshotIntentEnricher (hot-path intent enrichment) and
    CoreSessionSink (audit-cycle session mapping) so resource IDs are derived
    identically on both paths. The keys produced here are exactly the keys
    ``graph.normalizer.normalize_resource_id`` consumes
    (provider_id / path / service / filename / size / last_modified).
    """
    args = args or {}
    resource_info: dict = {}

    # Provider object IDs (SharePoint, OneDrive, etc.)
    for key in ("provider_id", "item_id", "drive_item_id", "object_id"):
        if key in args:
            resource_info["provider_id"] = args[key]
            break

    # Path — covers file tools, URL tools, MCP resource URIs
    path = (
        args.get("path")
        or args.get("file_path")
        or args.get("file")
        or args.get("url")
        or args.get("uri")
        or ""
    )
    if path:
        resource_info["path"] = str(path)

    # Service inference from tool name
    service = _infer_service(tool, args)
    if service:
        resource_info["service"] = service

    # Heuristic fields
    if "filename" in args:
        resource_info["filename"] = args["filename"]
    if "size" in args:
        resource_info["size"] = args["size"]
    if "last_modified" in args:
        resource_info["last_modified"] = args["last_modified"]

    return resource_info


def _infer_service(tool: str, args: "Mapping[str, object]") -> str:
    """Infer the service/datasource name from tool name or args."""
    t = tool.lower()
    if "sharepoint" in t or "sp_" in t:
        return "sharepoint"
    if "onedrive" in t or "od_" in t:
        return "onedrive"
    if "gmail" in t or "email" in t:
        return "email"
    if "slack" in t:
        return "slack"
    # MCP namespace from tool name pattern
    if "_" in t:
        return t.split("_")[0]
    return ""


def _derive_container_id(path: str, service: str) -> str:
    """Derive container ID as service + parent directory."""
    import os
    if not path:
        return service or ""
    # Strip query/fragment for URL paths
    import re
    path = re.sub(r"[?#].*$", "", path)
    parent = os.path.dirname(path)
    if not parent or parent == path:
        parent = "/"
    if service:
        return f"{service}:{parent}"
    return parent

"""Resource / container identity derivation from a tool call.

Shared by the hot-path enricher (SnapshotIntentEnricher) and the audit-path sink
(CoreSessionSink) so a resource gets the SAME id on both paths. It lives here, next
to ``graph.normalizer`` (which both paths already depend on), rather than as private
functions of the enricher that the sink reached across module boundaries.

The keys produced by ``derive_resource_info`` are exactly the keys
``graph.normalizer.normalize_resource_id`` consumes (provider_id / path / service /
filename / size / last_modified).
"""
from __future__ import annotations

import os
import re
from typing import Mapping


def derive_resource_info(tool: str, args: "Mapping[str, object]") -> dict:
    """Extract a normalizer-ready ``resource_info`` dict from a tool name + args."""
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


def derive_container_id(path: str, service: str) -> str:
    """Derive container ID as service + parent directory."""
    if not path:
        return service or ""
    # Strip query/fragment for URL paths
    path = re.sub(r"[?#].*$", "", path)
    parent = os.path.dirname(path)
    if not parent or parent == path:
        parent = "/"
    if service:
        return f"{service}:{parent}"
    return parent

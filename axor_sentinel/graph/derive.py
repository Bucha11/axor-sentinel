"""Resource / container identity derivation from a tool call.

Shared by the hot-path enricher (SnapshotIntentEnricher) and the audit-path sink
(CoreSessionSink) so a resource gets the SAME id on both paths. It lives here, next
to ``graph.normalizer`` (which both paths already depend on), rather than as private
functions of the enricher that the sink reached across module boundaries. Both
paths call :func:`derive_identity` — one function, so parity holds by construction
rather than by two call sites staying in step.

The keys produced by ``derive_resource_info`` are exactly the keys
``graph.normalizer.canonicalize`` consumes (provider_id / path / service /
filename / size / last_modified).

The identity must not depend on WHICH tool touched a resource: the only thing the
tool name contributes is an explicitly recognised provider (``KNOWN_PROVIDERS``),
matched on whole name tokens. Its verb (read / write / export / fs …) never reaches
the id — otherwise switching tools launders a resource's reputation.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from axor_sentinel.graph.normalizer import KNOWN_PROVIDERS, CanonicalResource, canonicalize

# Arg keys, in precedence order. First non-empty wins.
_PATH_KEYS: tuple[str, ...] = ("path", "file_path", "file", "url", "uri")
_PROVIDER_ID_KEYS: tuple[str, ...] = ("provider_id", "item_id", "drive_item_id", "object_id")

# Tool-name token boundaries: "_" (so "__" too), ".", "-", whitespace, "/" and ":"
# (MCP-style "server/tool", "server:tool"), plus a lower→upper camelCase edge so
# "sendEmail" → ["send", "email"] and "WebFetch" → ["web", "fetch"].
_TOKEN_SPLIT = re.compile(r"[_.\-\s/:]+")
_CAMEL_EDGE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tool_tokens(tool: str) -> tuple[str, ...]:
    """Lower-cased whole-word tokens of a tool name.

    ``mcp__sharepoint__get_item`` → ``("mcp", "sharepoint", "get", "item")``.
    Matching against these (never ``substring in name``) is what keeps ``wasp_tool``
    from reading as SharePoint and ``postgres_query`` from reading as an export.
    """
    spaced = _CAMEL_EDGE.sub("_", str(tool or ""))
    return tuple(t for t in _TOKEN_SPLIT.split(spaced.lower()) if t)


def infer_provider(tool: str) -> str:
    """The first recognised provider token in the tool name, or ``""``.

    First in NAME order, so ``onedrive_to_sharepoint_copy`` is deterministic.
    Two adjacent tokens are also tried joined, because camelCase splitting turns
    ``OneDriveRead`` into ``one`` / ``drive`` / ``read`` — still whole tokens, so
    ``wasp`` never becomes ``sp``. Unrecognised services return ``""`` — a namespace
    is never invented from a verb.
    """
    toks = tool_tokens(tool)
    for i, tok in enumerate(toks):
        if i + 1 < len(toks) and tok + toks[i + 1] in KNOWN_PROVIDERS:
            return tok + toks[i + 1]
        if tok in KNOWN_PROVIDERS:
            return tok
    return ""


def _first_present(args: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        val = args.get(key)
        if val is None or val == "" or isinstance(val, bool):
            continue
        return val
    return None


def derive_resource_info(tool: str, args: Mapping[str, object]) -> dict:
    """Extract a normalizer-ready ``resource_info`` dict from a tool name + args."""
    args = args if isinstance(args, Mapping) else {}
    resource_info: dict = {}

    path = _first_present(args, _PATH_KEYS)
    if path is not None:
        resource_info["path"] = str(path)

    # Collected even when a path is present; canonicalize() only falls back to it
    # when there is no path, and only under a recognised provider (an attacker's
    # object_id riding next to a real path must not replace it).
    provider_id = _first_present(args, _PROVIDER_ID_KEYS)
    if provider_id is not None:
        resource_info["provider_id"] = provider_id

    provider = infer_provider(tool)
    if provider:
        resource_info["service"] = provider

    # Heuristic fields
    for key in ("filename", "size", "last_modified"):
        if key in args:
            resource_info[key] = args[key]

    return resource_info


def derive_identity(tool: str, args: Mapping[str, object]) -> CanonicalResource | None:
    """Canonical (resource, container) identity of a tool call, or ``None``.

    ``None`` means the call names no resource (``bash``, ``send_email {"to": …}``):
    the caller records NO access. Emitting ``""`` instead made one shared node for
    every path-less call everywhere — flag it once and every such call crossed the
    floor.
    """
    return canonicalize(derive_resource_info(tool, args))

"""Resource identity — the canonical id a resource's reputation accumulates under.

Cross-session detection only works if ONE resource maps to ONE id. Two failure
modes, both attacker-reachable, drive every rule below:

* **Evasion (one resource, many ids).** If the id depends on anything the caller
  can vary without changing *what* is touched — the tool's verb, the encoding of
  the path, a redundant ``./`` — the attacker spreads its accesses over several
  nodes and none of them accumulates enough to flag.
* **Poisoning (many resources, one id).** If the id drops something that
  distinguishes resources — the URL host, the query — the attacker's traffic on
  ``evil.com/x`` raises (or launders) the reputation of ``corp.sharepoint.com/x``.

Id format (the namespace before the first ``:`` never depends on the tool VERB):

========================  ==================================================  ======
source                    id                                                  tier
========================  ==================================================  ======
local path                ``file:/data/secret.txt`` (``file:rel/x`` if rel.)  path
provider path             ``sharepoint:/sites/hr/salary.xlsx``                path
http(s) URL               ``url:corp.example.com/data/x?a=1&b=2``             path
other-scheme URL          ``url:s3://bucket/key``                             path
remote ``file://`` URL    ``file://fileserver/share/x``                       path
provider object id        ``sharepoint:item:42``                              provider_id
fingerprint               ``heuristic:name|size|mtime`` (or ``<prov>:heuristic:…``)  heuristic
========================  ==================================================  ======

Normalisation rules (all PURE / lexical — no filesystem, network or clock access,
because the hot-path enricher and the audit-path sink run on different hosts and
must compute byte-identical ids from identical inputs):

* Percent-decoding is applied ONCE, so ``/data/%73ecret.txt`` == ``/data/secret.txt``
  but a doubly-encoded ``%2573`` stays distinct (decoding to a fixed point would let
  a literal ``%`` in a name be re-interpreted arbitrarily many times).
* ``.``/``..`` and repeated ``/`` are resolved lexically; ``..`` never climbs above
  the root of an absolute path. Symlinks are NOT resolved: ``realpath`` touches the
  disk on the hot path, answers differently on the enricher's and the sink's hosts
  (breaking id parity) and is racy (the link can be repointed between check and
  use). A symlink alias is therefore a residual split — see architecture.md §10a.
* Only the URL scheme and host are lowercased (both are case-insensitive by RFC
  3986). Paths keep their case: POSIX filesystems and most URL paths are
  case-sensitive, so folding case would merge distinct resources (poisoning).
* Local paths keep ``#`` and ``?`` — both are legal filename characters, and
  truncating at them merged ``/a#1`` and ``/a#2`` into ``/a``. Only a URL has a
  fragment (dropped: it never reaches the server) and a query (kept, canonicalised).
* URL queries are kept — the query routinely IS the identity (``?id=1`` vs
  ``?id=2``) — as sorted, re-encoded ``key=value`` pairs so parameter order and
  encoding cannot split an id. The one exception is credential-bearing params
  (``_CREDENTIAL_PARAMS``): they rotate per request (splitting identity) and would
  otherwise be persisted into the graph and the snapshot, so they are dropped.
* URL userinfo (``user:pass@``) is dropped for the same reason; default ports
  (80/443) are dropped, other ports kept.
* A provider object id is used only when there is NO path/URL (a path the tool
  actually opens outranks an attacker-supplied ``object_id`` riding along in the
  args) and only under an explicitly recognised provider — a bare ``42`` would be a
  single node shared by every tool that happens to number its objects.

An empty id means "no resource" and is never a node: callers skip the access.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

# Confidence tiers
_CONFIDENCE_PROVIDER_ID = 1.0
_CONFIDENCE_PATH = 0.7
_CONFIDENCE_HEURISTIC = 0.4

# Normalization method names
METHOD_PROVIDER_ID = "provider_id"
METHOD_PATH = "path"
METHOD_HEURISTIC = "heuristic"

# Providers whose name may namespace an id. Matched as WHOLE tokens of the tool
# name (see graph.derive.tool_tokens), never as substrings: "sp_" used to match
# "wasp_tool" and "od_" matched "prod_read". Deliberately an allowlist — an
# unrecognised service must not invent a namespace (the old code took the tool's
# first "_" token, i.e. its verb, so read_file / write_file / fs_read landed the
# same file in three different nodes).
KNOWN_PROVIDERS: frozenset[str] = frozenset({
    "sharepoint", "onedrive", "gdrive", "gmail", "outlook", "slack", "teams",
    "dropbox", "box", "confluence", "jira", "notion", "github", "gitlab",
    "salesforce",
})

# Query params that carry credentials, not identity. Lower-cased exact names plus
# the signed-URL families by prefix. Kept narrow on purpose: every name dropped
# here is a name whose values can no longer tell two resources apart.
_CREDENTIAL_PARAMS: frozenset[str] = frozenset({
    "access_token", "id_token", "refresh_token", "token", "api_key", "apikey",
    "sig", "signature",
})
_CREDENTIAL_PARAM_PREFIXES: tuple[str, ...] = ("x-amz-", "x-goog-")

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")
_WEB_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
# RFC 3986 unreserved + sub-delims + ":@/" — everything else in a decoded URL path
# is re-encoded, so e.g. a decoded "%3F" stays "%3F" and can never be confused
# with the "?" that starts the kept query.
_URL_PATH_SAFE = "/:@!$&'()*+,;=-._~"


@dataclass(frozen=True)
class CanonicalResource:
    """A resource's canonical identity plus the container it belongs to.

    ``container_id`` is derived from the NORMALISED locator (never the raw arg),
    so ``/data/./x`` and ``/data//y`` share the ``file:/data`` container. A
    resource with no hierarchy (provider object id, fingerprint) is its own
    container: grouping every ``sharepoint:item:*`` under one ``sharepoint``
    container would make all of them adjacent and share one container
    reputation — a poisoning surface, not a signal.
    """
    resource_id: str
    container_id: str
    method: str
    confidence: float


def canonicalize(resource_info: dict) -> CanonicalResource | None:
    """
    Canonical identity for a ``resource_info`` dict, or ``None`` for "no resource".

    Priority (path first — see module docstring for why):
      1. ``path`` (file path or URL)            → method "path",        conf 0.7
      2. ``provider_id`` under a known provider → method "provider_id", conf 1.0
      3. ``filename`` (+ size, last_modified)   → method "heuristic",   conf 0.4

    ``service`` is honoured only when it is in ``KNOWN_PROVIDERS``; anything else is
    ignored rather than turned into a namespace.
    """
    service = str(resource_info.get("service", "") or "").lower().strip()
    provider = service if service in KNOWN_PROVIDERS else ""

    # Tier 1: path / URL. The tool opens this, so it outranks a provider id.
    path = resource_info.get("path", "")
    if path not in (None, ""):
        canon = _canonical_locator(str(path), provider)
        if canon is not None:
            rid, container = canon
            return CanonicalResource(rid, container, METHOD_PATH, _CONFIDENCE_PATH)

    # Tier 2: provider object id — only namespaced by a recognised provider.
    provider_id = resource_info.get("provider_id", "")
    if provider and provider_id not in (None, "") and not isinstance(provider_id, bool):
        pid = str(provider_id).strip()
        if pid:
            rid = f"{provider}:item:{pid}"
            return CanonicalResource(rid, rid, METHOD_PROVIDER_ID, _CONFIDENCE_PROVIDER_ID)

    # Tier 3: heuristic fingerprint.
    filename = str(resource_info.get("filename", "") or "").strip()
    if filename:
        parts = [filename]
        size = resource_info.get("size", "")
        last_modified = resource_info.get("last_modified", "")
        if size != "" and size is not None:
            parts.append(str(size))
        if last_modified != "" and last_modified is not None:
            parts.append(str(last_modified))
        fingerprint = "|".join(parts)
        ns = f"{provider}:heuristic" if provider else "heuristic"
        rid = f"{ns}:{fingerprint}"
        return CanonicalResource(rid, rid, METHOD_HEURISTIC, _CONFIDENCE_HEURISTIC)

    return None


def normalize_resource_id(resource_info: dict) -> tuple[str, str, float]:
    """
    ``(canonical_id, normalization_method, canonical_confidence)`` for a resource.

    Thin tuple view over :func:`canonicalize`. Returns ``""`` as the id when there is
    no resource — callers MUST skip such an access rather than record it: ``""`` as a
    node would be one reputation shared by every path-less call (bash, send_email…).
    """
    canon = canonicalize(resource_info)
    if canon is None:
        return "", METHOD_HEURISTIC, _CONFIDENCE_HEURISTIC
    return canon.resource_id, canon.method, canon.confidence


# ── locators ──────────────────────────────────────────────────────────────────

def _canonical_locator(raw: str, provider: str) -> tuple[str, str] | None:
    """(resource_id, container_id) for a path or URL, or None if it is blank."""
    raw = raw.strip()
    if not raw:
        return None
    if _URL_RE.match(raw):
        return _canonical_url(raw)
    # A non-URL path under a recognised provider lives in that provider's path
    # space (a SharePoint server-relative path is not a local file); otherwise it
    # is a local filesystem path — one ``file:`` namespace whatever the tool.
    ns = provider or "file"
    path = _lexical_path(unquote(raw))
    return f"{ns}:{path}", f"{ns}:{_parent(path)}"


def _canonical_url(raw: str) -> tuple[str, str]:
    try:
        parts = urlsplit(raw)
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").rstrip(".")   # hostname is already lowercased
        port = parts.port
    except ValueError:
        # Malformed netloc (bad port / IPv6). Still give it an id — skipping it
        # would let a deliberately malformed URL dodge reputation entirely.
        scheme, _, rest = raw.partition("://")
        rest = rest.split("#", 1)[0]
        rid = f"url:{scheme.lower()}://{rest}"
        return rid, rid

    if scheme == "file":
        path = _lexical_path(unquote(parts.path) or "/")
        if host in ("", "localhost"):
            # file:///data/x IS the local path /data/x — same node as a read_file.
            return f"file:{path}", f"file:{_parent(path)}"
        # Remote file URL: "file://" + host can never collide with a local
        # "file:/…" id, because lexical normalisation never leaves a leading "//".
        return f"file://{host}{path}", f"file://{host}{_parent(path)}"

    if ":" in host:                     # IPv6 literal — urlsplit stripped brackets
        host = f"[{host}]"
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        host = f"{host}:{port}"

    path = quote(_lexical_path(unquote(parts.path) or "/"), safe=_URL_PATH_SAFE)
    query = _canonical_query(parts.query)

    # http and https name the same resource; other schemes (s3, ftp, mcp…) keep
    # their scheme so s3://b/k and https://b/k stay distinct.
    prefix = "url:" if scheme in _WEB_SCHEMES else f"url:{scheme}://"
    base = f"{prefix}{host}"
    rid = f"{base}{path}" + (f"?{query}" if query else "")
    return rid, f"{base}{_parent(path)}"


def _canonical_query(query: str) -> str:
    """Sorted, re-encoded query minus credential params ("" when empty)."""
    if not query:
        return ""
    pairs = [
        (k, v)
        for k, v in parse_qsl(query, keep_blank_values=True)
        if not _is_credential_param(k)
    ]
    pairs.sort()
    return urlencode(pairs, quote_via=quote, safe="")


def _is_credential_param(key: str) -> bool:
    k = key.lower()
    return k in _CREDENTIAL_PARAMS or k.startswith(_CREDENTIAL_PARAM_PREFIXES)


def _lexical_path(path: str) -> str:
    """
    Resolve ``.``, ``..`` and repeated ``/`` without touching the filesystem.

    Unlike ``posixpath.normpath`` this also collapses a leading ``//`` (which POSIX
    leaves implementation-defined) so ``//data/x`` cannot split from ``/data/x``.
    Trailing slashes are dropped (``/data/dir/`` == ``/data/dir``). ``..`` at the
    root of an absolute path is discarded; in a relative path it is kept, since the
    base directory is unknown here.
    """
    absolute = path.startswith("/")
    out: list[str] = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if out and out[-1] != "..":
                out.pop()
            elif not absolute:
                out.append("..")
            continue
        out.append(seg)
    joined = "/".join(out)
    if absolute:
        return "/" + joined
    return joined or "."


def _parent(path: str) -> str:
    """Lexical parent of an already-normalised path."""
    if path in ("/", "."):
        return path
    head, sep, _ = path.rpartition("/")
    if not sep:
        return "."          # relative single segment: "x" → "."
    return head or "/"      # "/x" → "/"

"""
Resource identity — the attack-shaped cases, and hot-path / audit-path parity.

Cross-session detection is only as good as "one resource ⇒ one id":

* evasion  — one resource under several ids (tool verb in the namespace, encoding,
  ``./`` / ``//``, an ``object_id`` overriding the real path) spreads its
  accesses so none accumulates;
* poisoning — several resources under one id (URL host / query dropped, ``""`` for
  every path-less call) lets traffic on one raise or launder another.

Every row below is a reported case. The parity test pins that the enricher (hot
path) and CoreSessionSink (audit path) derive byte-identical ids from the same
tool call — the snapshot the cycle writes is looked up by the enricher's id.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from axor_sentinel.graph.construct import _adjacency_pairs, upsert_graph
from axor_sentinel.graph.derive import derive_identity, infer_provider, tool_tokens
from axor_sentinel.graph.model import SignalType
from axor_sentinel.integration.core_sink import CoreSessionSink
from axor_sentinel.integration.intent_enricher import SnapshotIntentEnricher
from axor_sentinel.sentinel.cycle import ResourceAccess, SessionSummary
from axor_sentinel.sentinel.evidence import evidence_from_session
from axor_sentinel.sentinel.snapshot import ReputationSnapshot


def _rid(tool: str, args: dict) -> str | None:
    ident = derive_identity(tool, args)
    return None if ident is None else ident.resource_id


def _cid(tool: str, args: dict) -> str | None:
    ident = derive_identity(tool, args)
    return None if ident is None else ident.container_id


# ── 1. identity does not depend on the tool verb ────────────────────────────────

@pytest.mark.parametrize("tool,args", [
    ("read_file", {"path": "/data/secret.txt"}),
    ("write_file", {"path": "/data/secret.txt"}),
    ("export_file", {"path": "/data/secret.txt"}),
    ("fs_read", {"path": "/data/secret.txt"}),
    ("mcp__fs__read_file", {"path": "/data/secret.txt"}),
    ("Read", {"file_path": "/data/secret.txt"}),
    ("fetch", {"url": "file:///data/secret.txt"}),
    ("wasp_tool", {"path": "/data/secret.txt"}),      # "sp_" is not SharePoint
    ("prod_read", {"path": "/data/secret.txt"}),      # "od_" is not OneDrive
    ("read_email", {"path": "/data/secret.txt"}),     # "email" is not a namespace
])
def test_same_local_file_same_id_whatever_the_tool(tool, args):
    assert _rid(tool, args) == "file:/data/secret.txt"
    assert _cid(tool, args) == "file:/data"


@pytest.mark.parametrize("tool,expected", [
    ("mcp__sharepoint__get_item", "sharepoint"),
    ("sharepoint.read", "sharepoint"),
    ("OneDriveRead", "onedrive"),          # camelCase one+drive, joined
    ("SharePointGet", "sharepoint"),
    ("onedrive-download", "onedrive"),
    ("gmail_read", "gmail"),
    ("wasp_tool", ""),
    ("prod_read", ""),
    ("fs_read", ""),
    ("read_email", ""),
    ("onedrive_to_sharepoint_copy", "onedrive"),   # first in name order
])
def test_provider_matched_on_whole_tokens(tool, expected):
    assert infer_provider(tool) == expected


def test_tool_tokens_split_rules():
    assert tool_tokens("mcp__fs__read_file") == ("mcp", "fs", "read", "file")
    assert tool_tokens("a.b-c_d") == ("a", "b", "c", "d")
    assert tool_tokens("sendEmail") == ("send", "email")


# ── 2. URLs keep host and query; local names keep # and ? ───────────────────────

@pytest.mark.parametrize("a,b", [
    ("https://evil.com/data/secret.txt", "https://corp.sharepoint.com/data/secret.txt"),
    ("https://h.com/doc?id=1", "https://h.com/doc?id=2"),
    ("https://h.com/Doc", "https://h.com/doc"),                 # URL path case kept
    ("https://h.com:8443/x", "https://h.com/x"),                # non-default port kept
    ("s3://bucket/key", "https://bucket/key"),                  # scheme ≠ web
    ("https://h.com/a%3Fb", "https://h.com/a?b="),              # encoded ? ≠ query
])
def test_distinct_urls_stay_distinct(a, b):
    assert _rid("fetch", {"url": a}) != _rid("fetch", {"url": b})


@pytest.mark.parametrize("a,b", [
    ("https://H.COM/x", "https://h.com/x"),                     # host case
    ("https://h.com:443/x", "https://h.com/x"),                 # default port
    ("http://h.com/x", "https://h.com/x"),                      # http ≡ https
    ("https://h.com/x?b=2&a=1", "https://h.com/x?a=1&b=2"),     # param order
    ("https://h.com/x#frag", "https://h.com/x"),                # fragment
    ("https://user:pw@h.com/x", "https://h.com/x"),             # userinfo
    ("https://h.com/x?id=1&token=abc", "https://h.com/x?id=1&token=xyz"),  # creds
    ("https://h.com/a/./b/../%63", "https://h.com/a/c"),        # lexical + %-decode
    ("https://h.com/x/", "https://h.com/x"),                    # trailing slash
])
def test_equivalent_urls_share_one_id(a, b):
    assert _rid("fetch", {"url": a}) == _rid("fetch", {"url": b})


def test_url_id_format():
    rid = _rid("fetch", {"url": "https://Corp.Example.com/Data/x.txt?b=2&a=1#f"})
    assert rid == "url:corp.example.com/Data/x.txt?a=1&b=2"
    assert _cid("fetch", {"url": "https://corp.example.com/Data/x.txt?a=1"}) == (
        "url:corp.example.com/Data"
    )


@pytest.mark.parametrize("a,b", [
    ("/data/report#1.txt", "/data/report#2.txt"),
    ("/data/what?.txt", "/data/what"),
    ("/data/Secret.txt", "/data/secret.txt"),                   # case-sensitive FS
])
def test_distinct_local_names_stay_distinct(a, b):
    assert _rid("read_file", {"path": a}) != _rid("read_file", {"path": b})


@pytest.mark.parametrize("variant", [
    "/data/%73ecret.txt",           # percent-encoded
    "/data//secret.txt",            # doubled slash
    "//data/secret.txt",            # leading double slash
    "/data/./secret.txt",           # dot segment
    "/data/x/../secret.txt",        # dot-dot segment
    "/../data/secret.txt",          # dot-dot above root
    "  /data/secret.txt  ",         # surrounding whitespace
])
def test_local_path_aliases_collapse(variant):
    assert _rid("read_file", {"path": variant}) == "file:/data/secret.txt"


def test_percent_decoding_applied_once_only():
    # %2573 → %73 (one decode), not → s: a literal "%" must not be re-read forever.
    assert _rid("read_file", {"path": "/data/%2573ecret.txt"}) == "file:/data/%73ecret.txt"


# ── 3. an object_id riding along cannot override the path ──────────────────────

def test_object_id_does_not_override_path():
    args = {"path": "/data/secret.txt", "object_id": "42"}
    assert _rid("read_file", args) == "file:/data/secret.txt"
    assert _rid("sharepoint_read", {"url": "https://c.sharepoint.com/x", "item_id": "42"}) == (
        "url:c.sharepoint.com/x"
    )


def test_provider_id_needs_recognised_provider():
    assert _rid("sharepoint_get_item", {"item_id": "42"}) == "sharepoint:item:42"
    assert _rid("onedrive_get", {"drive_item_id": "42"}) == "onedrive:item:42"
    # Unrecognised provider: a bare "42" would be one node for every tool that
    # numbers objects — no id at all rather than a poisonable shared one.
    assert _rid("read_file", {"object_id": "42"}) is None
    assert _rid("wasp_tool", {"item_id": "42"}) is None


# ── 4. no "" resource / container, anywhere ────────────────────────────────────

@pytest.mark.parametrize("tool,args", [
    ("bash", {"command": "cat /etc/passwd"}),
    ("send_email", {"to": "x@y.z", "body": "hi"}),
    ("read_file", {"path": ""}),
    ("read_file", {"path": "   "}),
    ("read_file", {}),
    ("read_file", None),
])
def test_pathless_call_has_no_identity(tool, args):
    assert derive_identity(tool, args) is None


def test_evidence_skips_empty_resource_id():
    accesses = [
        ResourceAccess("", "", 0.7, SignalType.READ),
        ResourceAccess("file:/a", "file:/", 0.7, SignalType.READ),
    ]
    pairs = evidence_from_session("o", "s", 0.0, True, accesses)
    assert [rid for rid, _ in pairs] == ["file:/a"]


def test_construct_skips_empty_ids():
    class _Rec:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def run(self, query, **params):
            self.calls.append((query, params))

    rec = _Rec()
    s = SessionSummary(
        session_id="s1", agent_id="a", started_at=0.0, had_taint=True,
        had_export_attempt=False, had_failed_export=False, had_escalation=False,
        accessed_resources=[
            ResourceAccess("", "", 0.7, SignalType.READ),
            ResourceAccess("file:/a", "file:/", 0.7, SignalType.READ),
        ],
    )
    upsert_graph(rec, [s], {}, {"": ["", "x"], "file:/": ["file:/a", "", "file:/b"]},
                 flag_threshold=0.7)
    access_params = [p for q, p in rec.calls if "accesses" in p]
    assert [a["resource_id"] for a in access_params[0]["accesses"]] == ["file:/a"]
    assert _adjacency_pairs({"": ["x", "y"], "c": ["", "z", "w"]}) == [
        {"source": "z", "target": "w"}, {"source": "w", "target": "z"},
    ]


# ── 5. container comes from the NORMALISED id ─────────────────────────────────

@pytest.mark.parametrize("path", [
    "/data/secret.txt", "/data/./secret.txt", "/data//secret.txt",
    "/data/sub/../secret.txt", "/data/%73ecret.txt", "/data/other.txt",
])
def test_container_from_normalised_path(path):
    assert _cid("read_file", {"path": path}) == "file:/data"


def test_hierarchy_less_resource_is_its_own_container():
    # Grouping all sharepoint items under "sharepoint" would make every item
    # adjacent to every other and share one container reputation.
    assert _cid("sharepoint_get", {"item_id": "42"}) == "sharepoint:item:42"
    assert _cid("x", {"filename": "a.txt"}) == "heuristic:a.txt"


# ── hot path ≡ audit path ─────────────────────────────────────────────────────

_PARITY_TABLE: list[tuple[str, dict]] = [
    ("read_file", {"path": "/data/secret.txt"}),
    ("write_file", {"path": "/data/./secret.txt"}),
    ("mcp__fs__read_file", {"path": "/data/%73ecret.txt"}),
    ("Read", {"file_path": "/Data/Report#1.txt"}),
    ("fetch", {"url": "https://Corp.Example.com/x?b=2&a=1#f"}),
    ("web_fetch", {"uri": "http://h.com:8080/a/../b?token=t&id=9"}),
    ("sharepoint_get_item", {"item_id": "42"}),
    ("sharepoint_read", {"path": "/sites/hr/salary.xlsx", "object_id": "7"}),
    ("read_file", {"path": "/data/secret.txt", "object_id": "42"}),
    ("gdrive_find", {"filename": "budget.xlsx", "size": 10}),
    ("fetch", {"url": "file:///etc/passwd"}),
    ("fetch", {"url": "s3://bucket/key"}),
    ("read_file", {"path": "rel/./x"}),
]
_PATHLESS: list[tuple[str, dict]] = [
    ("bash", {"command": "ls"}),
    ("send_email", {"to": "a@b.c"}),
    ("wasp_tool", {"item_id": "42"}),
]


@dataclass(frozen=True)
class _FakeInvocation:
    tool: str
    args: dict
    executed: bool = True


@dataclass(frozen=True)
class _FakeRecord:
    session_id: str = "s"
    agent_id: str = "a"
    started_at: float = 0.0
    taint_active: bool = False
    taint_sources: tuple[str, ...] = ()
    event_kinds: tuple[str, ...] = ()
    tool_invocations: tuple[_FakeInvocation, ...] = field(default_factory=tuple)
    source_class: str = ""


def _intent(tool: str, args: dict) -> SimpleNamespace:
    # The enricher reads only intent.payload — no axor-core import needed.
    return SimpleNamespace(payload={"tool": tool, "args": args})


def _sink_ids(tool: str, args: dict) -> list[tuple[str, str]]:
    sink = CoreSessionSink()
    asyncio.run(sink.on_session_closed(
        _FakeRecord(tool_invocations=(_FakeInvocation(tool, args),))
    ))
    return [(a.resource_id, a.container_id) for a in sink.drain_pending()[0].accessed_resources]


@pytest.mark.parametrize("tool,args", _PARITY_TABLE + _PATHLESS)
def test_enricher_and_core_sink_derive_identical_ids(tool, args):
    hot = SnapshotIntentEnricher(None)._derive_ids(_intent(tool, args))
    audit = _sink_ids(tool, args)
    if hot is None:
        assert audit == []
    else:
        assert audit == [hot]
        assert all(hot)          # never "" on either side


@dataclass(frozen=True)
class _Normalized:
    target_resource_reputation: float = 0.0
    target_container_reputation: float = 0.0


@pytest.mark.parametrize("tool,args", _PARITY_TABLE)
def test_audit_id_is_found_by_the_enricher(tool, args):
    # End to end: score what the audit path recorded, read it on the hot path.
    [(rid, cid)] = _sink_ids(tool, args)
    snap = ReputationSnapshot(
        version=1, generated_at=time.time(),
        resource_reputation={rid: 1.0}, container_reputation={cid: 0.4},
    )
    out = SnapshotIntentEnricher(snap).enrich(_Normalized(), _intent(tool, args))
    assert out.target_resource_reputation == pytest.approx(1e-3)
    assert out.target_container_reputation == pytest.approx(0.6)


def test_flagged_pathless_node_cannot_reach_other_calls():
    # Even a snapshot that (from an old sentinel) carries a "" entry is never hit.
    snap = ReputationSnapshot(
        version=1, generated_at=time.time(),
        resource_reputation={"": 1.0}, container_reputation={"": 1.0},
    )
    enricher = SnapshotIntentEnricher(snap)
    base = _Normalized()
    for tool, args in _PATHLESS:
        assert enricher.enrich(base, _intent(tool, args)) is base

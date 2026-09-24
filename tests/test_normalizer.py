"""
Tests for graph/normalizer.py — three-tier resource ID normalization.

Covers the priority order (path > provider_id-under-a-known-provider > heuristic),
canonical_confidence values for each tier, path normalization edge cases, and the
empty-input fallback ("" = no resource; callers skip it). The attack-shaped cases
(verb laundering, host/query collisions, object_id override, …) live in
tests/test_resource_identity.py.
"""
from __future__ import annotations

import pytest

from axor_sentinel.graph.normalizer import (
    METHOD_HEURISTIC,
    METHOD_PATH,
    METHOD_PROVIDER_ID,
    normalize_resource_id,
)

_CONF_PROVIDER = 1.0
_CONF_PATH = 0.7
_CONF_HEURISTIC = 0.4


# ── Provider object ID (used only without a path, only under a known provider) ─

class TestProviderIdTier:
    def test_provider_id_namespaced_by_known_provider(self):
        rid, method, conf = normalize_resource_id(
            {"provider_id": "abc123", "service": "sharepoint"}
        )
        assert rid == "sharepoint:item:abc123"
        assert method == METHOD_PROVIDER_ID
        assert conf == _CONF_PROVIDER

    def test_bare_provider_id_without_known_provider_is_not_an_id(self):
        """A bare "abc123" would be one node shared by every tool numbering objects."""
        assert normalize_resource_id({"provider_id": "abc123"})[0] == ""
        assert normalize_resource_id(
            {"provider_id": "abc123", "service": "read"}
        )[0] == ""

    def test_path_preferred_over_provider_id(self):
        """The path the tool opens outranks an (attacker-suppliable) object id."""
        rid, method, conf = normalize_resource_id({
            "provider_id": "abc123",
            "path": "/some/path.txt",
            "service": "sharepoint",
        })
        assert method == METHOD_PATH
        assert rid == "sharepoint:/some/path.txt"
        assert conf == _CONF_PATH

    def test_provider_id_preferred_over_heuristic(self):
        rid, method, _ = normalize_resource_id({
            "provider_id": "xyz",
            "service": "onedrive",
            "filename": "report.xlsx",
            "size": 1024,
        })
        assert method == METHOD_PROVIDER_ID
        assert rid == "onedrive:item:xyz"

    def test_provider_id_stringified(self):
        """Non-string provider_id is coerced to str."""
        rid, method, _ = normalize_resource_id({"provider_id": 12345, "service": "box"})
        assert rid == "box:item:12345"
        assert method == METHOD_PROVIDER_ID


# ── Path / URL ────────────────────────────────────────────────────────────────

class TestPathTier:
    def test_local_path_in_file_namespace(self):
        rid, method, conf = normalize_resource_id({"path": "/data/report.csv"})
        assert rid == "file:/data/report.csv"
        assert method == METHOD_PATH
        assert conf == _CONF_PATH

    def test_path_case_preserved(self):
        """Paths are case-sensitive (POSIX, most URL paths): folding would merge."""
        rid, _, _ = normalize_resource_id({"path": "/Data/Report.CSV"})
        assert rid == "file:/Data/Report.CSV"

    def test_trailing_slash_stripped(self):
        rid_no_slash, _, _ = normalize_resource_id({"path": "/data/dir"})
        rid_with_slash, _, _ = normalize_resource_id({"path": "/data/dir/"})
        assert rid_no_slash == rid_with_slash == "file:/data/dir"

    def test_dotdot_resolved(self):
        rid, _, _ = normalize_resource_id({"path": "/data/sub/../report.csv"})
        assert rid == "file:/data/report.csv"

    def test_url_keeps_host(self):
        rid, method, _ = normalize_resource_id({
            "path": "https://SharePoint.Example.com/sites/team/report.docx"
        })
        assert method == METHOD_PATH
        assert rid == "url:sharepoint.example.com/sites/team/report.docx"

    def test_url_fragment_dropped_query_kept(self):
        rid, _, _ = normalize_resource_id({"path": "https://h.com/f?id=7#anchor"})
        assert rid == "url:h.com/f?id=7"

    def test_url_credential_params_dropped(self):
        rid, _, _ = normalize_resource_id({"path": "https://h.com/f?token=secret&id=7"})
        assert "secret" not in rid
        assert rid == "url:h.com/f?id=7"

    def test_local_path_keeps_hash_and_question_mark(self):
        rid, _, _ = normalize_resource_id({"path": "/file.txt?token=x#anchor"})
        assert rid == "file:/file.txt?token=x#anchor"

    def test_known_provider_namespaces_non_url_path(self):
        rid, _, _ = normalize_resource_id({
            "path": "/sites/hr/salary.xlsx",
            "service": "sharepoint",
        })
        assert rid == "sharepoint:/sites/hr/salary.xlsx"

    def test_service_lowercased(self):
        rid, _, _ = normalize_resource_id({"path": "/report.csv", "service": "OneDrive"})
        assert rid == "onedrive:/report.csv"

    def test_unknown_service_does_not_namespace(self):
        rid, _, _ = normalize_resource_id({"path": "/report.csv", "service": "write"})
        assert rid == "file:/report.csv"

    def test_no_filesystem_access(self, tmp_path):
        """Lexical only: a symlink is NOT resolved (hot path must not touch disk,
        and the enricher's and the sink's hosts would disagree)."""
        import os

        target = tmp_path / "real.txt"
        target.write_text("x")
        link = tmp_path / "alias.txt"
        os.symlink(target, link)
        rid, _, _ = normalize_resource_id({"path": str(link)})
        assert rid == f"file:{link}"


# ── Tier 3: heuristic fingerprint ────────────────────────────────────────────

class TestHeuristicTier:
    def test_returns_heuristic_when_no_provider_or_path(self):
        rid, method, conf = normalize_resource_id({
            "filename": "budget.xlsx",
            "size": 2048,
            "last_modified": 1700000000.0,
        })
        assert method == METHOD_HEURISTIC
        assert conf == _CONF_HEURISTIC
        assert "budget.xlsx" in rid

    def test_heuristic_includes_size_and_mtime(self):
        rid, _, _ = normalize_resource_id({
            "filename": "report.pdf",
            "size": 1024,
            "last_modified": 9999,
        })
        assert "1024" in rid
        assert "9999" in rid

    def test_heuristic_filename_only(self):
        """Heuristic with only filename — still returns a canonical ID."""
        rid, method, conf = normalize_resource_id({"filename": "readme.txt"})
        assert method == METHOD_HEURISTIC
        assert "readme.txt" in rid

    def test_service_prefix_on_heuristic(self):
        rid, _, _ = normalize_resource_id({
            "filename": "notes.txt",
            "service": "gdrive",
        })
        assert rid == "gdrive:heuristic:notes.txt"

    def test_unknown_service_not_prefixed_on_heuristic(self):
        rid, _, _ = normalize_resource_id({"filename": "notes.txt", "service": "fs"})
        assert rid == "heuristic:notes.txt"


# ── Fallback: empty input ─────────────────────────────────────────────────────

class TestEmptyFallback:
    def test_empty_dict_returns_empty_id(self):
        rid, method, conf = normalize_resource_id({})
        assert rid == ""
        assert method == METHOD_HEURISTIC
        assert conf == _CONF_HEURISTIC

    def test_confidence_ordering(self):
        """Tier confidence values obey provider > path > heuristic."""
        assert _CONF_PROVIDER > _CONF_PATH > _CONF_HEURISTIC


# ── Parametrized path edge cases ──────────────────────────────────────────────

@pytest.mark.parametrize("raw_path,expected", [
    ("/Data/Sub/../File.TXT", "file:/Data/File.TXT"),
    ("https://host.com/path/to/doc.pdf", "url:host.com/path/to/doc.pdf"),
    ("/trailing/slash/", "file:/trailing/slash"),      # trailing slash stripped
    ("/UPPER/LOWER", "file:/UPPER/LOWER"),             # case preserved
    ("/", "file:/"),                                    # root stays root
    ("/../../etc/passwd", "file:/etc/passwd"),          # .. cannot climb above /
    ("rel/./x/../y", "file:rel/y"),                     # relative stays relative
])
def test_path_normalization_parametrized(raw_path: str, expected: str) -> None:
    rid, method, _ = normalize_resource_id({"path": raw_path})
    assert method == METHOD_PATH
    assert rid == expected

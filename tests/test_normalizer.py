"""
Tests for graph/normalizer.py — three-tier resource ID normalization.

Covers the priority order (provider_id > path > heuristic), canonical_confidence
values for each tier, path normalization edge cases, and the empty-input fallback.
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


# ── Tier 1: provider object ID ────────────────────────────────────────────────

class TestProviderIdTier:
    def test_returns_provider_id_when_present(self):
        rid, method, conf = normalize_resource_id({"provider_id": "abc123"})
        assert rid == "abc123"
        assert method == METHOD_PROVIDER_ID
        assert conf == _CONF_PROVIDER

    def test_provider_id_preferred_over_path(self):
        """provider_id takes priority even when path is also present."""
        rid, method, conf = normalize_resource_id({
            "provider_id": "abc123",
            "path": "/some/path.txt",
        })
        assert method == METHOD_PROVIDER_ID
        assert rid == "abc123"

    def test_provider_id_preferred_over_heuristic(self):
        rid, method, _ = normalize_resource_id({
            "provider_id": "sp:driveItem:xyz",
            "filename": "report.xlsx",
            "size": 1024,
        })
        assert method == METHOD_PROVIDER_ID

    def test_provider_id_stringified(self):
        """Non-string provider_id is coerced to str."""
        rid, method, _ = normalize_resource_id({"provider_id": 12345})
        assert rid == "12345"
        assert method == METHOD_PROVIDER_ID


# ── Tier 2: normalized path ───────────────────────────────────────────────────

class TestPathTier:
    def test_returns_path_when_no_provider_id(self):
        rid, method, conf = normalize_resource_id({"path": "/data/report.csv"})
        assert method == METHOD_PATH
        assert conf == _CONF_PATH
        assert "report.csv" in rid

    def test_path_lowercased(self):
        rid, _, _ = normalize_resource_id({"path": "/Data/Report.CSV"})
        assert rid == rid.lower()

    def test_trailing_slash_stripped(self):
        rid_no_slash, _, _ = normalize_resource_id({"path": "/data/dir"})
        rid_with_slash, _, _ = normalize_resource_id({"path": "/data/dir/"})
        assert rid_no_slash == rid_with_slash

    def test_dotdot_resolved(self):
        rid, _, _ = normalize_resource_id({"path": "/data/sub/../report.csv"})
        assert ".." not in rid
        assert "report.csv" in rid

    def test_url_scheme_and_host_stripped(self):
        rid, method, _ = normalize_resource_id({
            "path": "https://sharepoint.example.com/sites/team/report.docx"
        })
        assert method == METHOD_PATH
        assert "sharepoint.example.com" not in rid
        assert "report.docx" in rid

    def test_url_query_and_fragment_stripped(self):
        rid, _, _ = normalize_resource_id({"path": "/file.txt?token=secret#anchor"})
        assert "token" not in rid
        assert "anchor" not in rid

    def test_service_prefix_added(self):
        rid, _, _ = normalize_resource_id({
            "path": "/sites/hr/salary.xlsx",
            "service": "sharepoint",
        })
        assert rid.startswith("sharepoint:")

    def test_service_lowercased(self):
        rid, _, _ = normalize_resource_id({
            "path": "/report.csv",
            "service": "OneDrive",
        })
        assert rid.startswith("onedrive:")


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
        assert rid.startswith("gdrive:heuristic:")


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

@pytest.mark.parametrize("raw_path,expected_fragment", [
    ("/Data/Sub/../File.TXT", "file.txt"),
    ("https://host.com/path/to/doc.pdf", "doc.pdf"),
    ("/trailing/slash/", "slash"),      # trailing slash stripped, not root
    ("/UPPER/LOWER", "upper/lower"),
])
def test_path_normalization_parametrized(raw_path: str, expected_fragment: str) -> None:
    rid, method, _ = normalize_resource_id({"path": raw_path})
    assert method == METHOD_PATH
    assert expected_fragment in rid

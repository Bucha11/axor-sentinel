"""Tests for the axor-core compatibility check (Phase 6)."""

from __future__ import annotations

import pytest

from axor_sentinel.compatibility import (
    MAX_AXOR_CORE,
    MIN_AXOR_CORE,
    _parse,
    check_axor_core_version,
)


def test_installed_core_is_compatible():
    # The pinned axor-core in this monorepo must be within the declared range.
    assert check_axor_core_version() is True


def test_parse_handles_suffixes():
    assert _parse("0.7.1") == (0, 7, 1)
    assert _parse("0.7.1+local") == (0, 7, 1)
    assert _parse("0.7") == (0, 7, 0)


def test_skew_warns_then_raises(monkeypatch):
    import axor_sentinel.compatibility as compat
    # Force the installed version below the minimum by stubbing axor_core.
    import axor_core
    monkeypatch.setattr(axor_core, "__version__", "0.1.0", raising=False)

    with pytest.warns(RuntimeWarning):
        assert compat.check_axor_core_version() is False
    with pytest.raises(RuntimeError):
        compat.check_axor_core_version(raise_on_mismatch=True)


def test_range_is_sane():
    assert MIN_AXOR_CORE < MAX_AXOR_CORE

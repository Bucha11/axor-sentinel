"""SnapshotIntentEnricher.reload only ever moves FORWARD.

load_snapshot returns None for every failure (checksum, signature, parse, level
binding, missing link), and reload used to assign that None — switching the
node's reputation off exactly when the file on disk was corrupted or tampered
with. And load_snapshot has no memory, so an older, validly-signed snapshot
re-linked as current loaded fine: a rollback only the stateful reader can see.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings

import pytest

from axor_sentinel.integration.intent_enricher import SnapshotIntentEnricher
from axor_sentinel.sentinel.snapshot import (
    SNAPSHOT_KEY_ENV,
    ReputationSnapshot,
    atomic_swap,
)

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink layout"),
    # every unkeyed load warns UNAUTHENTICATED; that is not what these test
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


def _snap(version: int, suspicion: float = 1.0) -> ReputationSnapshot:
    return ReputationSnapshot(
        version=version,
        generated_at=time.time(),
        resource_reputation={"r1": suspicion},
        container_reputation={},
    ).with_checksum()


@pytest.fixture(autouse=True)
def _keyed(monkeypatch):
    monkeypatch.setenv(SNAPSHOT_KEY_ENV, "enricher-test-key")


def _held(enricher: SnapshotIntentEnricher) -> ReputationSnapshot | None:
    return enricher._snapshot


def test_reload_takes_a_newer_snapshot(tmp_path):
    atomic_swap(tmp_path, _snap(1))
    enricher = SnapshotIntentEnricher.from_dir(tmp_path)
    atomic_swap(tmp_path, _snap(2, 0.4))
    enricher.reload(tmp_path)
    assert _held(enricher).version == 2
    assert _held(enricher).resource_reputation["r1"] == 0.4


def test_a_failed_load_keeps_the_previous_snapshot(tmp_path, caplog):
    atomic_swap(tmp_path, _snap(1))
    enricher = SnapshotIntentEnricher.from_dir(tmp_path)

    # a tampered v2 (signature no longer matches) fails to load
    atomic_swap(tmp_path, _snap(2))
    live = tmp_path / "snapshot_v2.json"
    data = json.loads(live.read_text())
    data["generated_at"] = 0.0
    live.unlink()
    live.write_text(json.dumps(data))

    with caplog.at_level(logging.WARNING, logger="axor.sentinel.enricher"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enricher.reload(tmp_path)
    assert _held(enricher) is not None and _held(enricher).version == 1
    assert "keeping version 1" in caplog.text


@pytest.mark.parametrize("damage", ["truncate", "unlink_link", "unlink_target"])
def test_corrupt_or_missing_files_do_not_switch_reputation_off(tmp_path, damage):
    atomic_swap(tmp_path, _snap(1))
    enricher = SnapshotIntentEnricher.from_dir(tmp_path)
    if damage == "truncate":
        (tmp_path / "snapshot_v1.json").write_text('{"version": 1, "resou')
    elif damage == "unlink_link":
        (tmp_path / "snapshot_current").unlink()
    else:
        (tmp_path / "snapshot_v1.json").unlink()  # dangling link
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enricher.reload(tmp_path)
    assert _held(enricher) is not None and _held(enricher).version == 1


def test_a_rolled_back_link_is_refused(tmp_path, caplog):
    """An older snapshot_vN.json carries a perfectly valid signature; re-linking
    snapshot_current to it passes load_snapshot. The enricher refuses it."""
    atomic_swap(tmp_path, _snap(1, 0.0))   # v1: resource clean
    atomic_swap(tmp_path, _snap(2, 1.0))   # v2: resource flagged
    enricher = SnapshotIntentEnricher.from_dir(tmp_path)
    assert _held(enricher).version == 2

    current = tmp_path / "snapshot_current"
    current.unlink()
    current.symlink_to("snapshot_v1.json")

    with caplog.at_level(logging.WARNING, logger="axor.sentinel.enricher"):
        enricher.reload(tmp_path)
    assert _held(enricher).version == 2
    assert _held(enricher).resource_reputation["r1"] == 1.0
    assert "rollback" in caplog.text


def test_the_same_version_is_a_no_op(tmp_path):
    atomic_swap(tmp_path, _snap(1))
    enricher = SnapshotIntentEnricher.from_dir(tmp_path)
    before = _held(enricher)
    enricher.reload(tmp_path)
    assert _held(enricher) is before


def test_reload_from_empty_still_loads_the_first_snapshot(tmp_path):
    enricher = SnapshotIntentEnricher.from_dir(tmp_path)
    assert _held(enricher) is None
    enricher.reload(tmp_path)          # nothing there yet: still None, no error
    assert _held(enricher) is None
    atomic_swap(tmp_path, _snap(3))
    enricher.reload(tmp_path)
    assert _held(enricher).version == 3

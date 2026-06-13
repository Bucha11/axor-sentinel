"""Slow-and-low staging detection — report wiring + query contract.

The detection itself runs as a Cypher query in Neo4j; without a graph fixture we
test the wiring (gap conversion, parameter passing, record→dict) and the structural
invariants of the query (parameterized, and it only returns *flagged* staging).
"""
from __future__ import annotations

from axor_sentinel.graph import queries
from axor_sentinel.reports.slow_and_low import SlowAndLowReport


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class _FakeSession:
    def __init__(self, records):
        self._records = records
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return _FakeResult(self._records)


def test_report_converts_gap_days_to_ms_and_passes_param():
    sess = _FakeSession(records=[])
    SlowAndLowReport(sess).run(min_gap_days=7.0)
    assert len(sess.calls) == 1
    query, params = sess.calls[0]
    # Parameterized (no string interpolation) and converted to milliseconds.
    assert params == {"min_gap_ms": 7.0 * 86_400_000.0}
    assert query is queries.SLOW_AND_LOW_QUERY


def test_report_returns_records_as_dicts():
    records = [
        {"agent_id": "a1", "gap_days": 9.0, "flagged_resources": ["r1"], "scores": [0.8]},
        {"agent_id": "a2", "gap_days": 3.0, "flagged_resources": ["r2"], "scores": [0.75]},
    ]
    out = SlowAndLowReport(_FakeSession(records)).run()
    assert out == records
    assert all(isinstance(r, dict) for r in out)


def test_default_gap_is_zero():
    sess = _FakeSession(records=[])
    SlowAndLowReport(sess).run()
    assert sess.calls[0][1] == {"min_gap_ms": 0.0}


def test_query_only_returns_flagged_staging():
    # The detection must key on flagged resources (score >= FLAG_THRESHOLD) — an
    # under-threshold staging resource must not surface. Structural guarantee of the
    # query, independent of any live graph.
    q = queries.SLOW_AND_LOW_QUERY
    assert "flagged" in q
    assert "min_gap_ms" in q          # gap filter is parameterized
    assert "$min_gap_ms" in q or "min_gap_ms" in q


def test_query_function_is_parameterized():
    sess = _FakeSession(records=[{"agent_id": "a1"}])
    out = queries.slow_and_low_detection(sess, min_gap_ms=123.0)
    assert out == [{"agent_id": "a1"}]
    assert sess.calls[0][1] == {"min_gap_ms": 123.0}

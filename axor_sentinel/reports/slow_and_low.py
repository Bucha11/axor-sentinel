from __future__ import annotations

from typing import Any

from axor_sentinel.graph import queries


class SlowAndLowReport:
    """
    Runs the slow-and-low staging detection Cypher query against Neo4j.

    Detects agents with a tainted session followed by an export session,
    where flagged resources were accessed during the tainted session.
    Results are sorted by gap_days descending — widest gaps first.
    """

    def __init__(self, neo4j_session: Any) -> None:
        """
        Args:
            neo4j_session: live neo4j.Session
        """
        self._session = neo4j_session

    def run(self, min_gap_days: float = 0.0) -> list[dict]:
        """
        Execute the slow-and-low detection query.

        Args:
            min_gap_days: minimum gap in days between tainted and export session.
                          Defaults to 0.0 (all cross-session patterns).

        Returns:
            List of dicts with keys:
              agent_id           — agent_id string
              s1.session_id      — tainted session ID
              s2.session_id      — export session ID
              gap_days           — float, days between sessions
              flagged_resources  — list[str] of resource IDs
              scores             — list[float] of suspicion scores
            Ordered by gap_days descending.
        """
        min_gap_ms = min_gap_days * 86_400_000.0
        return queries.slow_and_low_detection(self._session, min_gap_ms=min_gap_ms)

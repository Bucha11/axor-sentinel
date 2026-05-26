from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SignalType(str, Enum):
    """Hot-signal involvement depth for a resource in a tainted session."""
    READ = "read"                              # 0.4 — read-only access
    READ_SUMMARIZE = "read_summarize"          # 0.6 — read + summarized
    READ_EXPORT_ADJACENT = "read_export_adjacent"  # 0.8 — read + export nearby
    READ_EXPORT_FAILED = "read_export_failed"  # 1.0 — failed export / escalation triggered

    @property
    def _rank(self) -> int:
        """
        Explicit severity rank for ordering comparisons.

        Using a fixed dict instead of list(SignalType).index() prevents silent
        breakage if enum members are ever reordered or a new member is inserted.
        _SIGNAL_TYPE_RANK is defined at module level (below) to avoid Python
        treating a class attribute as an additional enum member.
        """
        return _SIGNAL_TYPE_RANK[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SignalType):
            return NotImplemented
        return self._rank < other._rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, SignalType):
            return NotImplemented
        return self._rank <= other._rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SignalType):
            return NotImplemented
        return self._rank > other._rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, SignalType):
            return NotImplemented
        return self._rank >= other._rank


# Explicit severity ordering for SignalType.
# Defined after the class so Python does not treat it as an enum member.
# 1 = least severe, 4 = most severe.
_SIGNAL_TYPE_RANK: dict[str, int] = {
    "read": 1,
    "read_summarize": 2,
    "read_export_adjacent": 3,
    "read_export_failed": 4,
}

# Raw hot weights keyed by signal type
HOT_WEIGHTS: dict[SignalType, float] = {
    SignalType.READ: 0.4,
    SignalType.READ_SUMMARIZE: 0.6,
    SignalType.READ_EXPORT_ADJACENT: 0.8,
    SignalType.READ_EXPORT_FAILED: 1.0,
}


@dataclass
class ResourceNode:
    """
    (:Resource) node in the reputation graph.

    suspicion_score accumulates cross-session via graded hot and caution weights.
    flagged is updated on every score change — never deferred (invariant A-2).
    last_signal_at and last_decay_at are tracked separately (invariant A-3).
    """
    id: str
    type: str                       # "file" | "sharepoint" | "onedrive" | "email" | "api" | ...
    service: str
    directory: str
    suspicion_score: float = 0.0
    flagged: bool = False
    last_signal_at: float = 0.0     # unix timestamp of last weight-contributing event
    last_decay_at: float = 0.0      # unix timestamp of last decay application
    normalization_method: str = "heuristic"  # "provider_id" | "path" | "heuristic"
    canonical_confidence: float = 0.4        # 1.0 | 0.7 | 0.4


@dataclass
class ContainerNode:
    """
    (:Container) node — directory / workspace / MCP namespace.

    suspicion_score is aggregated from member resource scores.
    flagged is updated whenever suspicion_score changes (invariant A-2, A-9).
    """
    id: str
    type: str                       # "directory" | "workspace" | "project" | "mcp_namespace"
    suspicion_score: float = 0.0
    flagged: bool = False
    member_ids: list[str] = field(default_factory=list)


@dataclass
class AgentNode:
    """(:Agent) node — stable agent identity across sessions."""
    agent_id: str


@dataclass
class SessionNode:
    """(:Session) node — one governed session."""
    session_id: str
    agent_id: str
    started_at: float
    had_taint: bool = False
    had_export_attempt: bool = False
    had_failed_export: bool = False
    had_escalation: bool = False


@dataclass
class DestinationNode:
    """(:Destination) node — export target."""
    id: str
    is_external: bool = False


@dataclass
class AccessedEdge:
    """(:Session)-[:ACCESSED {weight, signal_type, at}]->(:Resource)."""
    session_id: str
    resource_id: str
    weight: float
    signal_type: str        # SignalType value
    at: float               # unix timestamp


@dataclass
class AdjacentToEdge:
    """
    (:Resource)-[:ADJACENT_TO {topology_factor}]->(:Resource).

    Derived from topology — computed once when a resource is first seen.
    topology_factor values per spec:
      same directory/workspace      1.0
      same service/datasource       0.7
      same MCP namespace/policy     0.6
      same project/shared user      0.5
      different service             0.4
    """
    source_id: str
    target_id: str
    topology_factor: float

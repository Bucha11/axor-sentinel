from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class TopologyResource:
    """A resource node in a synthetic topology."""
    resource_id: str
    container_id: str
    service: str
    normalization_method: str   # "provider_id" | "path" | "heuristic"
    canonical_confidence: float


@dataclass
class TopologyContainer:
    """A container node in a synthetic topology."""
    container_id: str
    service: str
    resource_ids: list[str] = field(default_factory=list)


@dataclass
class TopologyEdge:
    """An ADJACENT_TO edge between two resources."""
    source_id: str
    target_id: str
    topology_factor: float  # 1.0/0.7/0.6/0.5/0.4 per spec


@dataclass
class Topology:
    """
    A synthetic resource graph for bench scenario generation.

    Provides a realistic corporate-ish resource graph with multiple services,
    containers, and mixed normalization methods.
    """
    topology_id: str
    resources: list[TopologyResource]
    containers: list[TopologyContainer]
    edges: list[TopologyEdge]

    def __post_init__(self) -> None:
        # Build O(1) index once at construction time.
        self._resource_map: dict[str, TopologyResource] = {
            r.resource_id: r for r in self.resources
        }

    def resources_in_container(self, container_id: str) -> list[TopologyResource]:
        return [r for r in self.resources if r.container_id == container_id]

    def adjacent_resources(self, resource_id: str) -> list[tuple[TopologyResource, float]]:
        """Return (resource, topology_factor) pairs for adjacent resources."""
        result = []
        for edge in self.edges:
            if edge.source_id == resource_id:
                r = self._resource(edge.target_id)
                if r:
                    result.append((r, edge.topology_factor))
        return result

    def _resource(self, rid: str) -> TopologyResource | None:
        """O(1) lookup by resource ID."""
        return self._resource_map.get(rid)


# Topology factor values from spec
_SAME_DIR_FACTOR = 1.0
_SAME_SERVICE_FACTOR = 0.7
_CROSS_SERVICE_FACTOR = 0.4

# Normalization methods and their confidences
_NORM_METHODS = [
    ("provider_id", 1.0),
    ("path", 0.7),
    ("heuristic", 0.4),
]


class TopologyGenerator:
    """
    Generates synthetic resource topologies for bench scenarios.

    Each topology has:
    - n_services services
    - n_containers_per_service containers per service
    - n_resources_per_container resources per container
    - Mixed normalization methods (at least one of each tier)
    - ADJACENT_TO edges between resources in the same container and across services
    """

    def __init__(
        self,
        n_services: int = 3,
        n_containers_per_service: int = 2,
        n_resources_per_container: int = 4,
        rng: random.Random | None = None,
    ) -> None:
        self._n_services = n_services
        self._n_containers_per_service = n_containers_per_service
        self._n_resources_per_container = n_resources_per_container
        self._rng = rng or random.Random()

    def generate(self, topology_id: str) -> Topology:
        resources: list[TopologyResource] = []
        containers: list[TopologyContainer] = []
        edges: list[TopologyEdge] = []

        services = [f"svc{i}" for i in range(self._n_services)]

        for svc in services:
            for ci in range(self._n_containers_per_service):
                cid = f"{svc}_c{ci}"
                container = TopologyContainer(
                    container_id=cid,
                    service=svc,
                )
                for ri in range(self._n_resources_per_container):
                    rid = f"{cid}_r{ri}"
                    # Cycle through normalization methods to guarantee at least one of each
                    method_idx = (len(resources)) % len(_NORM_METHODS)
                    method, confidence = _NORM_METHODS[method_idx]
                    resource = TopologyResource(
                        resource_id=rid,
                        container_id=cid,
                        service=svc,
                        normalization_method=method,
                        canonical_confidence=confidence,
                    )
                    resources.append(resource)
                    container.resource_ids.append(rid)
                containers.append(container)

        # Add ADJACENT_TO edges: same-container pairs
        container_map: dict[str, list[str]] = {}
        for r in resources:
            container_map.setdefault(r.container_id, []).append(r.resource_id)

        for cid, rids in container_map.items():
            for i, rid_a in enumerate(rids):
                for rid_b in rids[i + 1:]:
                    # Bidirectional
                    edges.append(TopologyEdge(rid_a, rid_b, _SAME_DIR_FACTOR))
                    edges.append(TopologyEdge(rid_b, rid_a, _SAME_DIR_FACTOR))

        # Cross-service edges: one pair per service combination
        service_resource_map: dict[str, list[str]] = {}
        for r in resources:
            service_resource_map.setdefault(r.service, []).append(r.resource_id)

        svc_list = list(service_resource_map.keys())
        for i, svc_a in enumerate(svc_list):
            for svc_b in svc_list[i + 1:]:
                ra = self._rng.choice(service_resource_map[svc_a])
                rb = self._rng.choice(service_resource_map[svc_b])
                edges.append(TopologyEdge(ra, rb, _CROSS_SERVICE_FACTOR))
                edges.append(TopologyEdge(rb, ra, _CROSS_SERVICE_FACTOR))

        return Topology(
            topology_id=topology_id,
            resources=resources,
            containers=containers,
            edges=edges,
        )

from __future__ import annotations

import random

from axor_sentinel.bench.topology.generator import Topology, TopologyGenerator

# Number of pre-generated topologies in the pool
POOL_SIZE: int = 10


class TopologyPool:
    """
    Pool of 10 pre-generated topologies with fixed seeds for reproducibility.

    Each topology uses a deterministic RNG seed so datasets built with
    paper_baseline.yaml produce identical topologies across runs.
    """

    def __init__(self) -> None:
        self._topologies: list[Topology] | None = None

    def _build(self) -> list[Topology]:
        topologies = []
        for i in range(POOL_SIZE):
            rng = random.Random(i * 1337)  # deterministic seed per topology
            gen = TopologyGenerator(rng=rng)
            topologies.append(gen.generate(topology_id=f"topo_{i}"))
        return topologies

    @property
    def topologies(self) -> list[Topology]:
        if self._topologies is None:
            self._topologies = self._build()
        return self._topologies

    def get(self, index: int) -> Topology:
        """Return topology by index (0..POOL_SIZE-1)."""
        if index < 0 or index >= POOL_SIZE:
            raise IndexError(f"topology index {index} out of range [0, {POOL_SIZE})")
        return self.topologies[index]

    def __len__(self) -> int:
        return POOL_SIZE

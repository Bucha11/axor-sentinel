from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from axor_sentinel.bench.dataset.schema import Scenario
from axor_sentinel.bench.scenarios import attack as atk
from axor_sentinel.bench.scenarios import benign as ben
from axor_sentinel.bench.topology.pool import TopologyPool


# Default paper-baseline composition counts.
# NOTE: bench/configs/paper_baseline.yaml is the human-readable source of truth
# for these numbers.  Changes here must be mirrored in that file (and vice versa).
# To compose directly from the YAML, use DatasetComposer.compose_from_yaml().
_PAPER_BASELINE_COUNTS: dict[str, int] = {
    "slow_and_low_2": 90,
    "slow_and_low_4": 90,
    "slow_and_low_8": 90,
    "fanout_3": 30,
    "fanout_5": 30,
    "fanout_10": 30,
    "distributed_staging": 60,
    "benign_narrow": 150,
    "benign_broad_etl": 150,
    "benign_false_taint": 100,
}


class DatasetComposer:
    """
    Assembles synthetic bench datasets by config.

    Config dict keys (all optional, default to paper baseline):
        slow_and_low_2:       int — number of 2-session slow_and_low scenarios
        slow_and_low_4:       int
        slow_and_low_8:       int
        fanout_3:             int — fanout with 3 containers
        fanout_5:             int
        fanout_10:            int
        distributed_staging:  int
        benign_narrow:        int
        benign_broad_etl:     int  — split equally between broad and etl profiles
        benign_false_taint:   int
        seed:                 int — RNG seed (default: 42)
    """

    def __init__(self, pool: TopologyPool | None = None) -> None:
        self._pool = pool or TopologyPool()

    def compose(self, config: dict[str, Any] | None = None) -> list[Scenario]:
        """
        Compose a dataset from config.

        Returns list of Scenario objects sorted by scenario_id.
        """
        cfg = dict(_PAPER_BASELINE_COUNTS)
        if config:
            cfg.update(config)

        seed = int(cfg.get("seed", 42))
        rng = random.Random(seed)

        scenarios: list[Scenario] = []

        def _topo(i: int):
            return self._pool.get(i % len(self._pool))

        # slow_and_low variants
        for n, key in [(2, "slow_and_low_2"), (4, "slow_and_low_4"), (8, "slow_and_low_8")]:
            count = int(cfg.get(key, 0))
            for i in range(count):
                s = atk.slow_and_low(
                    atk.SlowAndLowConfig(staging_sessions=n),
                    _topo(i),
                    scenario_id=f"{key}_{i:04d}",
                    rng=rng,
                )
                scenarios.append(s)

        # fanout variants
        for n, key in [(3, "fanout_3"), (5, "fanout_5"), (10, "fanout_10")]:
            count = int(cfg.get(key, 0))
            for i in range(count):
                s = atk.fanout(
                    atk.FanoutConfig(n_containers=n, agent_profile="narrow"),
                    _topo(i),
                    scenario_id=f"{key}_{i:04d}",
                    rng=rng,
                )
                scenarios.append(s)

        # distributed staging
        dist_count = int(cfg.get("distributed_staging", 0))
        for i in range(dist_count):
            s = atk.distributed_staging(
                atk.DistributedStagingConfig(),
                _topo(i),
                scenario_id=f"dist_{i:04d}",
                rng=rng,
            )
            scenarios.append(s)

        # benign narrow
        bn_count = int(cfg.get("benign_narrow", 0))
        for i in range(bn_count):
            s = ben.benign_narrow(
                _topo(i),
                scenario_id=f"benign_narrow_{i:04d}",
                rng=rng,
            )
            scenarios.append(s)

        # benign broad/etl (split equally)
        be_count = int(cfg.get("benign_broad_etl", 0))
        for i in range(be_count):
            profile = "etl" if i % 2 == 0 else "broad"
            s = ben.benign_fanout_like(
                _topo(i),
                agent_profile=profile,
                scenario_id=f"benign_fanout_{profile}_{i:04d}",
                rng=rng,
            )
            scenarios.append(s)

        # benign false taint
        ft_count = int(cfg.get("benign_false_taint", 0))
        for i in range(ft_count):
            s = ben.benign_false_taint(
                _topo(i),
                scenario_id=f"benign_false_taint_{i:04d}",
                rng=rng,
            )
            scenarios.append(s)

        return scenarios

    @classmethod
    def compose_from_yaml(
        cls,
        path: Path | str,
        pool: TopologyPool | None = None,
    ) -> list[Scenario]:
        """
        Load composition config from a YAML file and return the dataset.

        Requires ``pyyaml`` (``pip install pyyaml``).  The bundled config is at
        ``axor_sentinel/bench/configs/paper_baseline.yaml``.

        Example::

            from pathlib import Path
            from axor_sentinel.bench.dataset.composer import DatasetComposer

            scenarios = DatasetComposer.compose_from_yaml(
                Path("axor_sentinel/bench/configs/paper_baseline.yaml")
            )

        Args:
            path: path to a YAML file with the same keys as _PAPER_BASELINE_COUNTS.
            pool: optional pre-built TopologyPool (uses default pool if None).

        Raises:
            ImportError: if pyyaml is not installed.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required to load YAML configs: pip install pyyaml"
            ) from exc
        with open(path, encoding="utf-8") as fh:
            config: dict[str, Any] = yaml.safe_load(fh)
        return cls(pool=pool).compose(config=config)

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class AgentProfile:
    """
    Statistical profile of an agent's normal container access behavior.

    Used to generate realistic session histories and baselines for bench scenarios.
    mean_containers and std are the parameters of the normal distribution
    from which unique container counts per session are sampled.
    """
    name: str
    mean_containers: float
    std: float
    description: str = ""


# Standard profiles from spec §Agent profiles
PROFILES: dict[str, AgentProfile] = {
    "narrow": AgentProfile(
        name="narrow",
        mean_containers=1.5,
        std=0.5,
        description="Specialized agent, tight access",
    ),
    "broad": AgentProfile(
        name="broad",
        mean_containers=6.0,
        std=2.0,
        description="Research agent, wide access",
    ),
    "noisy": AgentProfile(
        name="noisy",
        mean_containers=4.0,
        std=3.5,
        description="Unpredictable, high std",
    ),
    "etl": AgentProfile(
        name="etl",
        mean_containers=8.0,
        std=1.0,
        description="Wide but stable — fanout baseline absorbs",
    ),
    "research": AgentProfile(
        name="research",
        mean_containers=5.0,
        std=2.5,
        description="Analytical, moderate",
    ),
}


class AgentProfileGenerator:
    """
    Samples realistic per-session container counts from an agent profile.

    etl and broad are intentionally hard cases for fanout detection — they
    legitimately touch many containers and must NOT trigger FanoutSignal.
    """

    def __init__(self, profile_name: str, rng: random.Random | None = None) -> None:
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown agent profile: {profile_name!r}. "
                f"Valid: {list(PROFILES)}"
            )
        self.profile = PROFILES[profile_name]
        self._rng = rng or random.Random()

    def sample_containers(self) -> int:
        """
        Sample the number of unique containers for one session.

        Returns at least 1 (agents always access something).
        """
        sample = self._rng.gauss(
            self.profile.mean_containers,
            self.profile.std,
        )
        return max(1, round(sample))

    def generate_baseline_counts(self, n_sessions: int) -> list[int]:
        """Generate n_sessions container counts for baseline computation."""
        return [self.sample_containers() for _ in range(n_sessions)]

from __future__ import annotations

from dataclasses import dataclass

from axor_sentinel.graph.model import HOT_WEIGHTS, SignalType

# ── Thresholds ─────────────────────────────────────────────────────────────────

FLAG_THRESHOLD: float = 0.7            # suspicion_score >= this → flagged=True
BASE_CAUTION: float = 0.3             # base caution weight for adjacent resources
CONTAINER_MEMBER_THRESHOLD: float = 0.2  # only members above this count toward container score


# ── Core accumulation (logarithmic, invariant A-1) ─────────────────────────────

def accumulate(current: float, new_weight: float) -> float:
    """
    Logarithmic score accumulation — diminishing returns as score approaches 1.0.

    Each additional signal contributes proportionally to remaining headroom.
    Score is bounded to [0.0, 1.0] (invariant A-1).
    Two weak signals no longer equal one strong signal.
    """
    delta = new_weight * (1.0 - current)
    return min(1.0, current + delta)


# ── Time decay ─────────────────────────────────────────────────────────────────

def time_decay(days_since_last_decay: float) -> float:
    """
    Exponential half-life decay: score halves every 30 days.

    Always computed against last_decay_at, never last_signal_at (invariant A-3).
    At day 0 → 1.0; day 30 → 0.5; day 60 → 0.25; day 90 → 0.125.
    """
    return 0.5 ** (days_since_last_decay / 30.0)


def apply_decay_to_score(current_score: float, days_since_last_decay: float) -> float:
    """Multiply current score by time decay factor."""
    return current_score * time_decay(days_since_last_decay)


# ── Poisoning-mitigation factors ───────────────────────────────────────────────

def source_diversity_factor(signal_history: list[str], current_source: str) -> float:
    """
    Reduce effective weight when signals are concentrated from one taint source.

    Diverse independent sources produce a stronger signal than a single source
    repeating. Concentration at 100% → 70% weight reduction.

    Args:
        signal_history: list of taint_source values from past ReputationEvents on this resource
        current_source: taint_source for the new signal
    """
    total = len(signal_history)
    if total == 0:
        return 1.0
    from_this_source = sum(1 for s in signal_history if s == current_source)
    concentration = from_this_source / total   # 1.0 = all from one source
    return 1.0 - (concentration * 0.7)         # max 70% reduction at full concentration


def origin_dampening(prior_count: int) -> float:
    """
    Exponential dampening for repeated signals from the same taint source.

    Nth signal from the same source to the same resource:
      1st (prior_count=0): 1.0
      2nd (prior_count=1): 0.5
      3rd (prior_count=2): 0.25
      Nth: 0.5^(N-1) — never 0, never > 1 (invariant A-13)

    Args:
        prior_count: number of previous signals from this source to this resource
    """
    return 0.5 ** prior_count


@dataclass(frozen=True)
class WeightFactors:
    """Two views of one signal's weight, derived from a SINGLE computation of the
    diversity/dampening factors so the in-memory accumulate and the Cypher pre-scale
    cannot drift.

    effective:          raw * confidence * diversity * dampening — what the in-memory
                        ``accumulate`` consumes for the snapshot.
    without_confidence: raw * diversity * dampening — what the hot-weight Cypher gets;
                        the query multiplies it by ``r.canonical_confidence`` itself,
                        so passing the full effective weight would double-count
                        confidence.
    """
    effective: float
    without_confidence: float


def compute_weight_factors(
    raw_weight: float,
    canonical_confidence: float,
    signal_history: list[str],
    current_source: str,
    prior_count_from_source: int,
) -> WeightFactors:
    """Compute both weight views from ONE evaluation of diversity + dampening.

    The in-memory path and the Neo4j path used to compute ``diversity * dampening``
    independently (the cycle recomputed them inline for a hand-tuned pre-scale),
    which could silently drift from ``compute_effective_weight``. Both now derive
    from the same ``without_confidence`` here (invariant A-8)."""
    div_factor = source_diversity_factor(signal_history, current_source)
    damp_factor = origin_dampening(prior_count_from_source)
    without_confidence = raw_weight * div_factor * damp_factor
    return WeightFactors(
        effective=without_confidence * canonical_confidence,
        without_confidence=without_confidence,
    )


def compute_effective_weight(
    raw_weight: float,
    canonical_confidence: float,
    signal_history: list[str],
    current_source: str,
    prior_count_from_source: int,
) -> float:
    """
    Full effective weight per spec:
        effective_weight = raw_weight
                         * canonical_confidence
                         * source_diversity_factor
                         * origin_dampening
    (invariant A-8). Thin wrapper over compute_weight_factors().
    """
    return compute_weight_factors(
        raw_weight=raw_weight,
        canonical_confidence=canonical_confidence,
        signal_history=signal_history,
        current_source=current_source,
        prior_count_from_source=prior_count_from_source,
    ).effective


# ── Hot weight computation ─────────────────────────────────────────────────────

def compute_hot_weight(signal_type: SignalType) -> float:
    """Return raw hot weight for the given signal type."""
    return HOT_WEIGHTS[signal_type]


# ── Container aggregation ──────────────────────────────────────────────────────

def compute_container_score(member_scores: list[float]) -> float:
    """
    Container suspicion_score = mean of member scores above CONTAINER_MEMBER_THRESHOLD.

    Recomputed after any member resource score change (invariant A-9).
    Returns 0.0 if no member exceeds the threshold.
    """
    above = [s for s in member_scores if s > CONTAINER_MEMBER_THRESHOLD]
    if not above:
        return 0.0
    return sum(above) / len(above)


# ── Score update with invariant enforcement ────────────────────────────────────

def update_resource_score(
    current_score: float,
    effective_weight: float,
    now: float,
) -> tuple[float, bool, float]:
    """
    Apply effective_weight to current_score and return (new_score, flagged, last_signal_at).

    Invariant A-1: score never exceeds 1.0.
    Invariant A-2: flagged updated immediately on every score change.
    last_signal_at is set to `now` (invariant A-3: last_decay_at is NOT updated here).

    Returns:
        (new_score, flagged, new_last_signal_at)
    """
    new_score = accumulate(current_score, effective_weight)
    flagged = new_score >= FLAG_THRESHOLD
    return new_score, flagged, now

"""
Adversarial tests for the weight model.

Covers invariants A-1..A-4, A-8, A-13 and spec adversarial test matrix:
  - Graded hot signal ordering (3 variants)
  - Logarithmic saturation (3 variants)
  - Decay field separation (2 variants)
  - flagged consistency (2 variants)
  - Repeated-origin dampening (3 variants)
  - Source concentration reduction (3 variants)
  - Normalization confidence scaling (3 variants)
"""
from __future__ import annotations

import pytest
from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.weight import (
    FLAG_THRESHOLD,
    accumulate,
    apply_decay_to_score,
    compute_container_score,
    compute_effective_weight,
    compute_hot_weight,
    compute_weight_factors,
    origin_dampening,
    source_diversity_factor,
    time_decay,
    update_resource_score,
)


class TestWeightFactorsSingleSource:
    """The in-memory `effective` and the Cypher `without_confidence` derive from one
    computation, so they can't drift (the dual-write reconciliation risk)."""

    @pytest.mark.parametrize("conf", [0.4, 0.7, 1.0])
    @pytest.mark.parametrize("history,source,prior", [
        ([], "mcp", 0),
        (["mcp", "mcp"], "mcp", 2),
        (["web", "file"], "mcp", 0),
    ])
    def test_effective_equals_without_confidence_times_conf(self, conf, history, source, prior):
        wf = compute_weight_factors(
            raw_weight=0.8, canonical_confidence=conf,
            signal_history=history, current_source=source, prior_count_from_source=prior,
        )
        assert wf.effective == pytest.approx(wf.without_confidence * conf)

    def test_matches_compute_effective_weight(self):
        kwargs = dict(
            raw_weight=0.6, canonical_confidence=0.7,
            signal_history=["mcp"], current_source="mcp", prior_count_from_source=1,
        )
        assert compute_weight_factors(**kwargs).effective == pytest.approx(
            compute_effective_weight(**kwargs)
        )


# ── Graded hot signal ordering — 3 variants ────────────────────────────────────

class TestGradedHotSignalOrdering:
    def test_failed_export_highest(self):
        """read_export_failed > read_export_adjacent"""
        assert (
            compute_hot_weight(SignalType.READ_EXPORT_FAILED)
            > compute_hot_weight(SignalType.READ_EXPORT_ADJACENT)
        )

    def test_export_adjacent_above_summarize(self):
        """read_export_adjacent > read_summarize"""
        assert (
            compute_hot_weight(SignalType.READ_EXPORT_ADJACENT)
            > compute_hot_weight(SignalType.READ_SUMMARIZE)
        )

    def test_summarize_above_read(self):
        """read_summarize > read"""
        assert (
            compute_hot_weight(SignalType.READ_SUMMARIZE)
            > compute_hot_weight(SignalType.READ)
        )


# ── Logarithmic saturation — 3 variants ───────────────────────────────────────

class TestLogarithmicSaturation:
    def test_score_never_exceeds_one(self):
        """Invariant A-1: score bounded at 1.0 even with weight=1.0 applied many times."""
        score = 0.0
        for _ in range(100):
            score = accumulate(score, 1.0)
        assert score <= 1.0

    def test_repeated_access_does_not_saturate(self):
        """Many small weights accumulate to less than a single large weight."""
        # Applying 0.1 ten times should be less than applying 1.0 once
        score_many_small = 0.0
        for _ in range(10):
            score_many_small = accumulate(score_many_small, 0.1)

        score_one_large = accumulate(0.0, 1.0)

        assert score_many_small < score_one_large

    def test_diminishing_returns_property(self):
        """Each subsequent application of the same weight contributes less."""
        score = 0.0
        deltas = []
        for _ in range(5):
            new_score = accumulate(score, 0.5)
            deltas.append(new_score - score)
            score = new_score

        # Each delta should be strictly smaller than the previous
        for i in range(1, len(deltas)):
            assert deltas[i] < deltas[i - 1]


# ── Decay field separation — 2 variants ───────────────────────────────────────

class TestDecayFieldSeparation:
    def test_hot_signal_does_not_update_last_decay_at(self):
        """
        Invariant A-3: applying a hot weight updates last_signal_at but NOT last_decay_at.
        update_resource_score returns (new_score, flagged, new_last_signal_at) — it does
        NOT return a new last_decay_at. The caller tracks decay separately.
        """
        now = 1_000_000.0
        new_score, flagged, new_last_signal_at = update_resource_score(0.0, 0.4, now)
        # last_signal_at is updated
        assert new_last_signal_at == now
        # There is no new_last_decay_at in the return value — decay field is NOT updated
        # (verified by the return tuple length)
        assert isinstance(new_score, float)
        assert isinstance(flagged, bool)

    def test_decay_does_not_update_score_above_pre_decay(self):
        """
        Invariant A-4: decay reduces score; it is always applied before new weights.
        After decay, applying time_decay(30) to 1.0 gives 0.5.
        """
        score_before = 1.0
        decayed = apply_decay_to_score(score_before, days_since_last_decay=30.0)
        assert abs(decayed - 0.5) < 1e-9
        # Decayed score is strictly less than original (for non-zero elapsed time)
        assert decayed < score_before


# ── flagged consistency — 2 variants ──────────────────────────────────────────

class TestFlaggedConsistency:
    def test_flagged_true_when_score_at_threshold(self):
        """Invariant A-2: flagged=True when score >= FLAG_THRESHOLD (0.7)."""
        # Force score to exactly FLAG_THRESHOLD
        score = FLAG_THRESHOLD
        _, flagged, _ = update_resource_score(0.0, score, 0.0)
        # accumulate(0.0, 0.7) = 0.7 * (1-0) = 0.7
        assert flagged is True

    def test_flagged_false_below_threshold(self):
        """Invariant A-2: flagged=False when score < FLAG_THRESHOLD."""
        score, flagged, _ = update_resource_score(0.0, 0.3, 0.0)
        assert flagged is False
        assert score < FLAG_THRESHOLD


# ── Repeated-origin dampening — 3 variants ────────────────────────────────────

class TestRepeatedOriginDampening:
    def test_first_signal_no_dampening(self):
        """Invariant A-13: 1st signal from source = factor 1.0 (prior_count=0)."""
        assert origin_dampening(0) == 1.0

    def test_second_signal_half_dampening(self):
        """Invariant A-13: 2nd signal (prior_count=1) = factor 0.5."""
        assert origin_dampening(1) == 0.5

    def test_fifth_signal_exponential_dampening(self):
        """Invariant A-13: 5th signal (prior_count=4) = 0.5^4 = 0.0625."""
        assert abs(origin_dampening(4) - 0.0625) < 1e-9


# ── Source concentration reduction — 3 variants ───────────────────────────────

class TestSourceConcentrationReduction:
    def test_single_source_max_reduction(self):
        """100% concentration from one source → 70% weight reduction (factor=0.3)."""
        history = ["mcp"] * 10
        factor = source_diversity_factor(history, "mcp")
        assert abs(factor - 0.3) < 1e-9

    def test_perfectly_diverse_sources(self):
        """All unique sources → concentration = 1/n → minimal reduction."""
        history = ["mcp", "web", "file", "api", "memory"]
        # current source appears once (1 out of 5)
        factor = source_diversity_factor(history, "memory")
        # concentration = 1/5 = 0.2 → factor = 1 - 0.2*0.7 = 0.86
        assert abs(factor - (1.0 - 0.2 * 0.7)) < 1e-9

    def test_empty_history_no_reduction(self):
        """No prior signals → diversity factor = 1.0."""
        assert source_diversity_factor([], "mcp") == 1.0


# ── Normalization confidence scaling — 3 variants ─────────────────────────────

class TestNormalizationConfidenceScaling:
    def test_heuristic_weight_is_04x_provider_id(self):
        """Heuristic-normalized resource (confidence=0.4) contributes 40% vs provider ID (1.0)."""
        raw = 1.0
        history: list[str] = []

        eff_provider = compute_effective_weight(raw, 1.0, history, "mcp", 0)
        eff_heuristic = compute_effective_weight(raw, 0.4, history, "mcp", 0)

        assert abs(eff_heuristic - eff_provider * 0.4) < 1e-9

    def test_path_confidence_between_provider_and_heuristic(self):
        """Path-normalized (0.7) contributes more than heuristic (0.4) and less than provider (1.0)."""
        raw = 1.0
        history: list[str] = []

        eff_provider = compute_effective_weight(raw, 1.0, history, "mcp", 0)
        eff_path = compute_effective_weight(raw, 0.7, history, "mcp", 0)
        eff_heuristic = compute_effective_weight(raw, 0.4, history, "mcp", 0)

        assert eff_heuristic < eff_path < eff_provider

    def test_effective_weight_formula(self):
        """Invariant A-8: effective = raw * confidence * diversity * dampening."""
        raw = 0.8
        confidence = 0.7
        history = ["mcp"] * 3
        current = "mcp"
        prior_count = 2

        div = source_diversity_factor(history, current)
        damp = origin_dampening(prior_count)
        expected = raw * confidence * div * damp

        actual = compute_effective_weight(raw, confidence, history, current, prior_count)
        assert abs(actual - expected) < 1e-12


# ── Container aggregation — 2 variants ────────────────────────────────────────

class TestContainerAggregation:
    def test_container_only_counts_above_threshold(self):
        """Container score = mean of members above CONTAINER_MEMBER_THRESHOLD (0.2)."""
        scores = [0.5, 0.3, 0.1]  # 0.1 is below threshold
        result = compute_container_score(scores)
        assert abs(result - (0.5 + 0.3) / 2) < 1e-9

    def test_empty_container_score_is_zero(self):
        """Container with no members above threshold → score=0.0."""
        assert compute_container_score([]) == 0.0
        assert compute_container_score([0.1, 0.15]) == 0.0


# ── Decay table spot checks ────────────────────────────────────────────────────

@pytest.mark.parametrize("days,expected", [
    (0, 1.0),
    (30, 0.5),
    (60, 0.25),
    (90, 0.125),
])
def test_decay_table(days, expected):
    assert abs(time_decay(days) - expected) < 1e-9

from __future__ import annotations

# Phase 2 activation thresholds (spec §Layer 2 integration — Phase 2)
N_MIN_REAL_ATTACKS: int = 50    # minimum confirmed cross-session staging incidents
MAX_FP_RATE: float = 0.02       # maximum false positive rate on held-out clean sessions

# Current phase — set to 1 until Phase 2 activation conditions are met.
# Check is_phase2_active() rather than reading PHASE directly.
PHASE: int = 1


def is_phase2_active(confirmed_incidents: int, fp_rate: float) -> bool:
    """
    Return True only when all Phase 2 activation conditions are satisfied (invariant A-12).

    Conditions (all must be true):
      - confirmed_incidents >= N_MIN_REAL_ATTACKS (50) — synthetic data alone insufficient
      - fp_rate <= MAX_FP_RATE (0.02) — false positive rate on held-out clean sessions
      - Feature importance analysis would be done offline; this gate covers the data threshold.

    Until these conditions are met, Phase 1 deterministic rule is the sole enforcement
    mechanism. Reputation floats are NOT passed to Layer 2 (invariant A-11).
    """
    return (
        confirmed_incidents >= N_MIN_REAL_ATTACKS
        and fp_rate <= MAX_FP_RATE
    )

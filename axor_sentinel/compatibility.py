"""axor-core compatibility check.

axor-core is a *peer* dependency, not a hard pin (the packages are intentionally
decoupled and rebuilt independently). That decoupling makes silent version skew
possible: sentinel reads axor-core contracts (NormalizedIntent, reputation
fields) whose shape can drift between releases.

This module declares the compatible axor-core range and offers a warn-only check
the host application can call at startup. It never raises by default, so it
cannot take down a deployment — but it makes skew visible instead of silent.
"""

from __future__ import annotations

import logging
import warnings

log = logging.getLogger("axor.sentinel.compat")

# Inclusive lower bound, exclusive upper bound — the axor-core range these
# sentinel contracts were validated against. Bump on contract-affecting releases.
MIN_AXOR_CORE = (0, 7, 0)
MAX_AXOR_CORE = (0, 10, 0)


def _parse(version: str) -> tuple[int, int, int]:
    parts = version.split("+")[0].split("-")[0].split(".")
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])  # type: ignore[return-value]


def check_axor_core_version(*, raise_on_mismatch: bool = False) -> bool:
    """Return True if the installed axor-core is within the compatible range.

    Emits a warning (or raises, if requested) on skew or when axor-core is not
    importable. Safe to call at startup.
    """
    try:
        import axor_core
        installed = _parse(getattr(axor_core, "__version__", "0.0.0"))
    except Exception as exc:  # pragma: no cover - environment-specific
        msg = f"axor-core not importable for compatibility check: {exc}"
        if raise_on_mismatch:
            raise RuntimeError(msg) from exc
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False

    if MIN_AXOR_CORE <= installed < MAX_AXOR_CORE:
        return True

    msg = (
        f"axor-core {'.'.join(map(str, installed))} is outside the range "
        f"sentinel was validated against "
        f"[{'.'.join(map(str, MIN_AXOR_CORE))}, {'.'.join(map(str, MAX_AXOR_CORE))}). "
        "Contract drift may cause silent misbehaviour; pin compatible versions."
    )
    if raise_on_mismatch:
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    log.warning(msg)
    return False

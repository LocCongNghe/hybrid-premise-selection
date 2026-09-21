"""Cost-aware adaptive cascade policy.

Selects a per-query candidate limit ``K`` *before* retrieval, using only query-side
features (raw/unsimplified target tree size), so the backend can return exactly ``K``
candidates and the scoring stage runs on a bounded set. The policy is deterministic
and depends only on the query, never on candidate scores.
"""

from __future__ import annotations

from tbps.config import CascadeConfig


def choose_K(
    raw_target_size: int,
    config: CascadeConfig,
    *,
    query_depth: int | None = None,
) -> int:
    """Pick the candidate limit for a query from its raw (pre-simplify) tree size.

    ``k_bins`` has one more entry than ``size_thresholds``; each threshold is a strict
    upper bound for the *lower* bins. The final bin is unbounded (catches large trees).

    >>> cfg = CascadeConfig(enabled=True, k_bins=(100, 300, 750, 1500),
    ...                     size_thresholds=(20, 50, 100))
    >>> choose_K(5, cfg)
    100
    >>> choose_K(20, cfg)
    300
    >>> choose_K(75, cfg)
    750
    >>> choose_K(1000, cfg)
    1500
    """
    if config.policy != "raw_size":
        raise ValueError(f"unsupported cascade policy: {config.policy}")
    bins = config.k_bins
    thresholds = config.size_thresholds
    if not bins or len(bins) != len(thresholds) + 1:
        raise ValueError(
            f"cascade k_bins ({len(bins)}) must have one more entry than "
            f"size_thresholds ({len(thresholds)})"
        )
    # Each threshold is a strict upper bound for a lower bin: a query of size s maps to
    # bins[i] when s < thresholds[i], i.e. it is strictly below that bin's upper bound.
    # The final bin is unbounded (catches large trees).
    size = raw_target_size
    for index, upper in enumerate(thresholds):
        if size < upper:
            return bins[index]
    return bins[-1]

"""Evaluation and analysis utilities for relay-control policies."""

from .comparison import (
    PolicyComparisonConfig,
    PolicyComparisonResult,
    PolicyEpisodeResult,
    PolicySummary,
    compare_policies,
)

__all__ = [
    "PolicyComparisonConfig", "PolicyComparisonResult", "PolicyEpisodeResult",
    "PolicySummary", "compare_policies",
]

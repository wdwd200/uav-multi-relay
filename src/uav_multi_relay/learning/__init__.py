"""Neural-network building blocks for future multi-agent learning."""

from .networks import CentralizedTwinCritic, SharedGaussianActor

__all__ = ["CentralizedTwinCritic", "SharedGaussianActor"]

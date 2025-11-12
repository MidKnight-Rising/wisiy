"""
Smoke Runtime - Predictive Memory-Efficient Runtime for Large Model Inference

A framework for running large-scale models on limited GPU hardware through
predictive weight loading and just-in-time execution.

Core Components:
    - WeightCache: LRU-based RAM cache for model weights
    - SmokeTestSimulator: Predictive analysis of weight requirements
    - WeightPrefetcher: Async weight transfer pipeline (disk→RAM→GPU)
    - SmokeRuntime: Main orchestrator for the entire pipeline
"""

__version__ = "0.1.0"
__author__ = "MidKnight-Rising"

from .config import MemoryConfig, DeviceConfig
from .cache import WeightCache
from .simulator import SmokeTestSimulator
from .prefetcher import WeightPrefetcher
from .runtime import SmokeRuntime

__all__ = [
    "MemoryConfig",
    "DeviceConfig",
    "WeightCache",
    "SmokeTestSimulator",
    "WeightPrefetcher",
    "SmokeRuntime",
]

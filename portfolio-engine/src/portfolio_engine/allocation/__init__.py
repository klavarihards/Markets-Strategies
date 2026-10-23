"""Allocation methods. Phase 1: benchmarks only; the five methods arrive in Phase 2."""

from .base import Allocator, validate_weights
from .benchmarks import EqualWeight, FixedWeights, benchmark_allocators

__all__ = ["Allocator", "validate_weights", "EqualWeight", "FixedWeights", "benchmark_allocators"]

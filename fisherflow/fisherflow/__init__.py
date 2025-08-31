"""
Fisher Flow: Information-Geometric Framework for Sequential Estimation

A unified framework showing that many optimization methods (Adam, Natural Gradient, EWC)
are special cases of propagating Fisher information rather than probability distributions.
"""

from .core import FisherFlowEstimator, FisherInformation
from .optimizers import (
    DiagonalFisherFlow,
    FullFisherFlow,
    KroneckerFisherFlow,
    AdamAsFisherFlow,
    NaturalGradientFlow
)
from .utils import compute_empirical_fisher, compute_expected_fisher

__version__ = "0.1.0"
__author__ = "Fisher Flow Contributors"

__all__ = [
    "FisherFlowEstimator",
    "FisherInformation",
    "DiagonalFisherFlow",
    "FullFisherFlow",
    "KroneckerFisherFlow",
    "AdamAsFisherFlow",
    "NaturalGradientFlow",
    "compute_empirical_fisher",
    "compute_expected_fisher",
]
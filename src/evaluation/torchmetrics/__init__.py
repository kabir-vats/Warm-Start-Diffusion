"""Minimal metric exports needed by the ERA5-MINI training stack."""

import logging

_logger = logging.getLogger("src.evaluation.torchmetrics")

from src.evaluation.torchmetrics.metric import Metric
from src.evaluation.torchmetrics.regression.crps import ContinuousRankedProbabilityScore
from src.evaluation.torchmetrics.regression.gradient_magnitude_percent_diff import (
    GradientMagnitudePercentDifference,
)
from src.evaluation.torchmetrics.regression.mae import MeanAbsoluteError
from src.evaluation.torchmetrics.regression.mean_error import MeanError
from src.evaluation.torchmetrics.regression.mse import MeanSquaredError
from src.evaluation.torchmetrics.regression.spread_skill_ratio import SpreadSkillRatio

__all__ = [
    "ContinuousRankedProbabilityScore",
    "GradientMagnitudePercentDifference",
    "MeanAbsoluteError",
    "MeanError",
    "MeanSquaredError",
    "Metric",
    "SpreadSkillRatio",
]

"""Regression metrics used by ERA5-MINI evaluation."""

from src.evaluation.torchmetrics.regression.crps import ContinuousRankedProbabilityScore
from src.evaluation.torchmetrics.regression.gradient_magnitude_percent_diff import (
    GradientMagnitudePercentDifference,
)
from src.evaluation.torchmetrics.regression.mae import MeanAbsoluteError
from src.evaluation.torchmetrics.regression.mean import Average
from src.evaluation.torchmetrics.regression.mean_error import MeanError
from src.evaluation.torchmetrics.regression.mse import MeanSquaredError
from src.evaluation.torchmetrics.regression.spread_skill_ratio import SpreadSkillRatio
from src.evaluation.torchmetrics.regression.variance import StdDeviation

__all__ = [
    "Average",
    "ContinuousRankedProbabilityScore",
    "GradientMagnitudePercentDifference",
    "MeanAbsoluteError",
    "MeanError",
    "MeanSquaredError",
    "SpreadSkillRatio",
    "StdDeviation",
]

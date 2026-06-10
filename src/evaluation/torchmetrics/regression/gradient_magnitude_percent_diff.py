# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional

from torch import Tensor, tensor

from src.evaluation.metrics import weighted_mean_gradient_magnitude
from src.evaluation.torchmetrics.metric import Metric
from src.evaluation.torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["GradientMagnitudePercentDifference.plot"]


class GradientMagnitudePercentDifference(Metric):
    r"""Compute standard deviation or variance."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    sum_grad_mag_diff: Tensor
    total: Tensor

    def __init__(
        self,
        ensemble_dim: Optional[int] = 0,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weights = weights
        self.ensemble_dim = ensemble_dim
        assert ensemble_dim is None or ensemble_dim == 0, "Only ensemble_dim=0 is supported"
        self.ensemble_size = None
        self.add_state("sum_grad_mag_diff", default=tensor(0.0), dist_reduce_fx="sum")
        # self.add_state("sum_grad_mag_diff", default=torch.zeros(error_shape), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        weight_dims = tuple(range(-len(target.shape) + 1, 0))
        truth_grad = weighted_mean_gradient_magnitude(target, self.weights, weight_dims)
        if self.ensemble_dim is None:
            pred_grad = weighted_mean_gradient_magnitude(preds, self.weights, weight_dims)
        else:
            if self.ensemble_size is None:
                self.ensemble_size = preds.shape[self.ensemble_dim]
            assert (
                self.ensemble_size == preds.shape[self.ensemble_dim]
            ), f"Ensemble size changed from {self.ensemble_size} to {preds.shape[self.ensemble_dim]}"
            pred_grad = 0
            for ens_i, pred in enumerate(preds.unbind(self.ensemble_dim)):
                pred_grad += weighted_mean_gradient_magnitude(pred, self.weights, weight_dims)
            pred_grad /= self.ensemble_size
        self.sum_grad_mag_diff += 100 * ((pred_grad - truth_grad) / truth_grad).sum()
        self.total += target.shape[0]

    def compute(self) -> Tensor:
        sum_grad_mag_diff = self.sum_grad_mag_diff
        return sum_grad_mag_diff / self.total

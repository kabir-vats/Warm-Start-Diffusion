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
from collections.abc import Sequence
from typing import Any, Optional, Union

from torch import Tensor, tensor

from src.evaluation.metrics import weighted_mean
from src.evaluation.torchmetrics.metric import Metric
from src.evaluation.torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["StdDeviation.plot"]


class Average(Metric):
    r"""Compute standard deviation or variance."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    mean_sum: Tensor
    total: Tensor

    def __init__(
        self,
        batch_dim: int = 0,
        weights: Optional[Tensor] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.batch_dim = batch_dim
        self.weights = weights
        self.source = source

        self.add_state("mean_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, data: Tensor, target: Tensor = None) -> None:
        assert self.source is not None, "source must be specified. Otherwise, use `update_single_argument`"
        if self.source == "pred":
            data_to_use = data
        elif self.source == "target":
            data_to_use = target
        else:
            raise ValueError("source must be either 'pred' or 'target'")
        self.update_single_argument(data_to_use)

    def update_single_argument(self, data: Tensor) -> None:
        dims = [d for d in range(data.dim()) if d != self.batch_dim]
        if self.weights is None:
            mean_sum = data.mean(dim=dims)
        else:
            mean_sum = weighted_mean(data, weights=self.weights, dim=dims)
        self.mean_sum += mean_sum.sum()
        self.total += mean_sum.numel()

    def compute(self) -> Tensor:
        mean = self.mean_sum / self.total
        return mean

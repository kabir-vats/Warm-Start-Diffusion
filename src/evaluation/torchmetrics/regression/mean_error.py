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
from src.evaluation.torchmetrics.functional.regression.mean_error import _mean_error_update
from src.evaluation.torchmetrics.metric import Metric
from src.evaluation.torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from src.evaluation.torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from src.utilities.utils import get_logger

log = get_logger(__name__)

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MeanAbsoluteError.plot"]


class MeanError(Metric):
    r"""`Compute Mean Error`_ (Bias).

    .. math:: \text{MAE} = \frac{1}{N}\sum_i^N (\hat{y_i} - y_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mean_error`` (:class:`~torch.Tensor`): A tensor with the mean error over the state

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_error: Tensor
    total: Tensor

    def __init__(
        self,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.weights = weights
        self.add_state("sum_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_error, num_obs = _mean_error_update(preds, target, flatten=self.weights is None)
        self.sum_error = sum_error if self.total == 0 else self.sum_error + sum_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        if self.total == 0:
            log.warning(f"Did not record any data. Returning {tensor(0.0)}. Is this expected?")
            return tensor(0.0)
        sum_error = self.sum_error
        if self.weights is not None:
            sum_error = weighted_mean(sum_error, weights=self.weights)
        return sum_error / self.total

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting a single value
            >>> from src.evaluation.torchmetrics.regression import MeanAbsoluteError
            >>> metric = MeanAbsoluteError()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from src.evaluation.torchmetrics.regression import MeanAbsoluteError
            >>> metric = MeanAbsoluteError()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)

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

import torch
from torch import Tensor

from src.evaluation.torchmetrics.utilities.checks import _check_same_shape


def _mean_error_update(preds: Tensor, target: Tensor, flatten: bool) -> tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Error (Bias).

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    if flatten:
        preds = preds.view(-1)
        target = target.view(-1)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    sum_abs_error = torch.sum(preds - target, dim=0)
    return sum_abs_error, target.shape[0]

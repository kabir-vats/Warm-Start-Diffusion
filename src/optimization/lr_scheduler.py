"""
From:
 https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/ldm/lr_scheduler.py
"""

import math
import warnings
from typing import List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))

        self.last_lr = lr
        return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (
                self.cycle_lengths[cycle]
            )
            self.last_f = f
            return f


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, warmup_steps: int, max_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        warmup_steps (`int`):
            The number of steps for the warmup phase.
        max_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, warmup_steps: int, max_steps: int, num_cycles: int = 1, last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        warmup_steps (`int`):
            The number of steps for the warmup phase.
        max_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_dampened_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, warmup_steps: int, max_steps: int, num_cycles: float = 1, decay=0.9, last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        n_steps = float(max(1, max_steps - warmup_steps))
        progress = float(current_step - warmup_steps) / n_steps

        if progress >= 1.0:
            return 0.0

        cycle = (current_step - warmup_steps) // (n_steps / num_cycles)
        if cycle:
            ratio = decay**cycle
        else:
            ratio = 1

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1)))) * ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between
    warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and
    eta_min."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Maximum number of iterations for linear warmup
            max_steps (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        assert self.warmup_steps is not None, f"warmup_steps must be specified for {self.__class__.__name__}"
        assert self.max_steps is not None, f"max_steps must be specified for {self.__class__.__name__}"

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_steps:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_steps:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_steps - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_steps) % (2 * (self.max_steps - self.warmup_steps)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_steps - self.warmup_steps))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            / (
                1
                + math.cos(math.pi * (self.last_epoch - self.warmup_steps - 1) / (self.max_steps - self.warmup_steps))
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_steps:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / max(1, self.warmup_steps - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            for base_lr in self.base_lrs
        ]


class StableThenLinearDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, cooldown_frac: float = 0.45) -> None:
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.cooldown_frac = cooldown_frac
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            lrs = [self.last_epoch * (lr / self.warmup_steps) for lr in self.base_lrs]
            return lrs
        now_frac = min(0.9999, self.last_epoch / self.max_steps)
        assert 0 <= now_frac < 1
        lrf = 1.0
        if now_frac >= 1 - self.cooldown_frac:
            w = (1 - now_frac) / self.cooldown_frac
            lrf = w * 1.0 + (1 - w) * 0.1
        return [base_lr * lrf for base_lr in self.base_lrs]


class InverseSqrtLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, decay_step_start: int) -> None:
        self.warmup_steps = warmup_steps
        self.decay_step_start = decay_step_start
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        lrs = [base_lr / math.sqrt(max(self.last_epoch / self.decay_step_start, 1)) for base_lr in self.base_lrs]
        if self.last_epoch < self.warmup_steps:
            lrs = [self.last_epoch * (lr / self.warmup_steps) for lr in lrs]
        return lrs


def get_scheduler(optimizer, name, **kwargs):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == "cosine_warmup":
        return get_cosine_schedule_with_warmup(optimizer, **kwargs)
    if name == "cosine_hard_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, **kwargs)
    if name == "dampened_cosine_hard_restarts":
        return get_dampened_cosine_with_hard_restarts_schedule_with_warmup(optimizer, **kwargs)
    if name == "linear_warmup_cosine":
        return LinearWarmupCosineAnnealingLR(optimizer, **kwargs)
    if name == "linear_decay":
        return StableThenLinearDecayLR(optimizer, **kwargs)
    if name == "inverse_sqrt":
        return InverseSqrtLR(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler {name}")

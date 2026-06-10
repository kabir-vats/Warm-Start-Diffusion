"""
Timing utilities for profiling and logging function execution times to wandb.
"""

import functools
import time
from collections import defaultdict
from typing import Any, Callable, Optional

import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


# Global storage for timing accumulation
_timing_data = defaultdict(lambda: {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0})


def log_timing(
    name: Optional[str] = None,
    log_every_n: int = 100,
    rank_zero_only: bool = True,
) -> Callable:
    """
    Decorator to measure and log function execution time to wandb.

    Args:
        name: Name for the timing metric (default: function name).
        log_every_n: Log to wandb every N calls to reduce overhead.
        rank_zero_only: Only log from rank 0 in DDP.

    Example:
        @log_timing(name="getitem", log_every_n=50)
        def __getitem__(self, idx):
            return self.data[idx]
    """

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        stats = _timing_data[func_name]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Quick rank check
            if rank_zero_only and torch.distributed.is_initialized():
                if torch.distributed.get_rank() != 0:
                    return func(*args, **kwargs)

            # Check if we're in a DataLoader worker process - only time from worker 0
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None and worker_info.id != 0:
                return func(*args, **kwargs)

            # Time execution
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            # Update stats
            stats["count"] += 1
            stats["total"] += elapsed
            stats["min"] = min(stats["min"], elapsed)
            stats["max"] = max(stats["max"], elapsed)

            # Log to wandb periodically
            if wandb.run and stats["count"] % log_every_n == 0:
                mean = stats["total"] / stats["count"]
                log_dict = {
                    f"time/{func_name}/mean_ms": mean * 1000,
                    f"time/{func_name}/min_ms": stats["min"] * 1000,
                    f"time/{func_name}/max_ms": stats["max"] * 1000,
                }
                wandb.log(log_dict)
                # if "dataloader" in list(log_dict.keys())[0]: print(log_dict)

            return result

        return wrapper

    return decorator


def reset_timing_stats():
    """Reset all timing statistics."""
    _timing_data.clear()


class TimingCallback(Callback):
    """
    PyTorch Lightning callback for detailed training loop timing.

    Tracks:
    - Dataloader time (time between batches)
    - Backward pass time
    - Optimizer step time
    - Total batch time

    Usage:
        trainer = Trainer(callbacks=[TimingCallback(log_every_n_steps=50)])
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.training_step_end_time = None

    @rank_zero_only
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int):
        """Mark start of batch (includes dataloader time from previous batch)."""
        current_time = time.perf_counter()

        # Calculate dataloader time (time since last batch ended)
        if hasattr(self, "batch_end_time"):
            dataloader_time = current_time - self.batch_end_time
            if self.step_count % self.log_every_n_steps == 0:
                pl_module.log(
                    "time/dataloader_wait_ms",
                    dataloader_time * 1000,
                    on_step=True,
                    on_epoch=False,
                    rank_zero_only=True,
                )

        self.batch_start_time = current_time
        self.training_step_start_time = None  # Will be set by decorator

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        """Mark end of batch and log total batch time."""
        self.batch_end_time = time.perf_counter()
        self.step_count += 1

        if self.step_count % self.log_every_n_steps == 0:
            total_batch_time = self.batch_end_time - self.batch_start_time
            pl_module.log(
                "time/total_batch_ms",
                total_batch_time * 1000,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )

    @rank_zero_only
    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss: Any):
        """Mark start of backward pass."""
        self.backward_start_time = time.perf_counter()

    @rank_zero_only
    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        """Mark end of backward pass and log timing."""
        backward_time = time.perf_counter() - self.backward_start_time

        if self.step_count % self.log_every_n_steps == 0:
            pl_module.log(
                "time/backward_ms",
                backward_time * 1000,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )

    @rank_zero_only
    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any):
        """Mark start of optimizer step."""
        self.optimizer_start_time = time.perf_counter()

    @rank_zero_only
    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any):
        """Mark end of optimizer step and log timing."""
        if not hasattr(self, "optimizer_start_time"):
            return  # First step, no timing yet

        optimizer_time = time.perf_counter() - self.optimizer_start_time

        if self.step_count % self.log_every_n_steps == 0:
            pl_module.log(
                "time/optimizer_step_ms",
                optimizer_time * 1000,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )

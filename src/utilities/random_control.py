import os
import random
from contextlib import contextmanager
from enum import Enum
from typing import Optional

import numpy as np
import torch


class RNGMode(Enum):
    FIXED_GLOBAL = "fixed_global"  # Same RNG state for all calls
    FIXED_ITERATION = "fixed_per_iter"  # Same RNG state within each iteration
    RANDOM = "random"  # Different RNG state for each call


def set_seed(seed, device=None):
    """
    Sets the random seed for the given device.
    If using pytorch-lightning, preferably to use pl.seed_everything(seed) instead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        torch.cuda.manual_seed(seed)


@contextmanager
def controlled_rng(mode: RNGMode = RNGMode.RANDOM, seed: Optional[int] = None, iteration: int = 0):
    """
    Context manager for controlling PyTorch RNG state.

    Args:
        mode: RNG control mode from RNGMode enum
        seed: Base random seed to use
        iteration: Current iteration number (used for FIXED_ITERATION mode)
    """
    if isinstance(mode, str):  # Convert string to enum
        mode = RNGMode(mode)

    if mode == RNGMode.RANDOM:
        yield
        return  # Skip setting the RNG state

    seed = seed or int(os.environ.get("PL_GLOBAL_SEED", 7))
    use_cuda = torch.cuda.is_available()

    # Save current RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if use_cuda else None

    try:
        # Set appropriate seed based on mode
        if mode == RNGMode.FIXED_GLOBAL:
            seed_now = seed
        elif mode == RNGMode.FIXED_ITERATION:
            seed_now = seed + iteration
        else:
            raise ValueError(f"Invalid RNG mode: {mode}")

        torch.manual_seed(seed_now)
        if use_cuda:
            torch.cuda.manual_seed_all(seed_now)

        yield

    finally:
        # Restore original RNG state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)


# ----------------------------------------------------------------------------


class StackedRandomGenerator:
    """Wrapper for torch.Generator that allows specifying a different random seed for each sample in a minibatch."""

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


if __name__ == "__main__":
    from tqdm import tqdm

    x = "fixed_per_iter"
    print(RNGMode(x), x == RNGMode.FIXED_ITERATION, x == RNGMode.FIXED_ITERATION.value)
    progress_bar = tqdm([3, 4, 5], leave=False)
    for idx, i in enumerate(progress_bar):
        with controlled_rng(mode=RNGMode.FIXED_ITERATION, seed=42, iteration=idx):
            print(torch.randint(0, 100, (1,)))
    print("Done")
    for i in range(3):
        with controlled_rng(mode=RNGMode.FIXED_GLOBAL, seed=42):
            print(torch.randint(0, 100, (1,)))

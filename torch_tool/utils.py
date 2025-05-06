# torch_tools/utils.py

import torch
from typing import Callable, Any
import builtins
import numpy

def dry_run(fn: Callable[..., Any], enabled: bool = True) -> Callable[..., Any]:
    """
    Wrap fn so that if enabled=False it does nothing (returns None).
    Usage:
      real_fn = dry_run(my_fn, enabled=FLAGS.dry_run)
      real_fn(args...)
    """
    if not enabled:
        def no_op(*args, **kwargs):
            return None
        return no_op
    return fn

def get_best_device() -> torch.device:
    """
    Return the best available device:
    - tries CUDA
    - falls back to MPS
    - else CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def patch_numpy():
    """
    Monkey-patches numpy module to enable GPU-based overrides (like CuPy).

    Returns:
    --------
    contextlib.ContextDecorator
        Replaces numpy globally inside the context.
    """
    import contextlib

    @contextlib.contextmanager
    def _patch():
        original_numpy = builtins.__dict__.get("numpy", None)
        builtins.numpy = numpy  # Or CuPy in future
        yield
        if original_numpy is not None:
            builtins.numpy = original_numpy
        else:
            del builtins.numpy

    return _patch()
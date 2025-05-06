# torch_tool/context.py

import logging
import contextlib

import torch
from .device import set_device, get_device, clear_cuda_cache
from .precision import autocast
from .utils import patch_numpy

logger = logging.getLogger(__name__)

class DeviceContext:
    """
    A context manager to unify device selection, AMP context, auto‐tensorize
    and optional cache clearing in a single `with` block.

    Usage:
        with DeviceContext(device="cuda",
                           use_amp=True,
                           auto_tensorize=True,
                           clear_cache=False,
                           verbose=True) as dev:
            # inside: all torch ops run on `dev`, AMP enabled, NumPy patched
            ...

    Parameters:
    -----------
    device : str or torch.device, optional
        Target device ("cuda", "cuda:1", "cpu", None for auto‐detect).
    use_amp : bool
        If True and device is CUDA, enters torch.autocast for FP16.
    auto_tensorize : bool
        If True, patches NumPy to CuPy and converts Python/NumPy inputs via
        torch.as_tensor(..., device=dev).
    clear_cache : bool
        If True, runs torch.cuda.empty_cache() on enter (CUDA only).
    verbose : bool
        If True, logs entry/exit and active device info.

    Yields:
    -------
    torch.device
        The active device inside the context.
    """
    def __init__(
        self,
        device=None,
        use_amp=False,
        auto_tensorize=False,
        clear_cache=False,
        verbose=False,
    ):
        self.requested = device
        self.use_amp = use_amp
        self.auto_tensorize = auto_tensorize
        self.clear_cache = clear_cache
        self.verbose = verbose

    def __enter__(self):
        # save previous device
        self._prev = get_device()

        # select new device
        set_device(self.requested)
        dev = get_device()

        # optional cache clear
        if self.clear_cache and dev.type == "cuda":
            clear_cuda_cache()

        # optional NumPy→CuPy patch
        if self.auto_tensorize:
            self._np_patch = patch_numpy().__enter__()
        else:
            self._np_patch = None

        # optional AMP
        if self.use_amp and dev.type == "cuda":
            self._amp_cm = autocast().__enter__()
        else:
            self._amp_cm = None

        if self.verbose:
            logger.info(f"[DeviceContext] Entered on device {dev}")

        return dev

    def __exit__(self, exc_type, exc_val, exc_tb):
        # exit AMP
        if self._amp_cm is not None:
            autocast().__exit__(exc_type, exc_val, exc_tb)

        # exit NumPy patch
        if self._np_patch is not None:
            patch_numpy().__exit__(exc_type, exc_val, exc_tb)

        # restore previous device
        set_device(self._prev)
        if self.verbose:
            logger.info(f"[DeviceContext] Restored device {self._prev}")
        # don't suppress exceptions
        return False

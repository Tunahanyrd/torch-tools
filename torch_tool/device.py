# torch_tool/device.py

import torch
from typing import Optional, Callable, Any
import logging
import contextlib
from .precision import autocast
logger = logging.getLogger(__name__)

try:
    from torch._accelerator import get_accelerator
except ImportError:
    get_accelerator = None  # fall back if accelerator API is unavailable

# internal global state
_device: torch.device = None

def set_device(device: Optional[str] = None) -> torch.device:
    """
    Select and store the global torch.device.

    Parameters:
    -----------
    device : Optional[str]
        - If given, must be a string like "cuda", "cuda:1", "mps" or "cpu".
        - If None, we pick in order: torch._accelerator (cuda/mps/xpu) → torch.cuda → mps → cpu.

    Returns:
    --------
    torch.device
        The device that was set.
    """
    global _device
    if device:
        _device = torch.device(device)
    else:
        if get_accelerator is not None:
            acc = get_accelerator()
            _device = acc.device
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device

def get_device() -> torch.device:
    """
    Return the stored device, setting it automatically if needed.

    Returns:
    --------
    torch.device
        The current device.
    """
    if _device is None:
        return set_device(None)
    return _device

def clear_cuda_cache() -> None:
    """
    Clear CUDA memory cache if the current device is CUDA.
    """
    dev = get_device()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

def assert_free_memory(min_free_gb: float) -> None:
    """
    Raise if free GPU memory is below a threshold.

    Parameters:
    -----------
    min_free_gb : float
        Minimum required free memory in gigabytes.

    Raises:
    -------
    RuntimeError
        If on CUDA and available memory < min_free_gb.
    """
    dev = get_device()

    if dev.type != "cuda":
        return  # No-op if not on CUDA

    if not torch.cuda.is_available() or dev.index is None:
        raise RuntimeError("Invalid CUDA device for memory check.")

    props = torch.cuda.get_device_properties(dev)
    free = (props.total_memory - torch.cuda.memory_reserved(dev)) / 1e9
    if free < min_free_gb:
        raise RuntimeError(
            f"Need {min_free_gb:.2f} GB free, but only {free:.2f} GB available."
        )

def safe_run(
    fn,
    *args,
    max_cuda_retries: int = 2,
    use_autocast_on_oom: bool = False,
    **kwargs
):
    """
    Executes `fn(*args, **kwargs)` with enhanced safety and fallback logic:
    
    1. Tries running on the current CUDA device up to `max_cuda_retries` times.
    2. If `use_autocast_on_oom=True`, performs an additional retry using AMP (autocast) after OOM.
    3. If all retries fail, gracefully falls back to CPU and executes there.
    
    Notes:
    - Each attempt restarts the function from scratch (no internal state is preserved).
    - If AMP retry also triggers OOM, fallback to CPU is automatic.
    
    Parameters:
    -----------
    fn : callable
        The function to execute safely.
    max_cuda_retries : int
        Number of retry attempts on GPU (with cache clearing).
    use_autocast_on_oom : bool
        If True, enables one additional retry under AMP before falling back.
    *args, **kwargs : any
        Arguments to pass to the function.
    
    Returns:
    --------
    Any
        Whatever the function returns.
    
    Raises:
    -------
    RuntimeError
        If the function also fails on CPU, the error is raised.
    """  
    
    dev = get_device()
    last_exc = None

    # 1) Try GPU
    for attempt in range(1, max_cuda_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if dev.type == "cuda" and "out of memory" in str(e).lower():
                logger.warning(f"[safe_run] CUDA OOM on attempt {attempt}/{max_cuda_retries}, clearing cache and retrying...")
                clear_cuda_cache()
                last_exc = e
                continue
            raise

    # 2) Try gpu with the AMP (autocast)
    if use_autocast_on_oom and dev.type == "cuda":
        logger.info("[safe_run] OOM persisted: trying once more under autocast (FP16)…")
        with autocast():
            try:
                return fn(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("[safe_run] Still OOM under autocast; will fall back to CPU.")
                    last_exc = e
                else:
                    raise

    # 3) CPU fallback
    logger.info("[safe_run] Falling back to CPU execution.")
    with contextlib.ExitStack() as stack:
        stack.enter_context(set_device("cpu"))   # convert global device to CPU
        return fn(*args, **kwargs)


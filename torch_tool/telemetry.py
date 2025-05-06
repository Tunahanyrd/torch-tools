# torch_tool/telemetry.py

import time
import logging
import contextlib
from .device import get_device

_logger = logging.getLogger(__name__)

@contextlib.contextmanager
def telemetry(enabled: bool = False):
    """
    Context manager to log elapsed time and device.

    Parameters:
    -----------
    enabled : bool
        If True, logs "[telemetry] {seconds:.4f}s on {device}".
    """
    if not enabled:
        yield
        return
    t0 = time.time()
    dev = get_device()
    yield
    elapsed = time.time() - t0
    _logger.info(f"[telemetry] {elapsed:.4f}s on {dev}")

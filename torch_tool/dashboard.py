# torch_tool/dashboard.py

import time
import logging
import contextlib

_logger = logging.getLogger(__name__)

class Dashboard:
    """
    Live-dashboard metrics.
    - calls : int
    - total_time : float
    """
    calls = 0
    total_time = 0.0

@contextlib.contextmanager
def dashboard(enabled: bool = False):
    """
    Context manager to accumulate call count and time.

    Parameters:
    -----------
    enabled : bool
        If True, logs "[dashboard] calls={calls}, total_time={total_time:.4f}s".
    """
    if not enabled:
        yield
        return
    Dashboard.calls += 1
    t0 = time.time()
    yield
    dt = time.time() - t0
    Dashboard.total_time += dt
    _logger.info(
        f"[dashboard] calls={Dashboard.calls}, total_time={Dashboard.total_time:.4f}s"
    )

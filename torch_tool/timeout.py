# torch_tool/timeout.py

import threading
import functools
import time
from typing import Callable, Any

class TimeoutError(Exception):
    """Raised when a function call exceeds its timeout."""

def timeout(seconds: float):
    """
    Decorator factory to enforce a timeout on function execution.

    Parameters:
    -----------
    seconds : float
        Maximum allowed execution time in seconds.

    Returns:
    --------
    Callable
        A decorator that wraps a function and raises TimeoutError if
        it runs longer than `seconds`.
    """
    def decorator(fn: Callable[..., Any]):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = {}
            def target():
                try:
                    result['value'] = fn(*args, **kwargs)
                except Exception as e:
                    result['error'] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutError(f"{fn.__name__} timed out after {seconds}s")
            if 'error' in result:
                raise result['error']
            return result.get('value')
        return wrapper
    return decorator

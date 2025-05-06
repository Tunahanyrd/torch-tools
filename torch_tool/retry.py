# torch_tool/retry.py

import time
from typing import Callable, Any, Optional

def retry_loop(
    fn: Callable[[], Any],
    attempts: int = 0,
    delay: float = 0.0,
    on_exception: Optional[Callable[[Exception,int], None]] = None
) -> Any:
    """
    Execute a zero-arg function with retry logic.

    Parameters:
    -----------
    fn : Callable[[], Any]
        The function to call.
    attempts : int
        Number of retries after the first failure (total tries = attempts+1).
    delay : float
        Seconds to sleep between retries.
    on_exception : Optional[Callable[[Exception,int],None]]
        If provided, called as on_exception(exc, attempt_index).

    Returns:
    --------
    Any
        The return value of fn on success.

    Raises:
    -------
    Exception
        The last exception if all attempts fail.
    """
    last_exc = None
    for i in range(attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if on_exception:
                on_exception(e, i)
            if i < attempts and delay > 0:
                time.sleep(delay)
    raise last_exc

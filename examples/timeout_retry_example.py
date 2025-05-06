# examples/timeout_retry_example.py

import time
from torch_tool.core import timeout, retry_loop

@timeout(0.5)
def long_task():
    time.sleep(1.0)
    return "done"

def flaky_task():
    flaky_task.counter += 1
    if flaky_task.counter < 3:
        raise ValueError("try again")
    return "success"
flaky_task.counter = 0

def main():
    # Timeout demo
    try:
        print(long_task())
    except TimeoutError as e:
        print("Timeout caught:", e)

    # Retry demo
    result = retry_loop(flaky_task, attempts=5, delay=0.2,
                        on_exception=lambda e,i: print(f" retry #{i}: {e}"))
    print("Retry succeeded with:", result)

if __name__ == "__main__":
    main()

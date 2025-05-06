# tests/test.py

import time
import logging
import numpy as np
import torch
import sys
import os

# Make sure our logs print out
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_tool.core import (
    set_device, get_device, clear_cuda_cache, assert_free_memory, safe_run,
    autocast, grad_scaler,
    to_tensor, patch_numpy,
    timeout, TimeoutError,
    retry_loop,
    telemetry,
    dashboard,
    dry_run, get_best_device,
)



def test_device_selection():
    print("→ get_best_device():", get_best_device())
    print("→ set_device('cpu'):", set_device("cpu"))
    print("→ get_device():", get_device())
    # auto‐select
    dev = set_device(None)
    print("→ auto-set_device():", dev)

def test_clear_cache_and_free_mem():
    clear_cuda_cache()
    print("→ clear_cuda_cache() OK")
    try:
        # request an absurd amount to force error
        assert_free_memory(1e6)
    except RuntimeError as e:
        print("→ assert_free_memory(1e6) caught:", type(e).__name__)

def test_safe_run():
    def good(x, y):
        return x + y
    out = safe_run(good, 2, 3)
    print("→ safe_run(good,2,3):", out)
    def boom():
        raise RuntimeError("boom!")
    try:
        safe_run(boom)
    except RuntimeError as e:
        print("→ safe_run(boom) re-raises:", type(e).__name__)

def test_precision():
    dev = get_device()
    with autocast():
        a = torch.randn(100, device=dev)
        b = a * 0.5
        print("→ autocast: dtype after half‐matmul:", (a @ b).dtype)
    with grad_scaler() as scaler:
        assert scaler is not None
        print("→ grad_scaler: yielded scaler:", type(scaler).__name__)

def test_tensorize_and_patch_numpy():
    dev = get_device()
    for obj in (5, 3.14, [1,2,3], np.array([4,5,6], dtype=np.int64),
                torch.ones(2,2),):
        t = to_tensor(obj)
        print(f"→ to_tensor({type(obj).__name__}):", type(t), "on", t.device if hasattr(t, "device") else "")
    # patch_numpy (if cupy installed it proxies, else no‐op)
    with patch_numpy():
        import numpy as np2
        arr = np2.arange(3)
        print("→ patch_numpy: numpy.arange →", type(arr).__module__)

def test_timeout_decorator():
    @timeout(0.2)
    def slow():
        time.sleep(0.5)
        return "done"
    try:
        slow()
    except TimeoutError as e:
        print("→ timeout caught:", type(e).__name__)
    @timeout(1.0)
    def fast():
        return "ok"
    print("→ fast under timeout:", fast())

def test_retry_loop():
    counter = {"i":0}
    def flaky():
        counter["i"] += 1
        if counter["i"] < 3:
            raise ValueError("fail")
        return "success"
    out = retry_loop(flaky, attempts=5, delay=0.1, on_exception=lambda e,i: print(f" retry {i}"))
    print("→ retry_loop result:", out)

def test_telemetry_and_dashboard():
    with telemetry(True):
        time.sleep(0.1)
    with dashboard(True):
        time.sleep(0.05)
    with dashboard(True):
        time.sleep(0.01)
    print(f"→ Dashboard.calls={dashboard.__self__.calls if hasattr(dashboard,'__self__') else 'n/a'}")

def test_dry_run():
    def f(x): return x*2
    f1 = dry_run(f, enabled=True)
    print("→ dry_run(enabled=True):", f1(10))
    f2 = dry_run(f, enabled=False)
    print("→ dry_run(enabled=False):", f2(10))

if __name__ == "__main__":
    print("\n=== RUNNING ALL cuda_tools/core FEATURES ===\n")
    test_device_selection()
    print()
    test_clear_cache_and_free_mem()
    print()
    test_safe_run()
    print()
    test_precision()
    print()
    test_tensorize_and_patch_numpy()
    print()
    test_timeout_decorator()
    print()
    test_retry_loop()
    print()
    test_telemetry_and_dashboard()
    print()
    test_dry_run()
    print("\n=== ALL TESTS COMPLETE ===")

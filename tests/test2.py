# tests/test_comprehensive.py

import sys, os, time, logging

# —————————————————————————————————————————————————————————————
# 1) Make sure `torch_tool` is on our import path
# —————————————————————————————————————————————————————————————
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# —————————————————————————————————————————————————————————————
# 2) Import everything from core
# —————————————————————————————————————————————————————————————
from torch_tool.core import (
    # device & memory
    set_device, get_device, get_best_device,
    clear_cuda_cache, assert_free_memory,

    # precision
    autocast, grad_scaler,

    # tensorize
    to_tensor, patch_numpy,

    # control flow
    timeout, TimeoutError,
    retry_loop, safe_run,

    # observability
    telemetry, dashboard,

    # helpers
    dry_run,
)

import torch
import numpy as np

# Try import TF if available
try:
    import tensorflow as tf
    _TF = True
except ImportError:
    _TF = False

logging.basicConfig(level=logging.INFO)


def test_device_and_memory():
    print("\n[1] DEVICE & MEMORY")
    dev = get_best_device()
    print("→ get_best_device():", dev)

    set_device("cpu")
    print("→ set_device('cpu') →", get_device())

    set_device(None)  # auto-detect
    print("→ set_device(None) →", get_device())

    # invalid device should fallback or warn but not crash
    set_device("cuda:99") # -> Invalid device id
    print("→ set_device('cuda:99') →", get_device())

    clear_cuda_cache()
    print("→ clear_cuda_cache() OK")

    try:
        assert_free_memory(1e6)
    except RuntimeError as e:
        print("→ assert_free_memory(1e6) caught:", type(e).__name__)

    assert_free_memory(0)
    print("→ assert_free_memory(0) OK")


def test_precision():
    print("\n[2] PRECISION (AMP)")

    set_device(get_best_device())
    dev = get_device()

    x = torch.randn(8, 8, device=dev, dtype=torch.float32)
    y = torch.randn(8, 8, device=dev, dtype=torch.float32)

    # autocast on GPU
    if dev.type == "cuda":
        with autocast():
            z = x @ y
        print("→ autocast on CUDA dtype:", z.dtype)
    else:
        print("→ skipping autocast-on-CUDA, not on CUDA")

    # autocast on CPU (no-op)
    set_device("cpu")
    with autocast():
        z2 = x.cpu() @ y.cpu()
    print("→ autocast on CPU dtype:", z2.dtype)

    # grad_scaler
    set_device(get_best_device())
    with autocast(), grad_scaler() as scaler:
        print("→ grad_scaler provided:", type(scaler).__name__)


def test_tensorize_and_patch():
    print("\n[3] TENSORIZE & PATCH NUMPY")

    set_device(get_best_device())
    for obj in [
        42,
        3.14,
        [1, 2, 3],
        (4, 5, 6),
        np.array([7, 8, 9], dtype=np.int64),
        torch.ones(2, 2),
    ]:
        t = to_tensor(obj)
        print(f"→ to_tensor({type(obj).__name__}):", t.shape, "on", t.device, t.dtype)

    # TensorFlow conversion?
    if _TF:
        a = tf.constant([1, 2, 3], dtype=tf.float32)
        t2 = to_tensor(a)
        print("→ to_tensor(tf.Tensor):", t2.shape, "on", t2.device, t2.dtype)
    else:
        print("→ skipping tf.Tensor test (TensorFlow not installed)")

    # patch_numpy
    with patch_numpy():
        import numpy as np2
        arr = np2.arange(4)
        print("→ patch_numpy: numpy.arange() module =", arr.__class__.__module__)


def test_timeout_and_retry():
    print("\n[4] TIMEOUT & RETRY")

    @timeout(0.1)
    def will_timeout():
        time.sleep(0.2)

    @timeout(1.0)
    def will_pass():
        return "fast"

    try:
        will_timeout()
    except TimeoutError as e:
        print("→ timeout caught:", type(e).__name__)

    print("→ fast under timeout:", will_pass())

    # retry_loop
    counter = {"i": 0}
    def flaky():
        counter["i"] += 1
        if counter["i"] < 3:
            raise ValueError("fail")
        return "ok"
    out = retry_loop(flaky, attempts=5, delay=0.01,
                     on_exception=lambda e,i: logging.info(f"  [retry] attempt {i}: {e}"))
    print("→ retry_loop result:", out)


def test_safe_run():
    print("\n[5] SAFE_RUN (OOM → CPU fallback + AMP on OOM)")

    # simple pass
    set_device(get_best_device())
    def add(x,y): return x+y
    print("→ safe_run simple:", safe_run(lambda: add(1,2)))

    # stub that raises OOM on GPU, returns on CPU
    calls = []
    def oom_stub():
        calls.append(get_device().type)
        if get_device().type == "cuda":
            raise RuntimeError("CUDA out of memory: force OOM")
        return "cpu_ok"

    res = safe_run(
        oom_stub,
        max_cuda_retries=1,
        use_autocast_on_oom=True
    )
    print("→ safe_run OOM fallback:", res, "calls:", calls)


def test_observability():
    print("\n[6] TELEMETRY & DASHBOARD")

    with telemetry(True):
        time.sleep(0.05)

    for _ in range(2):
        with dashboard(True):
            time.sleep(0.03)

    print("→ telemetry/dashboard done (check logs above)")


def test_dry_run_and_context():
    print("\n[7] DRY_RUN & DeviceContext")

    def mul(x): return x*3

    f1 = dry_run(mul, enabled=True)
    print("→ dry_run enabled:", f1(5))

    f2 = dry_run(mul, enabled=False)
    print("→ dry_run disabled:", f2(5))

    # If you have a DeviceContext, test it:
    try:
        from torch_tool.context import DeviceContext
        orig = get_device()
        with DeviceContext(device="cpu"):
            print("→ DeviceContext inside:", get_device())
        print("→ DeviceContext restored:", get_device(), "== orig?", get_device()==orig)
    except ImportError:
        print("→ no DeviceContext to test (context.py missing)")


def test_compatibility():
    print("\n[8] COMPATIBILITY CHECKS")

    # mps (Apple) backend?
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("→ MPS available, get_best_device:", get_best_device())
    else:
        print("→ MPS not available, skipping")

    # multi-GPU?
    if torch.cuda.device_count() > 1:
        print("→ Multi-GPU count:", torch.cuda.device_count())
        print("→ get_best_device (MGPU):", get_best_device())
    else:
        print("→ Single-GPU or none, skipping MGPU test")


if __name__ == "__main__":
    print("\n=== COMPREHENSIVE TEST START ===")
    test_device_and_memory()
    test_precision()
    test_tensorize_and_patch()
    test_timeout_and_retry()
    test_safe_run()
    test_observability()
    test_dry_run_and_context()
    test_compatibility()
    print("\n=== COMPREHENSIVE TEST END ===\n")

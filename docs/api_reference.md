# API Reference

## Device Management (`device.py`)
- `set_device(device: Optional[str])`
  - Explicitly set the execution device (CUDA/CPU).
- `get_best_device() -> str`
  - Select the optimal available device automatically.
- `clear_cuda_cache()`
  - Clears the CUDA cache.
- `assert_free_memory(min_free_gb: float)`
  - Validates available GPU memory.

## Mixed Precision (`precision.py`)
- `autocast()`
  - AMP context manager for FP16 computations.
- `grad_scaler()`
  - Context manager providing PyTorch GradScaler instance.

## Tensorization (`tensorize.py`)
- `to_tensor(data, device)`
  - Converts various Python data structures to PyTorch tensors.
- `patch_numpy()`
  - Switches NumPy operations to CuPy if available.

## Retry and Timeout (`retry.py`, `timeout.py`)
- `retry_loop(fn, attempts: int, delay: float)`
  - Retries function execution on exception.
- `timeout(seconds: float)`
  - Decorator that limits function execution time.

## Telemetry and Dashboard (`telemetry.py`, `dashboard.py`)
- `telemetry`
  - Records function execution time and device usage.
- `dashboard`
  - Logs real-time function call metrics.

## Utilities (`utils.py`)
- `dry_run(fn, enabled: bool)`
  - Executes functions conditionally for testing without side-effects.
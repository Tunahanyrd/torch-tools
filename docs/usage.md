# Usage Guide for Torch Tool

This document covers comprehensive usage scenarios of Torch Tool components.

## Device Management
Easily manage device contexts explicitly:

```python
from torch_tool.device import set_device, get_best_device, clear_cuda_cache

device = get_best_device()
set_device(device)  # automatically picks GPU if available
clear_cuda_cache()  # free GPU cache
````

## Mixed Precision Training (AMP)

Torch Tool simplifies AMP usage:

```python
from torch_tool.precision import autocast, grad_scaler

with autocast():
    y_pred = model(x.float())  # Automatic FP16 on GPU

with grad_scaler() as scaler:
    loss = loss_fn(y_pred, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Automatic Tensor Conversion

Convert various types to tensors automatically:

```python
from torch_tool.tensorize import to_tensor

tensor = to_tensor([1, 2, 3], device="cuda")
tensor = to_tensor(np.array([1, 2, 3]), device="cuda")
```

## Timeout and Retry Handling

Ensure reliable execution:

```python
from torch_tool.timeout import timeout
from torch_tool.retry import retry_loop

@timeout(seconds=2)
def long_task():
    time.sleep(3)

retry_loop(long_task, attempts=3)
```

## Telemetry and Debugging

Real-time monitoring and performance tracking:

```python
from torch_tool.telemetry import telemetry
from torch_tool.dashboard import dashboard

@telemetry
@dashboard
def expensive_operation():
    time.sleep(1)
```

## Safe GPU-to-CPU Fallback

Automatic fallback if GPU runs out of memory:

```python
from torch_tool.device import safe_run

def memory_intensive_operation():
    tensor = torch.zeros((10**9, 10**9), device="cuda")
    return tensor.sum()

result = safe_run(memory_intensive_operation, max_cuda_retries=2)
```
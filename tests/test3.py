# tests/test_context.py

import torch
import numpy as np
import sys, os, time, logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_tool.core import *
def test_context_all():
    print("\n=== DeviceContext FULL TEST ===")

    with DeviceContext(device="cuda", use_amp=True, auto_tensorize=True, clear_cache=True, verbose=True) as dev:
        print("→ DeviceContext device:", dev)

        # NumPy array → tensor
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = torch.as_tensor(np_array, device=dev)
        print("→ as_tensor(np_array):", tensor, tensor.dtype, tensor.device)

        # AMP test
        a = torch.rand((2, 2), device=dev)
        b = torch.rand((2, 2), device=dev)
        result = a @ b
        print("→ AMP matmul dtype:", result.dtype)

    # Outside context: check device restored
    current = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print("→ After context, current device:", current)

if __name__ == "__main__":
    test_context_all()

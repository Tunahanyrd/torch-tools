# torch_tool/tensorize.py

import torch
import numpy as np
import contextlib
from typing import Any
from .device import get_device

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

def to_tensor(obj: Any) -> Any:
    """
    Convert Python/NumPy/CuPy objects to torch.Tensor on current device.

    Parameters:
    -----------
    obj : Any
        One of:
        - torch.Tensor → moved to device
        - int, float, np.generic → scalar tensor
        - list, tuple → tensor
        - np.ndarray → from_numpy
        - cupy.ndarray → via numpy() if CuPy installed

    Returns:
    --------
    torch.Tensor or original obj
    """
    dev = get_device()
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, (int, float, np.generic)):
        return torch.tensor(obj, device=dev)
    if isinstance(obj, (list, tuple)):
        return torch.tensor(obj, device=dev)
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            raise TypeError("object-dtype numpy arrays not supported")
        return torch.from_numpy(obj).to(dev)
    if _HAS_CUPY and isinstance(obj, cp.ndarray):
        return torch.from_numpy(cp.asnumpy(obj)).to(dev)
    return obj

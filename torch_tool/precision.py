# torch_tool/precision.py

import torch
import contextlib

@contextlib.contextmanager
def autocast():
    """
    Context manager for torch.autocast (automatic mixed precision).

    Yields:
    -------
    None
        Runs code block under autocast if CUDA is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        with torch.autocast(device_type=device.type):
            yield
    else:
        yield

@contextlib.contextmanager
def grad_scaler():
    """
    Context manager yielding a torch.amp.GradScaler if supported,
    otherwise falls back to torch.cuda.amp.GradScaler (for older PyTorch versions).

    Yields:
    -------
    torch.amp.GradScaler
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler()
    else:
        scaler = torch.cuda.amp.GradScaler()
    yield scaler

# examples/precision_example.py

import torch
from torch_tool.core import set_device, get_device, autocast, grad_scaler

def main():
    set_device("cuda")
    dev = get_device()

    x = torch.randn(1000, 1000, device=dev)
    y = torch.randn(1000, 1000, device=dev)

    # 1) simple autocast
    with autocast():
        z = x @ y  # runs in fp16
    print("z.dtype under autocast:", z.dtype)

    # 2) using grad_scaler in a toy training step
    model = torch.nn.Linear(1000, 1000, device=dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    with autocast(), grad_scaler() as scaler:
        pred = model(x)
        loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print("Training step with FP16 + GradScaler succeeded")

if __name__ == "__main__":
    main()

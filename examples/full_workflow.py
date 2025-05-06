# examples/full_workflow.py

import numpy as np
import torch
import time
from torch_tool.core import (
    set_device, to_tensor, retry_loop, timeout, telemetry, dashboard,
    autocast, grad_scaler, safe_run, get_best_device
)

# Pick the accelerator
set_device(get_best_device())

# Toy model & data
model = torch.nn.Linear(10, 1, device=get_best_device())
opt   = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

def train_step(batch):
    x, y = batch
    x, y = to_tensor(x).float(), to_tensor(y).float()
    with autocast(), grad_scaler() as scaler, telemetry(True), dashboard(True):
        def forward():
            pred = model(x)
            loss = loss_fn(pred, y)
            return loss, pred
        loss, pred = safe_run(lambda: forward())
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        return loss.item()

# Dummy data loader
data = [(np.random.rand(32,10), np.random.rand(32,1)) for _ in range(5)]

# Train for 3 epochs
for epoch in range(3):
    losses = retry_loop(lambda: train_step(data[epoch]), attempts=1)
    print(f"Epoch {epoch} avg loss {losses}")

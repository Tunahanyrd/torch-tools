# examples/tensorize_example.py

import numpy as np
import torch
from torch_tool.core import set_device, to_tensor, patch_numpy

def main():
    set_device("cuda")
    data_py = [1,2,3]
    data_np = np.arange(5).reshape(5,1)
    data_t = torch.eye(3)

    for obj in (data_py, data_np, data_t):
        t = to_tensor(obj)
        print(f"{type(obj).__name__} → tensor on {t.device}, shape {t.shape}")

    # patch numpy → cupy (if installed)
    with patch_numpy():
        import numpy as np2
        a = np2.linspace(0,1,10)  # now under cupy if available
        print("Under patch_numpy, module is", type(a).__module__)

if __name__ == "__main__":
    main()

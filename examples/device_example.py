# examples/device_example.py

import torch
from torch_tool.core import set_device, get_device, clear_cuda_cache, assert_free_memory, get_best_device

def main():
    # 1) pick best device
    best = get_best_device()
    print("Best device:", best)

    # 2) explicitly override
    set_device("cpu")
    print("Now using:", get_device())

    # 3) clear cache
    clear_cuda_cache()
    print("Cleared CUDA cache")

    # 4) check free memory (will error on absurd threshold)
    try:
        assert_free_memory(1000.0)
    except RuntimeError as e:
        print("Caught low-VRAM:", e)

if __name__ == "__main__":
    main()

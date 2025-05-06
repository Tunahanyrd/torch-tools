# examples/dry_run_example.py

from torch_tool.core import dry_run

def actual_compute(x):
    print("Computing on", x)
    return x * 2

def main():
    f = dry_run(actual_compute, enabled=False)
    print("dry_run disabled:", f(10))
    f = dry_run(actual_compute, enabled=True)
    print("dry_run enabled:", f(10))

if __name__ == "__main__":
    main()

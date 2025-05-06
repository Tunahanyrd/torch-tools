# examples/telemetry_dashboard_example.py

import time
from torch_tool.core import telemetry, dashboard

def main():
    with telemetry(True):
        time.sleep(0.2)

    for _ in range(3):
        with dashboard(True):
            time.sleep(0.1)

    print("Check your logs for telemetry/dashboard output.")

if __name__ == "__main__":
    main()

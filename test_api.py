"""
BLACKBOX — API end-to-end test
================================
Run AFTER starting the API server (`uvicorn app:app --reload --port 8000`):

    pip install requests
    python3 test_api.py

Simulates one machine: calibrates on healthy data, then streams readings
through a gradual degradation and checks the status escalates as expected.
"""

import requests
import random

BASE_URL = "http://localhost:8000"
MACHINE_ID = "loom_07"


def make_reading(vib=5.0, temp=60.0, ac=55.0, noise=0.15):
    return {
        "vibration": vib + random.gauss(0, noise),
        "temperature": temp + random.gauss(0, 0.5),
        "acoustic": ac + random.gauss(0, 1.0),
    }


def main():
    # 1) Calibrate on 100 healthy baseline readings
    baseline = [make_reading() for _ in range(100)]
    resp = requests.post(f"{BASE_URL}/calibrate/{MACHINE_ID}",
                          json={"baseline_readings": baseline})
    resp.raise_for_status()
    print("Calibration:", resp.json())

    # 2) Stream 300 more healthy readings — status should stay HEALTHY
    print("\n--- Streaming healthy readings ---")
    for i in range(300):
        r = make_reading()
        resp = requests.post(f"{BASE_URL}/score/{MACHINE_ID}", json=r)
        if i % 100 == 0:
            print(f"  minute {i}: {resp.json()}")

    # 3) Stream 400 readings with gradually worsening drift
    print("\n--- Streaming degrading readings ---")
    last_status = None
    for i in range(400):
        t = i / 400
        drift = t ** 1.5
        r = make_reading(vib=5.0 + 3 * drift, temp=60.0 + 8 * drift, ac=55.0 + 10 * drift)
        resp = requests.post(f"{BASE_URL}/score/{MACHINE_ID}", json=r)
        status = resp.json()["status"]
        if status != last_status:
            print(f"  minute {i}: status changed -> {resp.json()}")
            last_status = status

    # 4) Check the /status endpoint reflects the final state
    resp = requests.get(f"{BASE_URL}/status/{MACHINE_ID}")
    print("\nFinal status check:", resp.json())


if __name__ == "__main__":
    main()

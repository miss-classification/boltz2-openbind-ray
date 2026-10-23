"""Ask the autoscaler for N GPUs and time how long they take to arrive.

The pipeline's own clock starts after the data loads, so it never shows what a
cold cluster costs. On the workshop's critical path that is the number that
decides whether the GPU groups must be pinned before the session starts.

  python scripts/wait_gpus.py 21
"""

import sys
import time

import ray
from ray.autoscaler.sdk import request_resources

n = int(sys.argv[1])
timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1800

ray.init(address="auto")
t0 = time.time()
request_resources(bundles=[{"GPU": 1}] * n)
seen = -1
while True:
    have = int(ray.cluster_resources().get("GPU", 0))
    elapsed = time.time() - t0
    if have != seen:
        print(f"  {elapsed:6.0f}s  {have}/{n} GPUs", flush=True)
        seen = have
    if have >= n:
        print(f"SPIN-UP: {n} GPUs ready in {elapsed / 60:.1f} min", flush=True)
        break
    if elapsed > timeout_s:
        print(f"SPIN-UP: timed out at {have}/{n} GPUs after {elapsed / 60:.1f} min", flush=True)
        sys.exit(1)
    time.sleep(5)

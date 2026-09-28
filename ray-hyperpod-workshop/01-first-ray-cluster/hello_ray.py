"""First Ray program: see where Ray actually runs your code on the HyperPod cluster.

Runs three things and prints what happened:
  1. 40 small tasks        -> spread across the worker pods (CPU)
  2. one task per GPU      -> each lands on a different GPU node
  3. a stateful actor      -> lives on one node and keeps its state between calls
"""

import os
import socket
import time
from collections import Counter

import ray


@ray.remote(num_cpus=1)
def where_am_i(i: int) -> str:
    time.sleep(0.5)                       # long enough that Ray spreads the work out
    return socket.gethostname()


@ray.remote(num_gpus=1)
def gpu_probe() -> dict:
    return {
        "pod": socket.gethostname(),
        "ray_gpu_ids": ray.get_gpu_ids(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


@ray.remote(num_cpus=1)
class Counter_:
    def __init__(self):
        self.n = 0
        self.pod = socket.gethostname()

    def add(self, k: int) -> tuple[int, str]:
        self.n += k
        return self.n, self.pod


def main() -> None:
    ray.init()
    res = ray.cluster_resources()
    print(f"\nCluster: {int(res.get('CPU', 0))} CPUs, {int(res.get('GPU', 0))} GPUs, "
          f"{len([n for n in ray.nodes() if n['Alive']])} Ray nodes (head + workers)\n")

    t0 = time.time()
    hosts = ray.get([where_am_i.remote(i) for i in range(40)])
    print(f"1) 40 tasks finished in {time.time() - t0:.1f}s (40 x 0.5s of work) on these pods:")
    for pod, n in sorted(Counter(hosts).items()):
        print(f"     {pod:45s} {n:3d} tasks")

    n_gpus = int(res.get("GPU", 0))
    print(f"\n2) One task per GPU ({n_gpus}):")
    for r in ray.get([gpu_probe.remote() for _ in range(n_gpus)]):
        print(f"     {r['pod']:45s} ray_gpu_ids={r['ray_gpu_ids']} CUDA_VISIBLE_DEVICES={r['CUDA_VISIBLE_DEVICES']}")

    c = Counter_.remote()
    for k in (1, 2, 3):
        total, pod = ray.get(c.add.remote(k))
    print(f"\n3) Actor on {pod} kept its state across calls: total = {total} (expected 6)\n")


if __name__ == "__main__":
    main()

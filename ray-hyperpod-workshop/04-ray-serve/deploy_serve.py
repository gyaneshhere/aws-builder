"""Deploys serve_app on the running Ray cluster and exits; the application keeps running.

Run as a Ray job (deploy-serve.sh does this) so the code and dependencies travel with it.
"""

import argparse
import os
import time

import ray
from ray import serve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-fraction", type=float, default=0.25)
    a = ap.parse_args()

    ray.init()
    import serve_app  # imported after ray.init so MODEL_ID comes from the job's runtime env

    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(serve_app.build(a.gpu_fraction), name="sentiment", route_prefix="/classify", blocking=False)

    # Wait until the application reports RUNNING (model download happens here on first deploy).
    for _ in range(120):
        st = serve.status().applications.get("sentiment")
        if st and st.status == "RUNNING":
            print(f"Application 'sentiment' is RUNNING with model {os.environ.get('MODEL_ID')}")
            return
        time.sleep(5)
    raise SystemExit("Application did not reach RUNNING within 10 minutes; check the Serve tab in the dashboard")


if __name__ == "__main__":
    main()

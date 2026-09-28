"""Ray Serve sentiment API with request batching and replica autoscaling.

    POST /classify  {"text": "the match was great"}  ->  {"label": "POSITIVE", "score": 0.99, "replica": "..."}

MODEL_ID selects the model:
  * a Hugging Face model id (default in env.sh) -> transformers pipeline on a fraction of a GPU
  * "toy"                                        -> tiny keyword model, no download, no GPU
    (for laptops and clusters without internet access)
"""

from __future__ import annotations

import os
import socket

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

MODEL_ID = os.environ.get("MODEL_ID", "toy")


class _ToyModel:
    POS = {"good", "great", "love", "fast", "win", "awesome", "smooth"}
    NEG = {"bad", "slow", "hate", "lag", "lose", "broken", "crash"}

    def __call__(self, texts: list[str]) -> list[dict]:
        out = []
        for t in texts:
            words = set(t.lower().split())
            s = len(words & self.POS) - len(words & self.NEG)
            out.append({"label": "POSITIVE" if s >= 0 else "NEGATIVE", "score": 0.5 + min(abs(s), 5) / 10})
        return out


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 6,
        "target_ongoing_requests": 8,   # scale out when replicas average more than 8 in-flight requests
        "upscale_delay_s": 5,           # short delays so scaling is visible within a workshop
        "downscale_delay_s": 30,
    },
    max_ongoing_requests=32,
)
class Sentiment:
    def __init__(self) -> None:
        self.replica = f"{socket.gethostname()}/pid{os.getpid()}"   # unique per replica process
        if MODEL_ID == "toy":
            self.model = _ToyModel()
        else:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self.model = pipeline("sentiment-analysis", model=MODEL_ID, device=device)

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def predict(self, texts: list[str]) -> list[dict]:
        return self.model(texts)        # one forward pass for the whole batch

    async def __call__(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
            text = str(body["text"])[:2000]
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": 'send JSON like {"text": "..."}'}, status_code=400)
        result = await self.predict(text)
        return JSONResponse({**result, "replica": self.replica, "model": MODEL_ID})


def build(gpu_fraction: float = 0.25):
    """Each replica gets a quarter of a GPU, so one g5 node can host four replicas."""
    opts = {"num_gpus": gpu_fraction} if MODEL_ID != "toy" else {"num_cpus": 0.5}
    return Sentiment.options(ray_actor_options=opts).bind()


app = build()

# Module 4 — Ray Serve under load (30 min)

**Goal:** deploy a model API on the Ray cluster, load-test it, and watch Ray Serve autoscale replicas.

Wait for the Module 3 training job to finish first (`./02-distributed-training/job.sh status`), so the GPUs are free.

## The application

[serve_app.py](serve_app.py) is a sentiment API: `POST /classify {"text": "..."}`.

| Feature | Setting | Effect |
|---|---|---|
| Fractional GPUs | `num_gpus: 0.25` per replica | Four replicas share one GPU |
| Request batching | `@serve.batch(max_batch_size=32)` | Concurrent requests run in one forward pass |
| Autoscaling | `min_replicas 1`, `max_replicas 6`, `target_ongoing_requests 8` | Adds replicas when each averages more than 8 in-flight requests |
| Fast feedback | `upscale_delay_s 5`, `downscale_delay_s 30` | Scaling is visible within a few minutes |

The model defaults to `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (`SERVE_MODEL` in `env.sh`),
downloaded from Hugging Face on first start. **No internet from the nodes?** Use the built-in toy model:
`./04-ray-serve/deploy-serve.sh --toy` (no download, no GPU).

## 1. Deploy

```bash
./04-ray-serve/deploy-serve.sh
./01-first-ray-cluster/dashboard.sh start      # if tunnels are not running
curl -s 127.0.0.1:8000/classify -H 'Content-Type: application/json' -d '{"text": "the new patch is great"}'
```

```json
{"label": "POSITIVE", "score": 0.99, "replica": "ray-workshop-gpu-worker-abcde/pid812", "model": "distilbert/..."}
```

The deploy runs as a Ray job that exits once the app is RUNNING; the application keeps running on the cluster.

## 2. Load-test it

```bash
python3 04-ray-serve/loadgen.py                                  # default stages: 2, 16, 48, 4 clients
python3 04-ray-serve/loadgen.py --stages 1:30,64:120,2:90        # your own stages
```

`loadgen.py` needs only the Python standard library. Keep the dashboard's **Serve** tab open while it runs.

Output from a local run of the toy model on a laptop-sized Ray cluster (your numbers on GPUs will differ):

```text
stage clients   req/s   p50 ms   p95 ms   p99 ms  errors replicas seen replicas now
    1       1    51.4     18.9     22.2     28.2       0             1            1
    2      48    82.0    539.1    843.6   2286.8       0             6            6
    3       2    72.8     26.6     34.6     49.7       0             6            1
```

**What to notice**
- Under 48 clients, Serve scaled from 1 to the maximum of 6 replicas, with no errors.
- p99 is much higher than p50 during the ramp: requests queue while new replicas start.
- After load drops, replicas scale back down once `downscale_delay_s` passes.

## Try it

- Raise `max_replicas` in `serve_app.py` and redeploy: when do you run out of GPU slots?
- Set `max_batch_size=1` and compare throughput at 48 clients.

> **Going further:** Ray on HyperPod adds managed Karpenter autoscaling for Ray Serve, so nodes can scale with
> replicas. See [Ray on SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-ray.html).

✅ **Done when** `loadgen.py` shows `replicas now` rising under load and falling afterwards, with 0 errors.

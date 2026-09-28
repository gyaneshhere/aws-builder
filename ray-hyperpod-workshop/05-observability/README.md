# Module 5 — Observability (optional, 15 min)

**Goal:** see Ray metrics beyond the built-in dashboard.

## Option A: HyperPod Observability add-on (managed)

If your cluster has the **HyperPod Observability EKS add-on**, it collects metrics from Ray workloads and provisions
Grafana dashboards in Amazon Managed Grafana. Open your Amazon Managed Grafana workspace and find the Ray
dashboards while Module 2 (training) or Module 4 (load test) runs. Setup is described in
[Introducing new Ray capabilities on SageMaker HyperPod](https://aws.amazon.com/blogs/machine-learning/introducing-new-ray-capabilities-on-sagemaker-hyperpod/)
and [Ray on SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-ray.html).

## Option B: look at the raw metrics

Every Ray node exposes Prometheus metrics. From the head pod:

```bash
source env.sh
HEAD=$(kubectl -n $NAMESPACE get pod -l ray.io/cluster=$RAY_CLUSTER_NAME,ray.io/node-type=head -o name)
kubectl -n $NAMESPACE exec $HEAD -- ray status -v | head -30
kubectl -n $NAMESPACE exec $HEAD -- python -c \
  "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/metrics').read().decode())" \
  | grep -E "^ray_(node_|serve_)" | head -20
```

KubeRay configures port 8080 as the Ray metrics export port by default; if the command fails, check the
`metrics-export-port` in `ray start` on the head (`kubectl logs $HEAD | grep metrics`). You can scrape these with
your own Prometheus instead of the add-on.

## Questions to answer with the metrics

1. During Module 4's 48-client stage, how many requests per second did each replica handle?
2. During Module 3's recovery, how long was GPU utilization at zero on the rebooted node?

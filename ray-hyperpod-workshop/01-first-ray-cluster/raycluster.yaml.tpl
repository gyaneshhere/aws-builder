# RayCluster for the workshop. Rendered by create-cluster.sh from values in env.sh.
# Upstream KubeRay API, unchanged: HyperPod runs it as-is.
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ${RAY_CLUSTER_NAME}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: ray-hyperpod-workshop
spec:
  rayVersion: "${RAY_VERSION}"
  headGroupSpec:
    rayStartParams:
      num-cpus: "0"          # keep user work off the head so a worker-node fault never takes the head down
    template:
      spec:
        containers:
          - name: ray-head
            image: ${RAY_IMAGE}
            resources:
              requests: {cpu: "${HEAD_CPU}", memory: "${HEAD_MEMORY}"}
              limits:   {cpu: "${HEAD_CPU}", memory: "${HEAD_MEMORY}"}
            ports:
              - {containerPort: 6379, name: gcs-server}
              - {containerPort: 8265, name: dashboard}
              - {containerPort: 10001, name: client}
              - {containerPort: 8000, name: serve}
            volumeMounts:
              - {name: shared, mountPath: /fsx}
        volumes:
          - name: shared
            persistentVolumeClaim: {claimName: ${FSX_PVC}}
  workerGroupSpecs:
    - groupName: gpu
      replicas: ${GPU_WORKERS}
      minReplicas: 1
      maxReplicas: ${GPU_WORKERS}
      rayStartParams: {}
      template:
        spec:
          affinity:
            podAntiAffinity:       # spread workers across nodes so one node fault hits one worker
              preferredDuringSchedulingIgnoredDuringExecution:
                - weight: 100
                  podAffinityTerm:
                    topologyKey: kubernetes.io/hostname
                    labelSelector:
                      matchLabels: {ray.io/cluster: ${RAY_CLUSTER_NAME}, ray.io/group: gpu}
          containers:
            - name: ray-worker
              image: ${RAY_IMAGE}
              resources:
                requests: {cpu: "${WORKER_CPU}", memory: "${WORKER_MEMORY}", nvidia.com/gpu: 1}
                limits:   {cpu: "${WORKER_CPU}", memory: "${WORKER_MEMORY}", nvidia.com/gpu: 1}
              volumeMounts:
                - {name: shared, mountPath: /fsx}
          volumes:
            - name: shared
              persistentVolumeClaim: {claimName: ${FSX_PVC}}

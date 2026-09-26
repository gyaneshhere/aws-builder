# aws-builder

Two agents built with **Jev**, **Amazon Bedrock** and **Amazon Bedrock AgentCore**. Both take Jev beyond model
routing and put it where most agents spend time: making small, bounded decisions before any text gets written.

| Project | What it does | Jev decides | AgentCore services |
|---|---|---|---|
| [`alert-triage/`](alert-triage/) | Triages Amazon OpenSearch Service alerts from Grafana or PagerDuty into runbook tiers L1/L2/L3 and writes an investigation report | Runbook tier, blast radius, P(transient), P(needs a human now), rule mapping | Runtime, Gateway, Memory, Identity, Observability |
| [`regressionanalyzer/`](regressionanalyzer/) | Compares a load-test run against a Production window, metric by metric, and returns a PASS / WATCH / FAIL sign-off with a stakeholder narrative | Severity of each deviation (0–3), P(deviation is a test-setup artifact) | Runtime, Code Interpreter, Memory, Identity, Observability |

## The pattern

```
input ─▶ Jev: typed answers (choice / score / noul + confidence), ~100 ms
      ─▶ policy in code: floors, gates, fail-safes, final verdict
      ─▶ Bedrock model only when there is something worth writing about
```

Both projects split responsibility the same way:

- **Jev supplies judgment.** It reads state (the alert, or the computed statistics) and returns typed values
  your code branches on. It never does arithmetic and never has the final word.
- **Code owns consequences.** Runbook floors, SLO gates, confidence escalation, paging and verdicts are plain,
  unit-tested Python. If Jev is unreachable, both projects fail safe: L3 for alerts, statistics-only for regressions.
- **The LLM runs last and least.** Low-risk alerts and passing runs never reach a model. When one does run,
  it gets typed output (Pydantic) and must take its numbers from tools, not from its own head.

[Jev](https://jevtypesafeai.com) is a decision model from TypeSafe AI, called through `POST /api/v1/decide`.
The Jev client in both projects is adapted from
[kevinlupera/jev-strands-router](https://github.com/kevinlupera/jev-strands-router) (MIT).

## alert-triage

```
Grafana / PagerDuty alert ─▶ Memory (prior incidents) ─▶ Jev ─▶ policy ─▶ Strands agent
   L1 Nova 2 Lite │ L2 Nova Pro │ L3 Claude Sonnet 4.6
   tools via AgentCore Gateway ─▶ Lambda: read-only OpenSearch + CloudWatch diagnostics
```

- One Jev call answers four questions per alert, plus rule mapping for alerts not in the runbook.
- The tier picks both the Bedrock model and the tool allowlist. The Lambda issues signed `GET` requests only,
  to allowlisted domains and metrics.
- `watch` alerts (likely transient, low impact) and resolved alerts skip the LLM entirely.
- `triage/runbook.yaml` is starter content based on the AWS-recommended OpenSearch alarms; replace it with your rules.

```bash
cd alert-triage
python -m triage.evaluate --dry-run                      # Jev + policy against 12 labeled alerts
python -m deploy.setup_memory && python -m deploy.setup_gateway && python -m deploy.deploy_runtime
python -m deploy.invoke samples/alerts/cluster_red_prod.json
```

Details: [`alert-triage/README.md`](alert-triage/README.md)

## regressionanalyzer

```
LT + Prod series ─▶ Code Interpreter (exact stats) ─▶ Jev (severity, artifact) ─▶ policy ─▶ verdict
   flagged metrics only ─▶ Claude narrative, with follow-up Python in the same sandbox
   Memory: exact baselines (events) + past findings (semantic)
```

- The statistics kernel (warm-up trim, Mann-Whitney with Holm correction, Cliff's delta, bootstrap CI, SLO,
  shift vs baseline) is pure standard library and runs inside AgentCore Code Interpreter. Its p-values are
  cross-checked against SciPy in the tests.
- A run becomes the new baseline only on PASS or explicit `--accept-baseline`, so a regression never becomes normal.
- The included sample is synthetic, shaped like a 60K PSU sine-wave run with a warm-up reconnect burst.

```bash
cd regressionanalyzer
python -m lt_regress.cli samples/psu_sinewave_vs_prod.json --local-stats --decide-only --no-memory
python -m deploy.setup && python -m deploy.deploy_runtime
python -m deploy.invoke samples/psu_sinewave_vs_prod.json
```

Details: [`regressionanalyzer/README.md`](regressionanalyzer/README.md)

## Getting started

**Prerequisites**

- Python 3.12
- AWS credentials for a region with Amazon Bedrock AgentCore, and Bedrock model access for the models each
  project uses (Nova 2 Lite, Nova Pro and Claude Sonnet 4.6 for alert-triage; Claude Sonnet 4.6 for regressionanalyzer)
- A Jev API key
- An S3 bucket for the Runtime code package

**Setup (per project)**

```bash
cd <project>
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # JEV_API_KEY, AWS_REGION, AWS_PROFILE, DEPLOY_BUCKET, ...
python -m pytest -q         # offline tests: no AWS or Jev calls
```

Deploy scripts use boto3 directly (AgentCore direct code deploy, linux/arm64 wheels, no Docker). Each project
writes the IDs it creates to a local, git-ignored state file, and each has a `deploy/teardown.py`.

## Repository layout

```
alert-triage/
  triage/            decision layer, agent, pipeline, alert normalizer, runbook, CLI, evaluation
  lambda_tools/      Gateway Lambda target: 7 read-only diagnostic tools
  deploy/            setup_memory, setup_gateway, deploy_runtime, invoke, teardown
  iam/               trust and permission templates (Lambda, Gateway, Runtime)
  samples/           12 labeled alerts + single-alert files
  tests/
regressionanalyzer/
  lt_regress/        stats kernel, engines, decision layer, narrative, memory, pipeline, CLI
  deploy/            setup, deploy_runtime, invoke, teardown
  iam/               Runtime trust and permission templates
  samples/           synthetic comparison + generator
  tests/
```

## Security notes

- Secrets live in AgentCore Identity (Jev API key, Gateway OAuth client). Nothing sensitive is passed as a
  Runtime environment variable.
- Jev and the LLMs are not security boundaries. Permissions are enforced by IAM, the Lambda's GET-only code
  path and its allowlists.
- If your deploy profile cannot create IAM roles, create them from the `iam/` templates and set the role ARNs in `.env`.
- Never commit `.env` or the deploy state files; they can contain credentials for local testing.

## Caveats

- Pin `JEV_MODEL` to a concrete version before relying on decisions in production.
- Cost figures in the CLIs and evaluations use approximate list prices; check current
  [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) before quoting them.
- Both Runtimes use `PUBLIC` network mode because they call the Jev API.

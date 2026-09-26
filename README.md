# OpenSearch Alert Triage — Jev + Amazon Bedrock + AgentCore

Extends the "Jev as an LLM router" pattern
from *query complexity → model tier* to *OpenSearch alert → runbook tier (L1/L2/L3) → model + tool scope*.

```
Grafana / PagerDuty alert
  └─▶ AgentCore Runtime (main.py)
        ├─ AgentCore Memory     recall prior incidents for this domain
        ├─ Jev (1 call)         runbook_tier · blast_radius · likely_transient · needs_human_now [· matched_rule]
        ├─ Policy (code)        runbook floors · confidence escalation · page/notify/watch · fail-safe L3
        ├─ Strands agent        L1 Nova 2 Lite │ L2 Nova Pro │ L3 Claude Sonnet 4.6
        │    └─ AgentCore Gateway (MCP) ─▶ Lambda: read-only OpenSearch + CloudWatch diagnostics
        ├─ AgentCore Identity   Jev API key + Gateway OAuth token (no secrets in env)
        └─ AgentCore Memory     record decision + report
  Observability: OTEL spans (triage.tier, triage.action, triage.jev_*) → CloudWatch GenAI Observability
```

## Why this split

| Concern | Owner | Reason |
|---|---|---|
| Fuzzy judgments (tier, impact, flapping, urgency) | Jev | Typed answers + calibrated confidence, ~100 ms, input-token billing only |
| Consequences (floors, paging, fail-safe) | `triage/decision.py` | Auditable, unit-tested, cannot be argued out of by alert text |
| Evidence gathering + written report | Bedrock model sized by tier | Only alerts that need it pay for a frontier model |
| Cluster access | Gateway → Lambda | GET-only, domain allowlist, metric allowlist; Jev/LLM are not the security boundary |

Policy rules (all in code):
1. Start from Jev's `runbook_tier`; if its confidence < `TIER_CONFIDENCE_MIN` (0.60) escalate one tier.
2. `blast_radius` ≥ 2.5 → at least L3; ≥ 1.5 → at least L2.
3. Runbook `min_tier_prod` / `min_tier_nonprod` is a hard floor. Alerts not in the runbook → at least L2.
4. Action: nonprod never pages. Prod pages on L3 or `needs_human_now` ≥ 0.5; L2 notifies; L1 with
   `likely_transient` ≥ 0.7 and blast < 1 is **watch** (no LLM call); otherwise notify.
5. Resolved alerts → `record_only` (no Jev, no LLM). Jev error → L3.

## Layout

```
main.py                    Runtime entrypoint (BedrockAgentCoreApp)
triage/
  decision.py              Jev questions + policy  ← the core of the use case
  agent.py                 tier → model + tool allowlist, structured TriageReport
  pipeline.py              normalize → recall → decide → investigate → record
  alerts.py                Grafana / PagerDuty v3 / generic payload normalizer
  runbook.yaml             STARTER catalog (AWS-recommended alarms) — replace with your 19 rules
  jev_client.py  memory.py  credentials.py  config.py
  cli.py                   local run          evaluate.py   accuracy + cost benchmark
lambda_tools/handler.py    Gateway Lambda target (7 read-only tools + schemas)
deploy/                    setup_memory → setup_gateway → deploy_runtime → invoke / teardown
iam/                       trust + permission templates (${REGION} ${ACCOUNT_ID} ${DEPLOY_BUCKET})
samples/                   12 labeled alerts + single-alert files
tests/test_offline.py      18 offline tests (no AWS / Jev calls)
```

## 1. Local setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # set JEV_API_KEY, TRIAGE_DOMAINS, DEPLOY_BUCKET
aws sso login --profile dba-nonprod-aws-poweruser
python -m pytest -q               # 18 passed
```

## 2. Validate decisions before spending on Bedrock

```bash
python -m triage.evaluate --dry-run                 # Jev + policy only, all labeled samples
python -m triage.cli samples/alerts/cpu_spike_nonprod.json --decide-only --no-memory
```
Check `under_triage` = 0 first; tune thresholds in `.env` or runbook floors, not prompts.
Then add your own real alert payloads to `samples/labeled_alerts.json` with the tier your runbook says.

## 3. Deploy (run in order)

```bash
python -m deploy.setup_memory      # AgentCore Memory, semantic strategy, /incidents/{actorId}/
python -m deploy.setup_gateway     # Lambda + Cognito + Gateway + target + Identity providers
python -m deploy.deploy_runtime    # arm64 package → S3 → AgentCore Runtime (PUBLIC egress for Jev)
python -m deploy.invoke samples/alerts/cluster_red_prod.json
```
State (IDs, gateway URL, Cognito client) is written to `.triage_state.json` (git-ignored; contains the
Cognito client secret for local dev). `setup_gateway` prints the `GATEWAY_*` lines to add to `.env`
for local runs with tools.

Full benchmark (Bedrock cost, routed vs always-Sonnet):
```bash
python -m triage.evaluate --limit 6
```

## Prerequisites & known constraints

- **IAM:** PowerUser profiles normally cannot create roles. Either pre-create them from `iam/` and set
  `LAMBDA_ROLE_ARN`, `GATEWAY_ROLE_ARN`, `RUNTIME_ROLE_ARN`, or let the scripts try. An admin-supplied
  Gateway role must allow `lambda:InvokeFunction` on `os-triage-diagnostics` (in `iam/gateway-policy.json`).
- **OpenSearch FGAC:** map the Lambda role to a read-only backend role (e.g. `cluster_monitor` +
  `indices_monitor`) or `_cat`/`_cluster` calls return 403. The tools report 403s as data gaps.
- **VPC domains:** `setup_gateway` reads the domain's subnets/SGs and puts the Lambda there. The SG must
  allow 443 from itself. CloudWatch calls from a VPC Lambda need a NAT or a `monitoring` VPC endpoint.
- **Bedrock:** enable model access for Nova 2 Lite, Nova Pro and Claude Sonnet 4.6 in the region; the
  IDs are inference profiles (`us.` prefix), as verified in the source article repo.
- **Jev:** Runtime needs internet egress (`PUBLIC` network mode). Pin `JEV_MODEL` to a concrete version
  before production so decision boundaries don't shift. Costs in `agent.py` are approximate list prices.
- **Observability:** enable CloudWatch Transaction Search once per account to see Runtime traces.
- **Change management:** Lambda/Gateway/Runtime are new infra; stage in dba-nonprod against
  `metrictestopenai` first and route prod rollout through TOBOR as usual.
- **Wiring alerts in:** this project stops at `invoke_agent_runtime`. Connecting Grafana/PagerDuty
  webhooks (e.g. API Gateway → Lambda → Runtime) and posting the report back as a PagerDuty note or
  ServiceNow work note is the natural next step.

## Teardown

```bash
python -m deploy.teardown
```

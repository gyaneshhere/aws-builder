# LT vs Production Regression Analyzer — Jev + Amazon Bedrock + AgentCore

Compares a load-test run against a Production window, metric by metric, and produces a
sign-off verdict (PASS / WATCH / FAIL) plus a stakeholder narrative for anything flagged.

```
comparison JSON / CSV
  └─▶ AgentCore Runtime (main.py)
        ├─ AgentCore Memory             last accepted LT baseline for this test (exact numbers)
        ├─ AgentCore Code Interpreter   stats_kernel.py: warm-up trim, medians, Mann-Whitney,
        │                               Cliff's delta, bootstrap CI, SLO, vs-baseline
        ├─ Jev (batched)                per metric: severity score 0-3 + P(test-setup artifact)
        ├─ Policy (code)                Holm correction, practical thresholds, SLO gates, verdict
        ├─ Claude on Bedrock            narrative for flagged metrics only; can run follow-up
        │                               Python in the SAME Code Interpreter session
        └─ AgentCore Memory             record findings; new baseline only on PASS (or explicit accept)
  Identity: Jev API key · Observability: lt.verdict / lt.regressions / lt.jev_* span attributes
```

## Who decides what

| Question | Decided by | Why |
|---|---|---|
| Is the change real? | Statistics in Code Interpreter | Exact, reproducible, sandboxed; Jev never does arithmetic |
| Does it matter for sign-off? | Jev severity score | Context-dependent: SLO headroom, metric meaning, run notes |
| Is it a test-setup artifact? | Jev noul | Reads run notes (sizing differences, warm-up, restarts) |
| Flagged or not | `lt_regress/decision.py` | Auditable rules; Jev can't hide an SLO breach or promote noise |
| What to tell stakeholders | Claude | Only for flagged metrics; numbers must come from the stats or the sandbox |

Policy per metric:

1. An SLO breach in the worse direction is always a regression.
2. Otherwise the change must be Holm-significant (family-wise α = 0.05), have a bootstrap 95% CI that excludes
   zero, exceed the metric's `min_pct`, and go in the worse direction. Only then is Jev's judgment used:
   artifact ≥ 0.6 → **explained**; severity ≥ 2 → **regression**; severity confidence < 0.5 → **regression**
   (flagged for review); otherwise **watch**.
3. A significant, practical change in the better direction is an **improvement**; everything else is **noise**.
4. Jev unavailable → statistics-only rule (every real, practical, worse change is a regression).
5. Verdict: FAIL if any regression, WATCH if any watch, else PASS. PASS runs get a template summary, no LLM call.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                       # JEV_API_KEY at minimum
python -m pytest -q                        # 17 offline tests (Mann-Whitney cross-checked against SciPy)
python samples/generate_sample.py          # regenerate the synthetic sample (deterministic)

# Stats + Jev + policy only, kernel in-process:
python -m lt_regress.cli samples/psu_sinewave_vs_prod.json --local-stats --decide-only --no-memory
# Full run: kernel in AgentCore Code Interpreter + Claude narrative:
python -m lt_regress.cli samples/psu_sinewave_vs_prod.json
# Your own exports (wide CSV: timestamp,metricA,metricB,...) + a meta JSON:
python -m lt_regress.cli --csv lt.csv prod.csv meta.json
```

The sample is **synthetic**, shaped like a 60K PSU sine-wave run:

- p99 latency and DB timeouts regress → expected **regression**
- Aurora DB load rises, but LT runs smaller instances with reader autoscaling (stated in `notes`) → expected **explained**
- writer CPU drops → expected **improvement**
- a 1,500–3,000 connection reconnect burst in the first 5 LT minutes is removed by the warm-up window → **noise**

Its `expected` block is what the tests assert. Without Jev, the statistics-only fallback reports DB load as a
regression; the artifact judgment is the part Jev adds.

## Input

See `lt_regress/models.py`. Per metric: `name`, `unit`, `direction` (`higher_is_worse` | `lower_is_worse` |
`two_sided`), optional `slo`, `min_pct` (smallest median change that matters, default 5), `owner`, and the
`lt` / `prod` sample series. Per side: `interval_seconds`, `warmup_minutes`. Put anything reviewers should
know about the setup (instance sizes, autoscaling min/max, restarts) in `notes`: Jev reads it.

## Deploy

```bash
python -m deploy.setup             # Memory (events + semantic strategy) and Identity provider for the Jev key
python -m deploy.deploy_runtime    # arm64 package → S3 → AgentCore Runtime (direct code deploy, no Docker)
python -m deploy.invoke samples/psu_sinewave_vs_prod.json
python -m deploy.invoke run.json --accept-baseline        # human override: make this run the baseline
python -m deploy.teardown --keep-memory
```

Code Interpreter uses the AWS-managed `aws.codeinterpreter.v1` sandbox; no setup needed. The runtime role
template is in `iam/`; if your profile cannot create IAM roles, have it pre-created and set `RUNTIME_ROLE_ARN`.

## Notes

- Baselines are stored as exact JSON events, not semantic memories, because extraction would paraphrase
  numbers. A run becomes the baseline only on PASS or `--accept-baseline`, so a regression never becomes normal.
- Samples are treated as independent per interval. Autocorrelated series make p-values optimistic; the
  practical threshold, the CI check and the Holm correction limit false alarms from that.
- Pin `JEV_MODEL` before using verdicts for sign-off. Narrative cost uses approximate Sonnet list prices.
- The runtime uses `PUBLIC` network mode because it calls the Jev API.

# Integration tests

These tests talk to **real** SageMaker endpoints and **cost real money**
to run. They are intentionally skipped by default — the standard
``pytest`` invocation runs unit tests only, never these.

## How to run locally

```bash
export RUN_INTEGRATION_TESTS=1
export AWS_REGION=us-east-1
export AWS_PROFILE=<profile-with-sagemaker-and-marketplace-access>

# Per-model endpoints — set whichever ones you have deployed.
# Any unset (endpoint_name, arn) pair causes its tests to skip.
export JINA_TEST_V3_ENDPOINT=<endpoint-name>
export JINA_TEST_V3_ARN=<model-package-arn>

export JINA_TEST_V4_ENDPOINT=<endpoint-name>
export JINA_TEST_V4_ARN=<model-package-arn>

export JINA_TEST_CLIP_V2_ENDPOINT=<endpoint-name>
export JINA_TEST_CLIP_V2_ARN=<model-package-arn>

export JINA_TEST_RERANKER_ENDPOINT=<endpoint-name>
export JINA_TEST_RERANKER_ARN=<model-package-arn>

export JINA_TEST_READER_LM_V2_ENDPOINT=<endpoint-name>
export JINA_TEST_READER_LM_V2_ARN=<model-package-arn>

pytest tests/integration/ -v
```

## How to run in CI

The ``Integration tests`` workflow is opt-in:

- Trigger manually from the Actions tab (``workflow_dispatch``), or
- Wait for the weekly Monday-06:00 cron.

It expects these GitHub repository secrets / variables:

- ``AWS_INTEGRATION_TEST_ROLE_ARN`` — secret. An IAM role the workflow
  assumes via OIDC; needs ``sagemaker:InvokeEndpoint`` + the Marketplace
  permissions described in the notebooks.
- ``JINA_TEST_<MODEL>_ENDPOINT`` / ``JINA_TEST_<MODEL>_ARN`` — repository
  variables (not secrets — the ARN and endpoint names are not sensitive).
  Any unset pair skips its model's tests.

The workflow runs on a dedicated ``integration`` GitHub environment so the
secrets are scoped down and the run can be required to wait for human
approval if you want it.

## What the suite covers

For each supported model family, one happy-path inference call:

- **embed** (v3, v4, clip-v2) — text input, verifies shape of returned
  ``embedding`` list.
- **rerank** (v2/v3, m0) — ``["doc1", "doc2"]`` + query, verifies the
  ranked-result list.
- **read** (ReaderLM-v2) — a short prompt, verifies the response is
  parseable JSON.

The suite intentionally does NOT exercise ``create_endpoint`` or
``delete_endpoint``. Spinning endpoints up and down per run costs ~5 min
and ~$5 a pop; we assume the endpoints under test are already deployed
and warm.

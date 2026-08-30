# LiteLLM Operator Notes

The local proxy exposes four task-specific aliases. Keep model credentials and proxy keys in
the shell environment; never add them to these files.

## Keys

`config.yaml` reads `LITELLM_MASTER_KEY`. The controller reads the key named by
`operator-models.toml`, currently `LITELLM_API_KEY`. For a single local operator, export both
names with the same value before starting the proxy and controller. A missing key produces a
401 response. A `Database not connected` response from `/spend/logs` occurs after successful
authentication and means spend persistence is not configured.

Do not connect LiteLLM to the research controller database or enable prompt storage. Model
responses can contain source code and experiment context.

## Role Mapping

| Role | Alias backend | Effort | Reason |
| --- | --- | --- | --- |
| Orchestration | `gpt-5.6-terra` | medium | Low-latency bounded routing |
| Research | `gpt-5.6-sol` | medium | Scientific proposal quality |
| Implementor | `gpt-5.6-luna` | medium | Code generation with bounded reasoning cost |
| Validator | `gpt-5.6-sol` | high | Adversarial scientific and policy review |

The August 30, 2026 local smoke benchmark reported zero monetary cost for all aliases because
the `chatgpt` provider uses subscription quota. Representative results were:

| Role | Latency | Completion tokens | Reasoning tokens |
| --- | ---: | ---: | ---: |
| Orchestration | 2.0 s | 46 | 19 |
| Research | 4.7 s | 109 | 0 |
| Implementor, high | 26.3 s | 1,404 | 1,196 |
| Implementor, medium | 8.7 s | 429 | 332 |
| Validator | 3.9 s | 124 | 50 |

These are smoke measurements, not stable pricing or performance guarantees. Runtime logs are
the authority for request latency, tokens, quota windows, retries, and response selection.

## Health And Restart

LiteLLM 1.89.5's generic `/health` probe sends scalar Responses API input to the `chatgpt`
provider, which rejects it with `Input must be a list`. Real controller calls through
`/v1/chat/completions` succeed, so use a small structured completion as the deployment smoke
test instead of `/health` for these aliases.

Changes to `config.yaml`, including reasoning effort, require restarting the proxy:

```bash
uv run litellm --config config/litellm/config.yaml --host 127.0.0.1 --port 4000
```

Changes to `operator-models.toml` are loaded when a controller process starts or resumes.

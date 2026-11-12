# Native model providers — experimental fork

Based on Codex `rust-v0.159.2` (`ff6aec96948b70d94983af2641a6b67c94faeff5`),
the same release pinned by dzimplei. This branch is a protocol experiment,
not a replacement for the current production engine.

## What works

`wire_api` accepts `responses` (unchanged), `anthropic_messages`, and
`chat_completions`. Native HTTP requests enter Codex's existing inference
client; the same agent loop executes tools and owns approvals, sandboxing,
transcripts, interruption and inline compaction. No second planner or tool
executor runs beside Codex. Responses WebSockets stay limited to Responses.

The adapters preserve system/developer instructions, text, image blocks,
namespaced function tools, custom/freeform tools through a JSON input wrapper,
tool results and provider-signed reasoning replay. Native tool identities stay
stable across reordered inventories. Custom grammars remain in the tool
instructions; the actual tool executor still validates the input.

Tool calls are buffered until a valid terminal event, and the entire call set
is checked before any executable item is emitted. Malformed arguments,
unoffered names, duplicate call IDs, truncated streams, incomplete output and
refused tool batches fail instead of becoming work. A natural-language refusal
remains the provider's answer; no provider fallback is added. HTTP credentials
and network policy still go through Codex's existing provider/HTTP client.

Anthropic cache reads and cache writes count toward input usage. Missing Chat
Completions usage stays unknown, rather than pretending the request used zero
tokens. Streams have backpressure, cancellation and a 32 MiB input bound.
Opaque signed reasoning is stored in the existing rollout's provider-state
field with a protocol marker. This encoding is not additional encryption.
Reasoning from another protocol is rejected rather than rewritten.

## Explicit provider configuration

Use a separate `CODEX_HOME` for this experiment. Set credentials in the
experiment's environment; do not put their values in config or commit them.
These are API-key examples, not Claude subscription or Copilot sign-in flows.

```toml
model_provider = "anthropic-native"
model = "YOUR_EXACT_ANTHROPIC_MODEL_ID"
model_context_window = 200000 # replace with the chosen model's documented limit
model_auto_compact_token_limit = 150000
model_reasoning_effort = "none"
web_search = "disabled"

[features]
memories = false
tool_search = false
code_mode = false
image_generation = false

[model_providers.anthropic-native]
name = "Anthropic native experiment"
base_url = "https://api.anthropic.com/v1"
wire_api = "anthropic_messages"
requires_openai_auth = false
supports_websockets = false
env_http_headers = { "x-api-key" = "ANTHROPIC_API_KEY" }
```

For a Chat Completions provider, use `wire_api = "chat_completions"`, its own
`base_url`, model and context limits, and `env_key = "PROVIDER_API_KEY"` instead
of Anthropic's `env_http_headers`. The obsolete `wire_api = "chat"` spelling
still fails; the new protocol must be selected explicitly. No model-family
name heuristic changes the transport or account.

## Current boundaries

- Anthropic output is capped at 8,192 tokens. Explicit reasoning effort enables
  its budget-style thinking with 4,096 thinking tokens; leave effort `none` for
  models that require another thinking mode. Model-specific output budgets,
  adaptive thinking and prompt-cache policies need a subsequent capability pass.
- Structured outputs, native audio/video, provider server tools, encrypted
  OpenAI history, tool search and OpenAI-hosted memory endpoints are not ported.
  Unsupported request shapes fail before sending; they are not silently dropped.
- Gemini's native GenerateContent API is not implemented. Compatible endpoints
  can use Chat Completions, but that is not native Gemini parity.
- Claude subscription OAuth and Copilot account entitlement are separate from
  protocol support. Existing production login/router paths remain separate.
- Live paid providers, cross-provider history migration and desktop release
  packages are not validated by the local fixtures.

## Reproduce

Build the fork with its pinned Rust toolchain. Run targeted tests through the
repository's `just test` command for `codex-api`, `codex-model-provider-info`
and `codex-config`. Then:

```sh
node scripts/native-provider-smoke.mjs /absolute/path/to/codex
```

The smoke test creates a disposable home/workspace and two local API fixtures.
For each protocol, the real Codex agent receives its ordinary shell tool,
executes `printf native-provider-ok`, returns the actual result to the model,
and produces the final answer. It uses no paid credentials or personal chats.
The standard sandbox and approval settings are not changed in the engine.

## Design references (ideas, not implementation ports)

- OpenCode: explicit provider protocols and capability-aware model definitions.
  https://github.com/anomalyco/opencode/tree/dev/packages/ai/src/providers
- OpenClaude: provider routing/runtime metadata and compatibility boundaries.
  https://github.com/Twigpine/openclaude
  Its NOTICE identifies proprietary-derived source; no implementation was copied.
- Hermes: provider adapters and provider-aware context/model metadata.
  https://github.com/NousResearch/hermes-agent/tree/main/agent
- VS Code Copilot Chat: distinct Messages and Responses endpoint builders,
  signed thinking replay and cache-aware usage.
  https://github.com/microsoft/vscode-copilot-chat/blob/main/src/platform/endpoint/node/messagesApi.ts
- Anthropic's native streaming contract:
  https://platform.claude.com/docs/en/build-with-claude/streaming

All adapter code in this branch was written for this fork. Existing upstream
licenses and notices remain intact.

## Review staging before landing

This draft exceeds the upstream small-change guideline. Review the native
request codec (`native/request.rs`) first, then streaming normalization and
its HTTP fixtures (`native/mod.rs`, `native_tests.rs`), then provider/config/core
selection and the end-to-end smoke. These are separable landing stages; the
request codec is the smallest coherent first stage. No production branch is
being replaced by the combined experiment.

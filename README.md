# lite-model

The shared model-call layer for the lite modules (lite-rag, lite-mem, llm-steward). Call sites name a
**role**; configuration maps each role to a model on a **backend**; every reply says **which model actually
answered**, read from the response, never from config.

- Backends: `ollama`, `openai` (any OpenAI-compatible endpoint), `anthropic`, `gateway`
  ([llm-gateway](https://github.com/ArthurGao/llm-gateway)), `litellm` (LiteLLM Proxy). Name as many as you
  need: two OpenAI-compatible vendors side by side are two named backends.
- No fallback chain: a failed call raises after retries on the same model. Nothing switches models behind
  your back.
- Validated at startup: an unknown role, a missing key, a parameter the backend cannot take, a routing
  alias on a role that must stay pinned: `build()` refuses the config before any call.
- Routing by difficulty is optional and lives in a LiteLLM Proxy (`deploy/litellm/`). lite-model never
  imports the `litellm` package.

The design and its decisions: [`docs/lite-model-design.md`](docs/lite-model-design.md).

## Install

lite-model is not on PyPI. Each release attaches a wheel to its
[GitHub Release](https://github.com/ArthurGao/lite-model/releases); depend on it by URL:

```toml
[project]
dependencies = ["lite-model"]          # or "lite-model[anthropic]" for the Anthropic backend

[tool.uv.sources]
lite-model = { url = "https://github.com/ArthurGao/lite-model/releases/download/v0.2.1/lite_model-0.2.1-py3-none-any.whl" }
```

Python 3.12. Pinned: `openai==3.46.0`, `pydantic==2.14.3`, `pydantic-settings==2.16.2`, `typer==0.27.3`;
extra `anthropic==2.6.0`.

## Quick start

```python
from pydantic import BaseModel
from lite_model import Message, ModelsConfig, RolePolicy, RoleSpec, build

# The host declares its roles; only `routable` ones may use a routing alias.
ROLES = RoleSpec({
    "answer": RolePolicy(routable=True),
    "graph": RolePolicy(routable=False),   # part of an index's identity: always pinned
})

config = ModelsConfig(                      # defaults, then per-role overrides
    backend="ollama", model="qwen3:8b", reasoning_effort="none",
    roles={"graph": {"model": "qwen3:1.7b", "max_output_tokens": 2048}},
)
model = build(config, ROLES)                # raises ConfigError on a bad config, before any call

reply = model.text("answer", [Message("system", "Be brief."), Message("user", "Hi?")])
reply.text
reply.provenance.tag                        # "ollama:qwen3:8b" - the model that answered
reply.usage                                 # Usage(input=…, output=…, reasoning=…, cost_usd=…)

class City(BaseModel):
    name: str
    population: int | None = None

city = model.structured("answer", [Message("user", "Largest city in New Zealand?")], City).value
```

`build_async()` returns the same interface with `async` methods. `model.target("graph")` gives a role's
resolved target (backend, model, endpoint) before any call, e.g. to record it in an index's identity.

Hosts embed `ModelsConfig` in their own settings, so variables carry the host's prefix
(`LITERAG_LLM__MODEL=…`). A host's own defaults go on a subclass; whatever a config gives is merged on
top, so adding a backend or tuning one role never drops a default it did not name:

```python
class LLMSettings(ModelsConfig):
    backend: str = "ollama"
    model: str = "qwen3:8b"
    host_backends = {"ollama": BackendConfig(type="ollama", base_url="http://gpu:11434/v1")}
    host_roles = {"graph": RoleConfig(reasoning_effort="none", max_output_tokens=2048)}
```

A backend or role given under an existing name changes it field by field (`{"ollama": {"base_url": …}}`;
a role's `null` removes the host's value, so the role takes the top-level one); only a backend naming
another `type` starts afresh.

`litemodel dump-config` prints the merged config read from `LITEMODEL_*` and each role's resolved
target; keys appear as variable names only, and nothing is called.

## Configuration

```yaml
backend: ollama                 # required, no default: the default backend (a name under `backends`)
model: qwen3:8b                 # required, no default
temperature: null               # null: the backend's default (Ollama 0.0; not sent to OpenAI)
max_output_tokens: null         # null: not sent (required for anthropic)
reasoning_effort: null          # none | minimal | low | medium | high; null: not sent
timeout_s: 120
connect_timeout_s: 10
max_retries: 1                  # the same model only
roles:                          # per role; a field left out keeps the default
  answer: {backend: litellm, model: smart-router}
  extract: {backend: deepseek, model: deepseek-chat}
backends:                       # the five built-ins exist (type = name); add more by name
  deepseek: {type: openai, base_url: "https://api.deepseek.com/v1", api_key_env: DEEPSEEK_API_KEY}
  local: {type: openai, base_url: "http://127.0.0.1:11434/v1", keyless: true}   # no key, nothing Ollama-specific
forbid_channels: []             # tests and CI: [subscription, api]
prices:                         # optional, USD per million tokens, by "backend:answering-model"
  "deepseek:deepseek-chat": {input_per_mtok: 0.27, output_per_mtok: 1.10}
```

| type | protocol | channel | key | notes |
| --- | --- | --- | --- | --- |
| `ollama` | OpenAI SDK → `http://127.0.0.1:11434/v1` | local | none | sends `max_tokens`; temperature 0 by default |
| `openai` | OpenAI SDK, any compatible endpoint | api | `OPENAI_API_KEY` | sends `max_completion_tokens` |
| `anthropic` | Anthropic SDK (extra) | api | `ANTHROPIC_API_KEY` | needs `max_output_tokens`; `base_url` is the host root, without `/v1`; no `temperature` |
| `gateway` | OpenAI SDK → llm-gateway `:8787` | subscription | none | a parameter the gateway reports as ignored raises `IgnoredParameter` |
| `litellm` | OpenAI SDK → LiteLLM Proxy `:4400` | api | `LITELLM_API_KEY` | answering model from `x-litellm-model-name`; retries off per request |

Keys are only ever read from environment variables (`api_key_env` names the variable).

## Provenance

Every reply carries `Provenance(role, backend, requested, answered, channel, routed, via)`:

- `answered` is read from the response. Through LiteLLM it comes from the `x-litellm-model-name` header (a
  plain alias answers with the alias in the body); when the answering model cannot be read, the call raises
  `ProvenanceUnknown` instead of recording a guess.
- `tag` is `backend:answered` (the backend's configured name), or `backend:requested?` when an endpoint
  does not say. A host that stores `provider:model` today keeps its format by naming the backend after its
  provider (`backends: {openai: {type: ollama}}`).
- `via` keeps the endpoint's own provenance headers (LiteLLM's deployment id and alias group, the gateway's
  backend).

## Routing

Routing by request difficulty is done by a LiteLLM Proxy auto router, not by lite-model. Declare the alias
on the `litellm` backend (`router_aliases`, default `[smart-router]`) and use it only on routable roles;
`build()` refuses it on a pinned role, and on a gateway (which cannot say which model answered). The proxy
(pinned image, signature checked by cosign before it starts, Ollama-only tiers by default) is in
[`deploy/litellm/`](deploy/litellm/README.md).

## Tool turns

For a host that runs its own agent loop: one request in which the model may call tools. lite-model sends the
turn and reads it back; it never executes a tool, loops or keeps a session.

```python
from lite_model import Message, ToolResult, ToolResults, ToolSpec

clock = ToolSpec("get_time", "Current local time.", {"type": "object", "properties": {}})
history = [Message("user", "What time is it?")]

reply = model.turn("answer", history, [clock])      # a TurnReply
if reply.stop == "tool_calls":
    results = [ToolResult(c.id, run(c.name, c.arguments)) for c in reply.turn.tool_calls]
    history += [reply.turn, ToolResults(tuple(results))]   # the turn goes back unchanged
    reply = model.turn("answer", history, [clock])
print(reply.turn.text)
```

- `reply.turn.blocks` are in the endpoint's order: `Text`, `ToolCall` and `Opaque` (Anthropic thinking blocks,
  replayed verbatim to Anthropic and left out of a request to an OpenAI-compatible endpoint).
- `reply.stop == "length"`: the turn was cut off. Do not execute its calls; answer each with an error result.
  `text()` raises `Truncated` here; `turn()` returns, because the calls need their results.
- `ToolCall.arguments is None`: the endpoint's JSON did not parse (`raw_arguments` has the text).
- Hooks do not see `turn()`. `ToolModel` / `AsyncToolModel` are the protocols; `FakeModel.turn` scripts
  `AssistantTurn`s and records `turn_calls`.

## Embeddings (0.5.0)

```python
roles = RoleSpec({"extract": RolePolicy(), "embed": RolePolicy(kind="embed")})
config = ModelsConfig(backend="ollama", model="qwen3:8b", roles={"embed": {"model": "bge-m3:567m"}})
reply = build(config, roles).embed("embed", ["Ada lives in Wellington."])
reply.vectors, reply.dim, reply.provenance.tag      # [[...]], 1024, "ollama:bge-m3:567m"
```

- An embed role is never routable (an index is bound to its model) and takes no chat settings.
- Backends: `ollama`, `openai` (any OpenAI-compatible `/embeddings`), `litellm`; not `gateway` or `anthropic`.
- Texts go in batches of 64; the vectors come back in input order. Hooks do not see `embed()`.
- Tests: `FakeModel(vectors={"embed": hash_vectors(8)})`; calls are recorded in `embed_calls`.

## Errors

```
ModelError
├─ ConfigError            build(): every problem listed at once
├─ CallFailed             transport / HTTP / SDK error, after retries on the same model
├─ Truncated              cut at the output limit
├─ EmptyReply             no usable answer (empty, refused, filtered)
├─ BadStructuredReply     does not validate against the schema
├─ IgnoredParameter       the endpoint dropped a parameter lite-model sent
└─ ProvenanceUnknown      the answering model cannot be read
```

Messages carry the role, `backend:model` and the exception type only, never upstream text; the original
exception is chained. An error raised after the endpoint answered carries the call's `usage`.

## Usage and cost

`Usage(input, output, reasoning, cost_usd)`. Cost comes from the endpoint's own figure (LiteLLM's
`x-litellm-response-cost`), else from `prices` for the model that answered, else `None` — never estimated.
A subscription call costs `None`.

## Concurrency

```python
from lite_model import Bounded

bounded = Bounded(model, concurrency=4)      # at most 4 calls in flight, from any thread
headers = bounded.map(lambda chunk: bounded.text("contextual", prompt(chunk)).text, chunks)
```

`AsyncBounded` does the same for `AsyncModel`; `map()` keeps input order. Around a built model the slot covers only
the request itself: a hook's hit does not wait for a slot, and a hook may call the model through the same
`Bounded`. `map()` runs each item in a copy of the caller's context, so `call_context()` reaches the hooks.

## Hooks

```python
class Cache:
    name = "cache"
    def before(self, call):          # call.role, call.target, call.messages, call.params, call.schema_hash
        return stored.get(key(call))  # a reply ends the call: no request, reply.cached is True
    def after(self, call, reply):    # fresh replies only
        stored[key(call)] = reply

registration = model.hooks.register(Cache())
registration.remove()
```

Hooks see a read-only copy of the request actually sent. A hit keeps its original provenance; its `usage` is
what the hit saved. A hook that raises fails the call; `after` should not raise (the reply is lost).
`Call`, `Hook`, `AsyncHook` and `Hooks` live in `lite_model.hooks`. On a model from `build_async()` a hook may be an
`AsyncHook` (`async def before / after`, awaited, so a database lookup does not block the event loop); a sync model
refuses one with `TypeError`.

A `litellm` reply that the Proxy served from its own cache (non-empty `x-litellm-cache-key`) is `cached=True` too:
`cached` means nothing was spent again. Unlike a hook's hit, such a reply does reach `after`, so a hook that counts
spending should skip a reply whose `cached` is true.

Per-call values a hook needs (a cache's scope, whether this call site may be cached) travel in a context,
not as parameters, so `Model` stays `text(role, messages)` / `structured(role, messages, schema)`:

```python
from lite_model import call_context

with call_context(scope="acme/u42", cache=True):
    model.text("answer", messages)            # the hook reads call.context["scope"]
```

## Testing a host

```python
from lite_model.testing import FakeModel

fake = FakeModel(answer="hi", extract=[{"facts": []}, RuntimeError("down")])
fake.text("answer", messages).text   # "hi"; fake.calls records every call
```

`AsyncFakeModel` is the async twin. To check the exact request a host sends, give `build()` a recording
transport: `build(config, roles, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))`.

## Developing

```bash
uv sync --all-extras
uv run pytest                               # unit + contract; fakes only, sockets refused
uv run pytest -m ollama                     # opt-in: the local Ollama
uv run pytest -m litellm                    # opt-in: the local LiteLLM Proxy (Ollama aliases only)
uv run ruff check && uv run ruff format --check && uv run mypy
```

Agents: [`AGENTS.md`](AGENTS.md). License: Apache-2.0.

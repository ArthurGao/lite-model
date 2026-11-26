# lite-model Design Document

Status: S5 done (v0.1.0, 2026-11-18); host migration done (2026-11-19); 0.3.0 released (2026-11-20: `AsyncHook`, `Bounded` slot wraps only the request, `map` carries context, Proxy cache headers, measured against the Proxy and passed); 0.4.0 released (2026-11-21: `turn()`, see §3.1, used by lite-agent). Stages and acceptance are in §14.

## 1. Overview

lite-model is the **shared model-call layer** of the lite modules (lite-rag, lite-mem, llm-steward, and later lite-cache):
a call site names only a role (`answer`, `extract`, ...), and the config points the role at a model on a
backend. lite-model sends the request, handles the parameter differences between vendors, gets a structured result
back, and records **who actually answered**. Routing (choosing a model automatically by request difficulty) is done by the optional
LiteLLM Proxy. lite-model treats it as just one backend and adds, on the client side, the guards and
provenance records that routing needs.

Names: repo `lite-model`, Python package `lite_model`, CLI `litemodel`, environment variable prefix `LITEMODEL_`.

### Origin

- The discussion of 2026-11-18: each of the 4 modules has a nearly identical LLM client (lite-rag `llm.py` 178 lines, lite-mem `llm.py`
  198 lines, llm-steward `llm/client.py`, lite-eval `judges/client.py`). All of them handle provider differences, provenance records,
  error wrapping and usage counting, and adding a router would make each of them add it again.
- The owner's decision (§15): LiteLLM is used only in Proxy mode, as an optional backend; the lite modules can be used alone, and without LiteLLM they use the original
  way of configuring.
- Starting code: lite-mem `src/lite_mem/llm.py` (role, strict schema, Anthropic forced tool, provenance taken from the response)
  and lite-rag `src/lite_rag/llm.py` (the `PROVIDERS` data table, `BoundedCalls`, an error on empty replies).
- Rule sources: `KB/global/llm-app-architecture.md` §1 (role layer), §3.2 (no fallback chain), §10 (pluggable
  extension of shared modules); `KB/global/tech-selection/01b-llm-gateway-litellm.md` (LiteLLM selection, not verified).

### Goals

1. One interface: `text()` and `structured(schema)`, sync and async.
2. Role config: defaults + per-role overrides; validated at startup, and a config error refuses to start.
3. Backends: `ollama`, `openai` (any OpenAI-compatible endpoint), `anthropic`, `gateway` (llm-gateway),
   `litellm` (LiteLLM Proxy). The differences between vendors are data, not branches.
4. Provenance: every answer records the role, the requested alias, **the model that actually answered**, the backend, and the channel (local / API / subscription).
5. Routing support: only a role declared routable can use a routing alias; for a routing alias the answer records the actual model.
6. Errors: no fallback; one error hierarchy; truncation, an empty reply and a result that does not match the schema all raise; error messages carry no keys.
7. Usage: input / output / reasoning tokens, also recorded for failed calls; optionally compute cost from a price table.
8. Extension point: hooks before and after a call (lite-cache uses them for caching); registration is revocable and addressable by name.
9. Test tool: `FakeModel` (scripted replies per role, records the requests it receives).

### Non-goals (v1)

- **Does not wrap the `litellm` Python package**, and does not run routing in the process. Routing is done only in the LiteLLM Proxy (reasons in §7.1).
- No fallback to another model; no retries other than a simple retry of the same model.
- Does not execute tools, run an agent loop or manage session state. Since 0.4.0 it offers "one call round with tools" (§3.1): it is only one request and
  reply, and the loop stays with the caller (lite-agent).
- Does not manage prompt templates (they stay in each module's `prompts/`), and is not the cache itself (that is left to lite-cache).
- No streaming (none of the 4 modules uses it).
- Embedding is left to v2 (lite-rag and lite-mem each have an embedding client, see the open questions in §15).
- lite-eval's judge is not migrated (decided by the owner on 2026-11-21): it deliberately uses httpx2 directly and goes to Ollama's native `/api/chat`; the judge,
  as a measuring tool, must be fixed and isolated (lite-eval design §9).

## 2. Position and call chain

```
app (lite-rag / lite-mem / llm-steward / later lite-cache)
  └─ lite-model                         client lib: role -> target, provenance, guards
       ├─ HTTP → Ollama /v1             local models
       ├─ HTTP → OpenAI / Anthropic     API billing
       ├─ HTTP → llm-gateway :8787      subscription profiles (claude-sub-opus / -haiku, gpt-sub), owner-only
       └─ HTTP → LiteLLM Proxy :4400    unified aliases, auto router (optional)
                   └─ may forward to llm-gateway or vendors
```

- **Dependency direction**: lite modules → lite-model. lite-model depends on no lite module, and not on llm-gateway or LiteLLM
  code either; they are only HTTP endpoints.
- **llm-gateway does not import lite-model**: the gateway is a server (protocol conversion, subscription backends) and lite-model is a client;
  importing each other would let a change to the client interface drag in the server. The two sides only duplicate one small table of parameter differences, which is acceptable.
- **lite-cache plugs in from the other side**: lite-model provides the hook protocol (§10) and lite-cache implements it; lite-model does not depend on lite-cache.
- **Optional**: when `gateway` / `litellm` is not configured, lite-model connects directly to Ollama / OpenAI / Anthropic, the same as
  each module's behavior today. When `litellm` is configured and the proxy is unavailable it **raises an error** and does not fall back to another backend (no silent fallback).

## 3. Interface

```python
@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass(frozen=True)
class Provenance:
    role: str                 # the caller's role, e.g. "answer"
    backend: str              # the configured backend's name (e.g. ollama, deepseek), §4
    requested: str            # the model / alias / profile the call asked for
    answered: str | None      # the model that actually answered; None = the endpoint did not say
    channel: Channel          # local | api | subscription
    routed: bool              # requested was a routing alias
    via: Mapping[str, str]    # the endpoint's own provenance headers (allow-listed per profile, lower-cased)

    @property
    def tag(self) -> str: ... # "backend:answered" (or "backend:requested?" when answered is None)

@dataclass(frozen=True)
class Usage:
    input: int | None
    output: int | None
    reasoning: int | None
    cost_usd: float | None    # from the price table or the proxy's cost header; None = unknown

@dataclass(frozen=True)
class TextReply:
    text: str
    provenance: Provenance
    usage: Usage

@dataclass(frozen=True)
class StructuredReply[T: BaseModel]:
    value: T
    provenance: Provenance
    usage: Usage

class Model(Protocol):
    def text(self, role: str, messages: Sequence[Message]) -> TextReply: ...
    def structured[T: BaseModel](self, role: str, messages: Sequence[Message], schema: type[T]) -> StructuredReply[T]: ...

class AsyncModel(Protocol):
    async def text(self, role: str, messages: Sequence[Message]) -> TextReply: ...
    async def structured[T: BaseModel](self, role: str, messages: Sequence[Message], schema: type[T]) -> StructuredReply[T]: ...

def build(config: ModelsConfig, roles: RoleSpec, *, http_client: httpx2.Client | None = None) -> LiteModel: ...  # a Model with .hooks
def build_async(config: ModelsConfig, roles: RoleSpec, *, http_client: httpx2.AsyncClient | None = None) -> AsyncLiteModel: ...
```

- The caller passes only a role and messages; the model, endpoint, key and parameters all come from the config.
- `http_client` (added in S1): replaces the SDK's transport. The contract tests used when migrating hosts (§12.2) use it to record the request body the real SDK sends;
  it can also be used to connect a proxy. lite-model's own contract tests do the same (`tests/upstream.py`).
- Each module may keep its own `LLM` Protocol (lite-rag rule 1) and wrap lite-model with a thin adapter; it may also depend
  on `lite_model.Model` directly. Each module chooses for itself when migrating (§12).
- `RoleSpec`: the host module declares which roles it has, which are routable (§7.3), and which need structured output. Example:

  ```python
  ROLES = RoleSpec({
      "answer": RolePolicy(routable=True),
      "contextual": RolePolicy(routable=False),   # part of the index identity
      "graph": RolePolicy(routable=False),
  })
  ```

### 3.1 One round with tools: `turn()` (0.4.0)

For lite-agent's loop: the model itself decides whether to call a tool and which one. lite-model handles only the request and reply of **this one round**;
executing tools, deciding whether to continue, and saving the history are the caller's job (the §1 non-goals are unchanged).

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]          # JSON Schema of the input

@dataclass(frozen=True)
class Text:
    text: str

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] | None    # None: the endpoint's JSON text did not parse to an object
    raw_arguments: str | None = None       # the text as received, only when it did not parse

@dataclass(frozen=True)
class Opaque:                              # content only its own protocol understands
    protocol: Literal["openai", "anthropic"]
    data: Mapping[str, Any]

@dataclass(frozen=True)
class AssistantTurn:                       # goes back into the history unchanged
    blocks: tuple[Text | ToolCall | Opaque, ...]   # in the endpoint's order
    # .text, .tool_calls: read-only views

@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False

@dataclass(frozen=True)
class ToolResults:                         # every result of one assistant turn
    results: tuple[ToolResult, ...]

TurnMessage = Message | AssistantTurn | ToolResults

@dataclass(frozen=True)
class TurnReply:
    turn: AssistantTurn
    stop: Literal["end", "tool_calls", "length"]
    provenance: Provenance
    usage: Usage
    cached: bool = False

class ToolModel(Protocol):
    def turn(self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]) -> TurnReply: ...

class AsyncToolModel(Protocol):
    async def turn(self, role: str, messages: Sequence[TurnMessage], tools: Sequence[ToolSpec]) -> TurnReply: ...
```

- **A separate protocol**, with no methods added to `Model` / `AsyncModel`: the adapters and fakes that hosts implement for `Model` keep satisfying the original
  protocol. `LiteModel` / `AsyncLiteModel` both implement it. `text()`, `structured()`, `Message` and their request bodies are unchanged.
- **`blocks` keep the order the endpoint gave**, and the caller puts `reply.turn` back into the history as it is.
- **`stop`**: `length` = cut off by the output limit; otherwise it is `tool_calls` if there is a call (also when the endpoint reports `stop` but carries calls),
  and `end` if there is none.
- **Truncation does not raise `Truncated`** (unlike `text()`): the calls carried by this round cannot be executed, but before the next request every call needs
  a result, so the caller needs their ids. `Truncated` is raised only for "truncated with neither text nor a call".
- **No error when the arguments are not a valid JSON object**: `arguments=None` and `raw_arguments` is the original text; the caller returns it to the
  model as invalid arguments. When this round is replayed, the original text is sent as it is.
- Refusal, content filter, `pause_turn`, an unknown stop reason, neither text nor a call: `EmptyReply` (with usage).
  A reply with only thinking is also `EmptyReply`. For Chat Completions, `finish_reason` accepts only `stop`, `tool_calls`,
  `function_call`, `length` and absent; any other value is unusable, the error says "unexpected", and it does not echo the endpoint's string. (On such endpoints `text()` /
  `structured()` treat only `content_filter` as unusable; that behavior is unchanged.)
- These types are frozen, but only one level deep: `Mapping` fields keep the object the caller passed in (the same as `Provenance.via`); do not
  change them after handing them over.
- A `ToolResults` with no results produces no message (Anthropic rejects a user message with empty content).
- A system `Message` in the middle of the history: Chat Completions sends it in place; Anthropic merges it into the `system` field like the one at the start (following
  what `text()` does).
- `tool_choice` is not sent (the endpoint defaults to auto). Newer Anthropic models reject a forced `tool_choice` (400). `parallel_tool_calls` is not
  exposed. When `tools` is empty the `tools` field is not sent.
- **Does not go through hooks** (§10). The `Bounded` slot still applies as usual.
- When the backend type does not support tools (`BackendProfile.supports_tools`, §5), `turn()` raises `ConfigError` and sends no request.

The mapping of the two protocols (the contract test `tests/contract/test_turns.py` checks it field by field):

| | Chat Completions (ollama / openai / gateway / litellm) | Anthropic Messages |
| --- | --- | --- |
| `ToolSpec` | `{type: function, function: {name, description, parameters}}` | `{name, description, input_schema}` |
| `AssistantTurn` in the history | One assistant message: `content` = the text (`null` if none), `tool_calls[]` (`arguments` is JSON text) | One assistant message, content blocks in order: `text`, `tool_use`, and this protocol's `Opaque` as it is; an empty text block is not sent |
| `ToolResults` in the history | One `{role: tool, tool_call_id, content}` per result; the protocol has no error flag, so `is_error` is not sent | **One** user message, with one `tool_result` per result (`is_error: true` is carried only on an error) |
| Reply → blocks | `content` → `Text`; the entries in `tool_calls` with `type == function` → `ToolCall` | `text` → `Text`; `tool_use` → `ToolCall`; the rest (`thinking`, `redacted_thinking`, ...) → `Opaque("anthropic", received fields)`: the SDK object's `to_dict()`, i.e. the fields and values the endpoint sent (including explicit nulls and extra fields); it is a dict round trip, not a preservation of the original bytes |
| Truncation | `finish_reason: length` | `stop_reason: max_tokens` / `model_context_window_exceeded` |

**Replay rule for `Opaque`**: if `protocol` is the same as the target backend's protocol it is sent back as it is; if different it is not sent (the other protocol has no place for it).
It is not deleted when the model changes within the same protocol: Anthropic's API itself drops the thinking blocks that the target model cannot read, without billing; deleting them
ourselves could trigger a signature / order error. The Chat Completions backend in 0.4.0 produces no `Opaque` (see §15).

**lite-model does not handle Anthropic's "preserved thinking" history check for the caller** (newer models: when `system`, `tools` or earlier
messages change, all later thinking blocks become invalid and the request fails with 400): it adds no beta header and does no `drop_block`. Which
`Opaque` blocks the history carries is decided by the caller. An append-only history is not affected; after the caller rewrites the history (compaction, a new system or tools), it has to
stop carrying the earlier `Opaque` blocks itself. Such a 400 is `CallFailed` here.

Known limits: when replaying across protocols the tool call id is used as it is (no id has been seen to be rejected; not verified systematically); arguments that were not parsed
in the other protocol are an empty object `{}` when sent to Anthropic.

### 3.2 Vectors: `embed()` (0.5.0)

Why add it now: lite-rag, lite-mem and lite-cache each have an embedding client, and the Ollama and
OpenAI paths of lite-rag and lite-mem are nearly the same. Their calls should also work like LLM calls: configured by role, provenance recorded, no fallback on failure (the open
question in §15, decided by the owner on 2026-11-21). Only HTTP endpoints are accepted; in-process models such as fastembed and sentence-transformers stay in each engine,
otherwise every host would have to carry onnx or torch (the same reason as rule 1, no import of litellm).

```python
EMBED = RoleSpec({"embed": RolePolicy(kind="embed")})
reply = model.embed("embed", ["Ada lives in Wellington.", "..."])   # EmbedReply
reply.vectors      # list[list[float]], one per input, in the same order
reply.dim          # vector length
reply.provenance   # as for text(): answered comes from the response
reply.usage        # input = the tokens the endpoint reports; output = 0
```

- **Kinds of role**: `RolePolicy(kind="chat" | "embed")`, default `"chat"`. The rules:
  - An embed role is never routable: `kind="embed", routable=True` raises an error at construction (rule 5: the embedding is part of the index identity).
  - Calling `text()` / `structured()` / `turn()` on an embed role, or `embed()` on a chat role, is a `ConfigError`.
- **Backends**: `supports_embeddings` is profile data (since 0.7.0 it is the `serves` set, §3.3).
  - ollama, openai and litellm support it: it goes through the OpenAI SDK's `embeddings.create`, and Ollama uses its `/v1/embeddings`.
  - gateway does not support it (a non-goal of llm-gateway); anthropic does not either (Anthropic has no embedding API).
  - An embed role configured on an unsupported type raises an error at `build()`.
- **Request body**: `{model, input: [...], encoding_format: "float"}`. `"float"` is written out explicitly: when it is not written, openai 3.46.0 sends
  `base64` and then decodes it locally into float32 (`openai/lib/_parsing/_embeddings.py`), and whether Ollama accepts base64 has not been verified.
  `dimensions` is not sent; it will be added when a host needs it.
- **Call settings**: an embed role does not use chat parameters.
  - Explicitly setting `temperature`, `max_output_tokens` or `reasoning_effort` on an embed role is a `ConfigError`.
  - These three inherited from the top level are neither sent nor an error: the host's top level is configured for chat roles.
  - `timeout_s`, `connect_timeout_s` and `max_retries` apply as usual. Retries are done by the SDK on the same target (§8), with no fallback.
- **Batching**: at most 64 items per batch (lite-mem's original `BATCH`), sent batch by batch in order.
  - Every batch is checked for two things: the number of vectors equals the number of inputs, and all vectors have the same length. A mismatch is `BadStructuredReply` (the reply does not match the request), and the message carries no vector content.
  - The `answered` of all batches must be the same, otherwise it is `ProvenanceUnknown`. Usage is added up over the batches, and if any batch is unknown it is `None`.
  - Empty input is a `ValueError`: no request is sent, so there is no provenance to record.
  - On 400 / 413 a batch is split smaller and retried (lite-rag's approach), to be added when lite-rag is migrated.
- **Does not go through hooks**: the same as `turn()`. The hook's reply types are text or structured replies, and lite-cache does not cache vectors either.
- **Test tool**: `FakeModel.with_vectors({"embed": hash_vectors(8)}, **scripts)` / `AsyncFakeModel.with_vectors(...)` (since 0.5.1; the `vectors=` keyword of 0.5.0 made a host's `FakeModel(**scripts)` fail mypy).
  - The embed scripts are kept apart from the other scripts: they are not put into the `FakeReply` union type, otherwise the `list` scripts already in hosts' tests would fail type inference.
  - One callable per role: it receives `FakeEmbedCall(role, texts)` and returns vectors; an exception can also be given for it to raise.
  - Calls are recorded in `embed_calls`.
  - `hash_vectors(dim)` generates deterministic vectors from the sha256 of the text.
- **Measured** (`-m ollama`, 2026-11-21): the vectors bge-m3:567m returns through `/v1/embeddings` are element-wise equal to those the native `/api/embed` returns
  (one Chinese and one English sentence, 1024 dimensions). So after lite-mem is migrated, the vectors already stored remain usable.
- **Later (when lite-rag is migrated)**: query / document prefixes, Voyage / Cohere / Gemini, batch halving. These are all increments.

#### 3.2.1 Settings of the embed role and closing the client (0.6.0)

- The embed role's own three settings can be written only on `roles.<embed role>`, and a chat role that sets them is rejected at `build()` (symmetric with `CHAT_ONLY`):
  - `dimensions`: sent as the `dimensions` parameter, used for truncation. Ollama's `/v1/embeddings` supports it too; measured on 2026-11-22, bge-m3 truncated to 256 dimensions.
  - `batch_size`: how many items per request, default 64.
  - `max_concurrency`: the number of batches in flight at the same time, default 1, i.e. batch by batch. When concurrent, results are still merged in input order: sync uses a thread pool and async uses a semaphore.
- When a batch is rejected as too large (400 / 413) it is split in half, down to a single item; if a single item is still rejected, the whole call fails. The approach comes from lite-rag's `_fitting`.
  Splitting happens only on the batch that failed, and the other batches are not affected.
- `LiteModel.close()` / `AsyncLiteModel.aclose()`, and `with` / `async with`: close the SDK clients that lite-model built itself. The `http_client`
  a host passes in is closed by the host. The model derived by `limited()` shares the same set of clients, so the close of `Bounded` / `AsyncBounded` closes the wrapped
  model. A call after closing raises `RuntimeError`; the close of `FakeModel` does nothing.

### 3.3 Voyage and `rerank()` (0.7.0)

Voyage AI is the embedding provider recommended by the Anthropic docs (Anthropic itself has no embedding model). It has only embedding and rerank, and no chat.

- Built-in backend type `voyage`: `https://api.voyageai.com/v1`, `VOYAGE_API_KEY`, channel `api`. It goes through the OpenAI protocol module, because `/v1/embeddings` is in
  OpenAI's format. The differences from OpenAI are written in the profile (rule 7), and the client has no branch by name:
  - `serves={embed, rerank}`: the profile's `serves` lists the role kinds this type serves (it replaces `supports_embeddings` since 0.7.0); a chat role configured on voyage is rejected at `build()`.
  - `embed_dimensions_field="output_dimension"`: the embed role's `dimensions` is sent under this field name (through the SDK's `extra_body`).
  - `embed_encoding_format=None`: Voyage's `encoding_format` accepts only null or base64, so `"float"` is not sent, and the SDK is left to use base64 and decode it.
  - `embed_input_type=True`.
- `embed(role, texts, *, input_type=None)`: `"query"` / `"document"`, telling the model whether this batch of texts is queries or documents. Only types with `embed_input_type` accept it;
  passing it to other types raises `ConfigError` and it is not dropped silently. When a batch is split in half, every sub-batch carries the same `input_type`. Usage reads `prompt_tokens`, and if that is absent reads
  `total_tokens` (Voyage).
- `rerank(role, query, documents, *, top_k=None) -> RerankReply(scores, provenance, usage)`:
  - A new role kind `RolePolicy(kind="rerank")`, not routable, like embed: the reranker decides what the host keeps, which must be recorded and cannot be swapped by a router.
  - Only on types whose `serves` contains rerank (currently only voyage); a rerank role that sets chat or embed fields is rejected at `build()`.
  - The request is `POST /rerank`, `{query, documents, model, return_documents: false[, top_k]}`, through the SDK's generic `post`, so retries, timeouts and authentication are the same as for other calls.
  - The reply is sorted by relevance; lite-model sorts it back into input order by `index`, `scores[i]` is the score of the i-th document, and documents outside `top_k` are `None`.
    An index out of range, a duplicate, a score that is not a finite number, more entries than `top_k`, or fewer than `min(top_k, number of documents)` are all `BadStructuredReply`.
  - One request, no batch splitting, no hook calls. An empty document list raises `ValueError` and sends no request.
  - `RerankModel` / `AsyncRerankModel` protocols; `Bounded` passes through; `FakeModel.with_scores({...})` and `rerank_calls`.
- The request and reply formats are taken from Voyage's API reference (docs.voyageai.com, 2025-12-10), and the contract tests were written by hand against it; it has not yet been checked against the real API,
  and `-m voyage` (needs `VOYAGE_API_KEY`, billed per token, two very small calls) is the step that checks it.

### 3.4 Cohere and Gemini embedding (0.8.0)

The embedding APIs of both vendors are not in OpenAI's format (the paths, fields, reply structure and Gemini's authentication header all differ), so each has a protocol module that implements only embed
(`backends/cohere_embed.py`, `backends/gemini_embed.py`), on top of a small JSON-over-HTTP client `backends/rest.py`.
Where the client would depend on an SDK, it behaves like the SDK: 408 / 409 / 429 / 5xx and network errors are retried `max_retries` times on the same target (honoring `retry-after-ms` /
`retry-after`, otherwise doubling from 0.5 seconds up to 8 seconds); an HTTP error is raised with `status_code` (the 400 / 413 halving reads it), and the message has only the status code; only the client it built itself is closed.

- `cohere`: `https://api.cohere.com/v2`, `CO_API_KEY` (Bearer). `POST /embed`, `{model, texts, input_type, embedding_types: ["float"],
  output_dimension?}`; `input_type` is mapped to `search_query` / `search_document`. The reply `embeddings.float` is in input order, and `meta.billed_units.input_tokens` is the usage.
- `gemini`: `https://generativelanguage.googleapis.com/v1beta`, `GEMINI_API_KEY`, put in the `x-goog-api-key` header (not Bearer).
  `POST /models/{model}:batchEmbedContents`, one Content per text; `input_type` is mapped to `taskType` (`RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`),
  and it is not sent when not passed (for models like gemini-embedding-2 that use text prefixes, the host decides not to pass it); the dimension is `outputDimensionality`. The reply `embeddings[].values` has no usage.
- Neither vendor's reply says which model answered: `Provenance.answered` is None, and it is not guessed (rule 4). Both types serve only embed (`serves`), and a chat role configured on them is rejected at `build()`.
- The request and reply formats are taken from the formats lite-rag recorded from the API reference (2026-11-08), and the contract tests were written against them; `-m cohere` / `-m gemini` need a key and are billed per token, to be run once a key is available.
- Not migrated in: in-process embedders (fastembed's ONNX, sentence-transformers' torch). lite-model is an HTTP call layer; bringing them in would make every host carry
  these heavy dependencies, and they have no provenance, retries or usage to speak of. They stay in the host (lite-rag).

## 4. Configuration

lite-model provides pydantic config models, which the host module embeds in its own settings, with environment variables using the host's prefix
(`LITERAG_LLM__...`, `LITEMEM_LLM__...`). lite-model's own `LITEMODEL_` prefix is only for the CLI and tests.

```yaml
# defaults, then per-role overrides; a field left out keeps the default
backend: ollama             # required: no default (a lite-model release never changes a host's model)
model: qwen3:8b             # required
timeout_s: 120
connect_timeout_s: 10
max_retries: 1              # same target only (§8)
roles:
  answer:
    backend: litellm
    model: smart-router     # a routing alias: only allowed on a routable role
  extract:
    backend: gateway
    model: claude-sub-haiku
  graph:
    reasoning_effort: none
    max_output_tokens: 2048
backends:                   # named endpoints; the five built-ins are always there (changed by name)
  deepseek:  {type: openai, base_url: "https://api.deepseek.com/v1", api_key_env: DEEPSEEK_API_KEY}
  # built-ins (type = name), fields left out take the profile's value (§5):
  #   ollama    base_url http://127.0.0.1:11434/v1, no key, channel local
  #   openai    SDK default URL, OPENAI_API_KEY, channel api
  #   anthropic SDK default URL, ANTHROPIC_API_KEY, channel api
  #   gateway   http://127.0.0.1:8787/v1, no key, channel subscription
  #   litellm   http://127.0.0.1:4400/v1, LITELLM_API_KEY, channel api, router_aliases [smart-router]
  #   voyage    https://api.voyageai.com/v1, VOYAGE_API_KEY, channel api; embed and rerank only (§3.3)
  #   cohere    https://api.cohere.com/v2, CO_API_KEY, channel api; embed only (§3.4)
  #   gemini    https://generativelanguage.googleapis.com/v1beta, GEMINI_API_KEY, channel api; embed only (§3.4)
forbid_channels: []         # tests / CI set [subscription, api]
prices: {}                  # S4: "backend:model" -> {input_per_mtok, output_per_mtok}
```

**Named backends** (decided in S1, by the owner on 2026-11-18): each entry under `backends` has a name and a `type` (`ollama`, `openai`,
`anthropic`, `gateway`, `litellm`). The `type` determines the protocol and the parameter table of §5; `base_url`, `api_key_env`, `channel` and
`router_aliases` take the `type`'s defaults when not written. A role refers to a backend by name, so one host can use two
OpenAI-compatible vendors at the same time (DeepSeek and DashScope are both `type: openai`), and the endpoint and key are written in only one place. The five built-in names are always there.

`keyless: true` (0.1.1): this endpoint needs no key (for example Ollama's `/v1` used as `type: openai`, with no Ollama-specific parameters
added); the SDK gets a placeholder key; writing it together with `api_key_env` is rejected.

**The host's own defaults** (0.2.0, found in lite-mem S12): the host sets `host_backends` (laid over the built-in backends)
and `host_roles` (the call parameters of each role) on its subclass of `ModelsConfig`. Values given by the config are laid on top of them: a backend or role of the same name is changed field by field (for example changing only the built-in
`ollama`'s `base_url`; writing null in a role removes the value the host gave that role, and that role goes back to the top-level value). Only a backend written with a **different** `type` starts from scratch; writing the same `type` again
(`..._OPENAI__TYPE=openai`) does not lose the host's `base_url` or `keyless`. So the config **adding** a backend or changing a role does not lose the defaults it
did not mention. Where the key comes from (`api_key_env` or `keyless`) counts as one choice (0.2.1): if the config writes one of them, it replaces the host's one;
writing only `api_key_env` on a host's keyless backend means changing it to need a key, with no need to also write `keyless: false`. In 0.1.x, configuring any backend made the built-in value of the same name override the host's, so lite-mem's default role
was sent to api.openai.com.

Each role is resolved into a `Target` (frozen): backend (name), type, model, base_url, api_key_env, channel,
routed, reasoning_effort, max_output_tokens, temperature, timeout_s, connect_timeout_s, max_retries. The endpoint
and key belong to the backend, so when a role changes backend it changes endpoint and key, and does not keep the default backend's.
`max_output_tokens`, `reasoning_effort` and `temperature` are not sent when not configured (`temperature` takes the profile default, 0 for Ollama),
so the request body does not change when a host migrates (lite-mem does not send `max_tokens`, lite-rag sends 1024).
String fields (backend, model, base_url, api_key_env, router alias) do not accept an empty string: an empty value is not silently read as
"use the default". `temperature` is limited to 0–2.

**Validation at startup (in `build()`; a failure raises `ConfigError`)**, based on `KB/global/llm-app-architecture.md` §1.5:
- The config contains a role the host did not declare; a role the host declared cannot be resolved to a target (for example it refers to a backend name that does not exist).
- A missing key (the key environment variable is not set or is empty).
- An `anthropic` backend without the `anthropic` extra installed.
- A routing alias used on a role that is not routable (§7.3).
- The target's channel is in `forbid_channels`.
- A parameter the backend does not support (for example `anthropic` configured with `temperature`, see the parameter table in §5).
- All problems are listed at once (one `ConfigError`, one line each), and it does not stop at the first one. The error contains only the role, backend, model and
  the key's variable name, and never the key's value.
- `anthropic` without `max_output_tokens` configured (the Messages API requires `max_tokens`; no default is guessed, the host writes its own, lite-mem 2048,
  lite-rag 1024, so the request body does not change when migrating; the owner, 2026-11-18).
- An `anthropic` `base_url` ending in `/v1`: the Anthropic SDK builds `/v1/messages` itself (the OpenAI SDK wants `/v1` in
  `base_url`), and copying an OpenAI-style URL would request `/v1/v1/messages`, failing only at the first call.
- A routing alias used on a backend type with no `answered_header` (gateway: the actual model cannot be obtained, §6.2).

`litemodel dump-config`: reads the `LITEMODEL_*` environment variables and `.env`, prints the merged config and the target of every configured role
(the key shows only the variable name), sends no request, and also does **not run** the startup validation (the config can still be viewed when a key is missing; KB §10.3).

## 5. Backends

| backend | Protocol | Default channel | Structured output | Provenance (answered) |
| --- | --- | --- | --- | --- |
| `ollama` | OpenAI SDK → Ollama `/v1` | local | strict `json_schema` | response `model` |
| `openai` | OpenAI SDK, any OpenAI-compatible endpoint | api | strict `json_schema` | response `model` |
| `anthropic` | Anthropic SDK Messages (system messages are joined into the `system` field with `\n\n`; `max_tokens` / `model_context_window_exceeded` → `Truncated`; `end_turn` / `stop_sequence` / `tool_use` are complete replies; other stop_reasons (`refusal`, `pause_turn`, ...) → `EmptyReply`; a structured reply is the JSON of the text blocks joined together) | api | `output_config.format = {type: json_schema, schema: <strict schema>}`, no tools / `tool_choice` sent (0.5.0; before that it was a forced tool) | response `model` |
| `gateway` | OpenAI SDK → llm-gateway | subscription (can be overridden per profile) | strict `json_schema`; for DeepSeek / Qwen the gateway degrades to prompted, and the response header `X-LLMGW-Structured-Output: prompted` is recorded in the provenance | see §6.2 |
| `litellm` | OpenAI SDK → LiteLLM Proxy | api | strict `json_schema` (verified in S0 through `ollama_chat/`; to be verified when other upstreams are connected) | see §6.3 |

**Parameter differences are data** (following lite-rag's `PROVIDERS` approach), one `BackendProfile` table:

| Field | Description | Known values (verified in S0, 2026-11-18) |
| --- | --- | --- |
| `max_tokens_param` | `max_tokens` or `max_completion_tokens` | ollama: `max_tokens` (Ollama does not recognize `max_completion_tokens`, lite-rag 2026-11-08); openai: `max_completion_tokens`; litellm: `max_completion_tokens` (1.108.9 accepts both, and both took effect when measured through `ollama_chat/`; it uses OpenAI's new name, and LiteLLM converts it per upstream); gateway: `max_completion_tokens` (both map to `max_tokens`, the new name takes precedence, llm-gateway `translate.py:58-59`) |
| `default_temperature` | The value sent when not configured; None = not sent | ollama: 0.0; openai: None (reasoning models reject a non-default value); litellm: None (the upstream may be a reasoning model; `temperature: 0` is accepted through `ollama_chat/`) |
| `supports_temperature` | | anthropic 2.6.0: no (verified by lite-rag 2026-11-08) |
| `reasoning_effort` | How it is sent: top-level parameter, `extra_body`, or unsupported | litellm: top-level parameter; `"none"` turns off qwen3's thinking through `ollama_chat/` (measured), `"low"` does **not reduce** thinking for qwen3, and unset = thinking on. gateway: **currently dropped** (`schemas.py` has no such field, and it goes into `X-LLMGW-Ignored-Params`); until llm-gateway is fixed, configuring it raises an error rather than silently having no effect |
| `key_env` | The default key variable | |
| `answered_header` | Which response header holds the model that actually answered (when the response body `model` is not it) | litellm: `x-litellm-model-name`; others: none (read the response body) |
| `via_headers` | The provenance headers recorded into `Provenance.via` (allow list) | litellm: `x-litellm-model-id`, `-model-group`; gateway: `X-LLMGW-Backend`, `X-LLMGW-Structured-Output` |
| `ignored_params_header` | The response header that lists dropped fields | gateway: `X-LLMGW-Ignored-Params` |
| `request_headers` | Headers carried on every request | litellm: `x-litellm-num-retries: 0` |
| `role_tag` | Sends the role as `metadata.tags` (§7.3) | litellm: yes |
| `supports_tools` | Whether the endpoint accepts `tools` (`turn()`, §3.1); otherwise `ConfigError` | Yes for all five types (0.4.0). Measured: ollama `qwen3:1.7b` (2026-11-21, `-m ollama`); llm-gateway supports it according to its `docs/DESIGN.md` §9, not measured through lite-model; litellm, openai and anthropic have only contract tests |

Thinking tokens count toward the output limit: with qwen3 through LiteLLM and thinking on, a small `max_tokens` gives an empty `content` and
`finish_reason: "length"` (which should raise `Truncated`, not `EmptyReply`).

When the gateway backend sees, in the response header `X-LLMGW-Ignored-Params`, a field that lite-model **actually sent** (`max_*_tokens`,
`temperature`, `reasoning_effort`, `response_format`), it raises `IgnoredParameter` and does not treat the dropped setting as in effect;
other fields in the list (not sent by lite-model) are not an error. The error contains only lite-model's own field names and does not echo the gateway's list.

## 6. Provenance

### 6.1 Rules

- Every answer carries `Provenance`. `answered` takes **the model that actually answered in the response**, not the config; when the endpoint does not say it is `None`,
  and `tag` uses `requested?` to mean "unconfirmed", and does not pass it off as confirmed.
- The host stores `tag` (`backend:answered`), and stores `requested` and `channel` separately when needed. The `provider:model` format that lite-mem stores today
  stays compatible when migrating: the prefix of `tag` is the **backend's name**, so the host names the backend after the provider it stores now
  (lite-mem stores `openai:qwen3:8b` for the local Ollama: `backends: {openai: {type: ollama}}`), and the stored format does not change;
  it can also be mapped in the adapter.
- One operation fixes one target (KB §1.4): if the config changes during the call, it does not switch midway.

### 6.2 Through llm-gateway

The gateway's response `model` field holds the **profile name** (`translate.py:194,256`), and it does not forward the upstream's response headers. So:
- A fixed profile (one profile per model): `answered` = the profile name (the response body `model`), `via` records
  `x-llmgw-backend` (such as `claude-code`) and sometimes `x-llmgw-structured-output: prompted`, `routed=False`.
- A routing alias configured on the gateway (`backends.<gateway>.router_aliases`): until llm-gateway passes through the upstream model (§15),
  `build()` **rejects** it (the gateway profile has no `answered_header`, and routing is not allowed when the actual model cannot be obtained).

### 6.3 Through LiteLLM

Measured in S0 (1.108.9, 2026-11-18, `KB/global/litellm/README.md` §2–§3):
- For a **plain alias** (`workhorse`), the response body `model` is always the alias itself; the actual model is only in the response header
  `x-litellm-model-name` (the deployment's LiteLLM model string, such as `ollama_chat/qwen3:8b`).
- For a **routing alias** with `return_raw_model_name: true` turned on, the response body `model` is the LiteLLM model string of the chosen deployment
  (`ollama_chat/qwen3:1.7b`); when it is not turned on it is changed back to the alias, but the `x-litellm-model-name` header is still the actual model.
- Every successful response has `x-litellm-model-name`, `x-litellm-model-id` (the chosen deployment) and `x-litellm-model-group`
  (the alias called); error responses do not.

Rules (implemented in S3, `openai_compat._answered`):
- `answered` = the response header `x-litellm-model-name` (the profile's `answered_header`). For a routing alias the response body `model` must agree with the
  header (it does when `return_raw_model_name` is on); if not, `ProvenanceUnknown` is raised. When the header is missing, both plain aliases and routing aliases
  raise `ProvenanceUnknown` (when the actual model cannot be obtained, it is not passed off as known). It also raises when the header value is just the requested alias itself (an alias cannot be told from a model;
  so in the proxy config an alias must not have the same name as its model string, which `tests/unit/test_deploy_litellm.py` guards).
- `via` records `x-litellm-model-id` and `x-litellm-model-group` (only headers in the allow list, with lower-case names).
- `-m litellm` measured (2026-11-18): `workhorse` → `ollama_chat/qwen3:8b`; `smart-router` for "hi" →
  `ollama_chat/qwen3:1.7b`, and for a long multi-step programming question → `ollama_chat/qwen3:8b`; `routed=True`.
- The proxy config still requires routing aliases to turn on `return_raw_model_name` (guarded by `tests/unit/test_deploy_litellm.py`), so that the logs and
  the response body also point to the actual model.
- Cost: `x-litellm-response-cost` is **absent** from the successful responses of Ollama models (they have no price), and is `0` in error responses. It is written into `Usage.cost_usd` only when this header is present
  and the response is successful; to be verified when an upstream with prices is connected.

## 7. Routing

### 7.1 Why only the Proxy

- The litellm package was poisoned twice in 2026: 1.82.7 / 1.82.8 in March; 1.107.0 on 21 October (all 7 wheels on PyPI, and the
  Docker tags `latest`, `main-stable` and others). The payload ran at Python startup and stole credentials. Installed in the process, it exposes the database
  connections and all keys the host can reach; in a container, it gets only the few keys mounted into it.
- llm-steward imports lite-rag directly, so library dependencies would be passed on; steward already removed LiteLLM on 2026-11-08 because the dependencies were too heavy.
- The routing config (model_list, tiers, classifier) lives in the proxy, one copy shared by all modules.

### 7.2 Proxy config (`deploy/litellm/`, since S0)

`deploy/litellm/config.yaml` is the actual config; the S0 tiers are all local Ollama (`small` = `qwen3:1.7b`, `workhorse` =
`qwen3:8b`), so tests and verification cost nothing. When a paid upstream is connected, add a deployment in the shape below:

```yaml
model_list:
  - {model_name: small,     litellm_params: {model: ollama_chat/qwen3:1.7b, api_base: "http://host.docker.internal:11434"}}
  - {model_name: workhorse, litellm_params: {model: deepseek/deepseek-chat, api_key: os.environ/DEEPSEEK_API_KEY}}
  - {model_name: flagship,  litellm_params: {model: anthropic/claude-sonnet-5-5, api_key: os.environ/ANTHROPIC_API_KEY}}
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers: {SIMPLE: small, MEDIUM: workhorse, COMPLEX: flagship, REASONING: flagship}
        classifier_type: heuristic       # no extra model call; same input -> same tier
        return_raw_model_name: true      # §6.3
        context_compaction: false        # default on: rewrites history with an extra model call
      complexity_router_default_model: workhorse
litellm_settings:
  num_retries: 0                         # retries stay in lite-model (§8)
router_settings:
  disable_cooldowns: true
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
# no `fallbacks`: a failure raises (KB llm-app-architecture §3.2)
```

`context_compaction` (on by default in 1.108.9): when a request approaches the input limit of the chosen deployment, the router makes one more model call to compact the history into a
summary before sending it. This is both an invisible model call and a rewrite of the request the host built, so it is turned off. It also has to read the deployment's
`max_input_tokens`; Ollama models are not in LiteLLM's price table, so with it on **every** routed call returns 500 (measured in S0).

Running it: the version, digest, public key and upgrade steps are in `deploy/litellm/README.md`. The design requirements:
- The image is locked by digest; **never use 1.107.0, 1.82.7 or 1.82.8**, and never a floating tag. LiteLLM signs with a cosign **key**
  (not keyless), and the public key is pinned in the repo.
- Signature verification is in compose: the `verify` service runs `cosign verify` first, `litellm` depends on it exiting successfully, and if verification fails it does not start; cosign is not installed on this
  machine, and CI runs the same `verify`.
- No database is connected (`max_budget` has no effect without a database); it binds only 127.0.0.1:**4400** (4000 is registered to tenderted); only the
  needed keys are mounted.
- **The Proxy's cache is not turned on** (`litellm_settings.cache`, 2026-11-19): caching is lite-cache's job, and it plugs into
  lite-model as a hook (§10). lite-model currently cannot recognize a Proxy hit: the reply is not marked `cached`, and the tokens saved would be recorded as spent this time
  (§9). 0.3.0: when a non-empty `x-litellm-cache-key` is read, `cached=True` is set (the profile's `cached_header`), so even
  if someone turns it on, the usage is not counted wrongly. Based on the 1.108.9 source: `cache_key` is written into the hidden params only on the hit branch (`caching/caching_handler.py`),
  and an empty header is not sent (`proxy/common_request_processing.py`). Measured against the Proxy on 2026-11-20 (the owner agreed, with an in-memory cache turned on temporarily):
  a miss has no such header, a hit has it, and `x-litellm-model-name` is still the model that originally answered; the `usage` of a hit reply is the original token count
  (KB `global/litellm` §3.1).

**Subscription quota does not enter automatic routing**: the tiers contain no deployment that points at an llm-gateway subscription profile. To use a subscription, configure
`backend: gateway` explicitly on the role. When routing between subscription models by difficulty is really needed, start a separate alias (such as `sub-router`), which
takes effect only when explicitly chosen and is subject to `forbid_channels`.

### 7.3 Client guards

- `RolePolicy.routable`: declared by the host. For lite-rag's `contextual` and `graph`, the model id is part of the index identity
  (lite-rag `config.py:155,217`), so they must be `routable=False`; measuring uses such as a judge must also be `False`.
- `backends.litellm.router_aliases` lists which names are routing aliases; a role that is not routable and is configured with one of them is rejected at `build()`.
- Every request carries the role: `metadata: {tags: ["role:<role>"]}` (S0: both `user` and `metadata.tags` are accepted; without a database or
  log callback the proxy records them nowhere, and which one is visible will be verified when logging is connected). `tags` is used and not `user`; `user`
  is left for the host to denote the end user.
- Every request carries `x-litellm-num-retries: 0`. Measured in S0: with `num_retries: 2` configured, an unreachable deployment is retried
  2 times without the header (4.8 s) and 0 times with it (0.06 s). Retries happen in one place only, lite-model.

## 8. Errors and retries

```
ModelError (base)
├─ ConfigError            config invalid; raised by build(), never at call time if avoidable
├─ CallFailed             transport / HTTP error / SDK exception, after retries on the same target
├─ Truncated              finish_reason "length" / stop_reason "max_tokens", "model_context_window_exceeded"
├─ EmptyReply             no usable answer: no text (e.g. the whole budget went to thinking), no forced-tool
│                         input, or a stop reason that leaves none (refusal, pause_turn, content_filter)
├─ BadStructuredReply     the reply does not validate against the schema
├─ IgnoredParameter       the endpoint reported a configured parameter as ignored (§5)
└─ ProvenanceUnknown      the answering model cannot be read: a routed call, or any litellm call whose header is
                          missing or only repeats the alias (§6.3)
```

- **No fallback**: on failure it raises, and does not switch model or backend.
- Retries: only the same target is retried, using only the SDK's `max_retries` (default 1). LiteLLM's retries are turned off (§7.3), to avoid
  SDK × proxy × the host's task queue stacking up to dozens of attempts.
- Error messages contain only the role, `backend:model` and the exception type, and **no upstream text** (the SDK and proxy may echo keys, auth headers and URLs with credentials;
  following lite-mem's `_failed`); the original exception stays in `__cause__`.

Differences for `turn()` (§3.1): a truncated round is a reply (`stop="length"`), not `Truncated`, unless it carries nothing; there is no
`BadStructuredReply` (invalid arguments are `ToolCall.arguments=None`). The other errors, usage and the "no upstream text" rule are the same.

## 9. Usage and cost

- `Usage`: input, output, reasoning (`completion_tokens_details.reasoning_tokens`, recorded when present; through LiteLLM →
  Ollama it is null, and the thinking tokens are already included in `completion_tokens`, measured in S0), cost_usd.
- Failed and truncated calls also report usage: when the endpoint answered but the answer cannot be used (`Truncated`, `EmptyReply`, `BadStructuredReply`,
  `IgnoredParameter`, `ProvenanceUnknown`), the exception's `usage` is the tokens and cost spent on this call; for `CallFailed` and
  `ConfigError` the `usage` is `None` (there was no answer, and the SDK cannot get usage).
- Cost: the cost the endpoint itself reports is preferred (the profile's `cost_header`, litellm: `x-litellm-response-cost`, used only if it can be read as a non-negative finite
  number); otherwise the `prices` in the config are used (key `"backend:model"`, where model is the model that **actually answered**, consistent with `Provenance.tag`;
  USD per million tokens); if neither exists it is `None`, and it is not estimated. The cost of a subscription channel is always `None`. When the backend in a `prices` key
  does not exist, `build()` rejects it.
- A reply from a hook hit (`cached=True`) carries the `usage` of **the original** call: what was saved this time, not what was spent this time. A host that totals spend
  checks `cached` first.

## 10. Concurrency and hooks

- `Bounded(model, concurrency)` / `AsyncBounded`: follows lite-rag's `BoundedCalls`, wraps a `Model` / `AsyncModel`, and
  at most `concurrency` calls going through it (including those started by other threads / tasks) run at the same time; `map()` runs independent calls concurrently, returns them in input
  order, and raises the first error.
- Hook (KB §10: registered, revocable, addressable):

  ```python
  class Hook(Protocol):
      name: str
      def before(self, call: Call) -> TextReply | StructuredReply | None: ...   # a reply = short-circuit (cache hit)
      def after(self, call: Call, reply: TextReply | StructuredReply) -> None: ...

  model.hooks.register(hook) -> Registration   # Registration.remove() unregisters
  ```

  `Call` holds the role, the resolved `Target`, messages, `request` (the SDK parameters actually sent), `params` (with model,
  messages and the transport-level `extra_headers` removed), schema, `schema_name` and `schema_hash` (the sha256 of the strict schema's normalized JSON):
  exactly the fields lite-cache's L1 cache key needs (lite-cache design §3.1).
- Order: each call takes a snapshot of the hooks; `before` runs in registration order, and the **first** hook that returns a reply ends the call, no request is sent,
  and the returned reply has `cached=True` (set on the reply, not on `Provenance`: the provenance stays the model that really answered originally; the owner, 2026-11-18).
  `after` runs only for a reply that was just obtained from the endpoint (it does not run for hits or failed calls, so the cache is not written twice).
- When there is no hook no `Call` is built. Hooks are synchronous, shared by sync and async models (lite-cache's store is synchronous).
- Names are unique (registering a duplicate raises `ConfigError`), `model.hooks[name]` retrieves one, and `Registration.remove()` revokes it (revoking twice is harmless).
- Call context: `with call_context(scope="acme/u42", cache=True): model.text(...)`, and the hook reads it from `Call.context` (read-only);
  nested blocks stack and override the outer one. It is a contextvar: the `Model` protocol gets no new parameter, and the host adapter, `FakeModel` and `Bounded`
  are all unchanged; both synchronous calls and asyncio tasks created inside the block can see it. lite-model itself never reads these values (the owner, 2026-11-18).
- `LiteModel.target(role)` / `AsyncLiteModel.target(role)`: gets the resolved target of a role before the call (the host records it into the
  index identity, §12.1).
- A hook that raises = the call fails, and the error is raised as it is (not swallowed, not wrapped); when `before` raises no request is sent. When `after` raises, the reply that was already billed
  is dropped together with its `usage`: `after` should not raise (when lite-cache fails to write, it still returns the reply; there is no `store_error` field on the reply, and the failure is recorded in the log and in `CacheHook.store_errors`, lite-cache design §3.5).
- The `Call.request` a hook sees is a read-only deep copy of the actual request: a hook cannot change the request that is sent, and the cache key describes the one that was sent.
- When a hook returns a reply of the wrong type (`text()` gets a `StructuredReply`, `structured()` gets a `TextReply` or a value of another schema)
  it raises `TypeError` naming the hook: it fails where the error is, not when the caller reads the value.
- `build()` / `build_async()` return `LiteModel` / `AsyncLiteModel` (they satisfy `Model` / `AsyncModel`, with one extra
  `.hooks`); the `Model` protocol itself has no hooks, and `FakeModel` and hosts' adapters need not implement them.
- **0.3.0 (decided 2026-11-19, implemented 2026-11-20)**:
  - `AsyncHook` (`async def before / after`, `lite_model.hooks`), which `AsyncLiteModel` awaits; a synchronous `Hook` still works on an async
    model as before. Registering an `AsyncHook` on a sync model raises `TypeError` (it would never be awaited). Reason: llm-steward is async, and
    lite-cache's synchronous psycopg queries inside a synchronous hook would block the event loop.
  - When `Bounded` / `AsyncBounded` wraps `LiteModel` / `AsyncLiteModel`, the slot wraps only the request actually sent, inside the executor
    (it recognizes `Limitable` structurally: a model with `limited(slot)`, sharing routes and hooks; a host adapter only has to forward it): a hook hit no longer queues, and a hook can also call a model through the same `Bounded`
    (this used to deadlock at concurrency 1; the L2 verifier has exactly this shape). When it wraps another `Model` (`FakeModel`, a host adapter), the whole call
    still takes one slot as before.
  - The worker threads of `Bounded.map` run each item with a copy of the caller's context (one copy per item), so the values of `call_context()` reach the hook
    (found in lite-rag S18; lite-rag worked around it first in its own `BoundedLLM.map`; 2026-11-19).
  - `cached=True` means "no money was spent again this time", and it has two sources: a hook hit (it does not go through `after`), or a hit in the Proxy's own cache
    (`x-litellm-cache-key`, §7.2; it does go through `after`). When a hook sees `reply.cached` true in `after`, it should not count its
    `usage` as spent this time (lite-cache's `after` is still to follow up).
  - No `failed(call, error)` hook is added: lite-cache does not count failed calls (its design §0), and the usage of a failed call is on the exception (§9).

- **`turn()` does not go through hooks** (0.4.0): the hook's reply types are `TextReply | StructuredReply`, the only hook is lite-cache,
  and an agent's rounds are not cached (the context is different every round). Letting a hook see `TurnReply` would first require extending the `Hook` protocol and lite-cache's serialization
  (§15). The `Bounded` / `AsyncBounded` slot still applies to `turn()` as usual.

## 11. Test tools

- `lite_model.testing.FakeModel`: scripted replies per role (text or a structured object, or an exception to raise; a list is consumed per call,
  a function is computed per call), records every request (role, messages, schema), and implements `Model`. `AsyncFakeModel` has the same scripts and the same
  `calls`, and implements `AsyncModel` (one class cannot make a method of the same name both sync and async). The provenance has `backend="fake"`.
- `FakeModel.turn` / `AsyncFakeModel.turn` (0.4.0): a script is an `AssistantTurn` (`stop="tool_calls"` when there is a call, otherwise
  `"end"`), `(AssistantTurn, stop)` (to script `"length"`), or an exception; calls are recorded in `turn_calls` (`FakeTurnCall`: role,
  messages, tools) and do not go into `calls`, so the type of `FakeCall.messages` is unchanged and hosts' existing tests are not affected.
- The contract tests run once sync and once async (the `connect` fixture in `tests/contract/conftest.py`).
- Contract tests: **the real SDK**, with only the transport replaced by a fake endpoint that records requests (`build(..., http_client=...)`,
  `tests/upstream.py`), checking the request body each backend actually sends (parameter names, strict schema, forced tool, header).
- `tests/conftest.py` rejects every socket connection outside `-m ollama` / `-m litellm`: a test that forgot to inject a transport fails,
  rather than quietly calling a real service.
- Markers: `-m ollama` (real local Ollama, free); `-m litellm` (needs a local proxy, **only aliases that point at Ollama**);
  there is never a test that calls a subscription or paid API. The test config sets `forbid_channels: [subscription, api]`.

## 12. Migration

Each module migrates as one stage **in its own repo**, and the acceptance for each includes: the request bodies sent before and after migration are the same (contract test),
the eval numbers are the same, and the provenance format is compatible.

| Order | Module | Key points |
| --- | --- | --- |
| 1 | lite-mem | The most complete feature set (strict schema, Anthropic forced tool, per-role overrides), so it tests the interface best. `produced_by` keeps storing `provider:model`. **Migrated** (2026-11-18, lite-mem S12, using 0.1.1's `keyless`; 12 request bodies the same before and after) |
| 2 | lite-rag | The three LLM configs become three roles; `contextual` and `graph` are not routable; `Completion.model_id` becomes the model that actually answered; `literag eval` is the same before and after. **Migrated** (2026-11-18, lite-rag S15, using 0.2.0's `host_roles`; 5 request bodies the same before and after, index identity unchanged) |
| 3 | llm-steward | Async; also fixes its strict schema, which was never cleaned up (raw `model_json_schema()`). **Migrated** (2026-11-19, llm-steward P9; 5 request bodies the same before and after — its schema already had the strict shape, so the `strict_schema` result is unchanged; the pinned lite-rag and lite-mem were upgraded together to the post-migration versions) |
| — | lite-eval judge | Not migrated for now (§1 non-goals) |

lite-model is released as a GitHub release wheel, and hosts pin the version in `[tool.uv.sources]` (the same approach as lite-eval).

### 12.1 What is deleted and what stays after migration

The purpose of migration is to remove duplication: once a module is migrated, its own copy of the LLM client code is **deleted**. Modules can migrate one at a time,
and those not yet migrated work as usual; but a migrated module cannot keep the old client at the same time.

| Module | Deleted (now provided by lite-model) | Stays in the module |
| --- | --- | --- |
| lite-mem | In `llm.py`: `strict_schema`, `_failed`, `_parsed`, `_VendorLLM`, `OpenAICompatibleLLM`, `AnthropicLLM`, `RoutedLLM`; in `config.py`: `RoleOverride`, `ModelTarget`, `LLMConfig` (changed to embed lite-model's config); FakeLLM in `testing.py`; the corresponding unit tests | The role names (`extract`, `resolve`, `consolidate`) and their `RoleSpec`; serializing the payload into the user message; `prompts/*.j2`; the storage format from `Provenance` → `produced_by`; the embedding client (until v2) |
| lite-rag | In `llm.py`: `PROVIDERS`, `ProviderProfile`, `_Client`, `OpenAICompatLLM`, `AnthropicLLM`, `BoundedCalls`, `build_llm`; in `config.py`: `LLMConfig` and its provider validation; FakeLLM in `testing.py`; the corresponding unit tests | The declarations of the three roles `answer`, `contextual`, `graph` (the latter two `routable=False`); graph parsing JSON out of text (business logic); the LLM id in the index identity (changed to come from lite-model's target); `prompts/*.j2`; embedding, rerank |
| llm-steward | `llm/client.py`; the parts of `llm/roles.py` that resolve roles and call models; the code that generates schemas by itself in various places (changed to use lite-model's `strict_schema`); `tests/fakes/llm.py` | The role names (`memory.extract`, `memory.resolve`, `kb.risk`, `kb.worthiness`, `kb.draft`); mem0's LLM config (mem0 calls the model internally, not through lite-model) |
| lite-eval | Not migrated (§1 non-goals) | The whole judge client |

Each module keeps one **thin adapter** (a dozen lines): it connects the module's own `LLM` Protocol to `lite_model.Model`, so the call sites inside the module
need no change, and it also satisfies the rule "every component is a Protocol". Whether to change to depend directly on `lite_model.Model` and drop the adapter
is decided when each module migrates.

### 12.2 Migration steps for each module

1. First write contract tests that fix the request bodies actually sent **now** (parameter names, schema, forced tool, header).
2. Switch to calling lite-model and run the same contract tests: the request bodies are the same.
3. Run the module's own eval (lite-rag `literag eval`, lite-mem's eval): the numbers are the same; the numbers before and after are recorded in the KB.
4. Once they match, **delete** the old client, the old fake and the unit tests that tested only the old client; FakeLLM is replaced by `lite_model.testing.FakeModel`.
5. Update the module's design doc and `AGENTS.md`: the LLM-related rules (Protocol, no fallback, provenance record, Tests never call a real
   model) now point to lite-model; remove `openai` / `anthropic` from the dependency list if they are no longer used directly (and not used elsewhere).
6. Finish with the module's stage close-out process (/simplify → Codex review → fix → re-run all checks).

## 13. Project structure

```
lite-model/
├─ src/lite_model/
│  ├─ __init__.py        # build, build_async, Model, AsyncModel, Message, replies, errors
│  ├─ types.py           # Message, Provenance, Usage, replies, Channel
│  ├─ config.py          # ModelsConfig, BackendConfig, Target, RoleSpec, RolePolicy, resolve() + checks
│  ├─ client.py          # build(), Model, dispatch per role; SDK clients shared per endpoint
│  ├─ profiles.py        # BackendProfile table (§5)
│  ├─ schema.py          # strict_schema()
│  ├─ backends/          # __init__ (Backend protocol, provenance, finish), openai_compat.py
│  │                     # (ollama/openai/gateway/litellm), anthropic_messages.py; each: connect,
│  │                     # connect_async, endpoint, request(), reply(); client.py runs them
│  │                     # (provenance: finish() in backends/__init__; header rules in openai_compat, from the profile)
│  ├─ hooks.py           # Hook, Hooks, Registration, Call
│  ├─ bounded.py         # Bounded, AsyncBounded
│  ├─ errors.py
│  ├─ testing.py         # FakeModel, AsyncFakeModel
│  └─ cli.py             # litemodel dump-config
├─ tests/unit/ tests/contract/ tests/live/   # live: -m ollama / -m litellm
│  tests/upstream.py     # recording fake endpoint, plugged in as the SDK transport (contract tests)
│  tests/conftest.py     # refuses sockets outside -m ollama / -m litellm
├─ deploy/litellm/       # config.yaml, compose.yaml (pinned digest), README
└─ docs/lite-model-design.md
```

## 14. Stages and acceptance

At the end of each stage: ruff, ruff format, mypy and pytest are all green; every new test has been seen red once; then the close-out follows step 4 of the `AGENTS.md` Workflow
(/simplify → Codex read-only review → fix → re-run all checks).

| Stage | Content | Acceptance |
| --- | --- | --- |
| **S0** Environment and verification | Repo skeleton (uv, ruff, mypy, pytest, CI); version pinning (openai, anthropic and pydantic the same as lite-mem); start the LiteLLM Proxy (locked by digest, cosign signature check, no database, 127.0.0.1:4400 (originally 4000, which conflicts with tenderted), the §7.2 config, with the tiers using only Ollama models at first so nothing is spent); verify with curl: `max_tokens` / `max_completion_tokens`, `temperature`, the conversion of `reasoning_effort` to `ollama_chat`, strict `json_schema`, `return_raw_model_name`, the response headers (`x-litellm-model-id`, `-model-group`, `-response-cost`), `x-litellm-num-retries`, the field that passes the role through | The verification results are written into the KB (`KB/global/litellm/README.md`, and two outdated points in `tech-selection/01b` are corrected: `enable_context_window_escalation` defaults to false, and `classifier_fallback`); the §5 parameter table and §6.3, §7.3 are updated per the results; the port is registered in `KB/global/machine/` |
| **S1** Core | types, config (with startup validation), errors, `strict_schema`, the openai-compatible backend (ollama / openai), sync `text` / `structured`, provenance taken from the response, `FakeModel`, `dump-config` | Unit + contract tests cover every validation in §4, the ollama/openai parameters in §5, and every kind of error in §8; `-m ollama` runs text and structured through |
| **S2** Anthropic and async | The anthropic backend (forced tool, system field, `stop_reason`); `build_async` and the two async backends | Contract tests check the Anthropic request body; sync and async share one set of contract tests |
| **S3** gateway and litellm | The two backends; channel and `forbid_channels`; the `router_aliases` and `RolePolicy` guards; the §6.2 / §6.3 provenance; `IgnoredParameter`; `x-litellm-num-retries: 0` | Contract tests cover: a routing alias rejected on a role that is not routable, `ProvenanceUnknown` when `return_raw_model_name` is not on, an error when the gateway ignores a parameter; `-m litellm` runs through with an Ollama alias, and the provenance is the actual model |
| **S4** Usage, concurrency, hook | `Usage` (including reasoning, and usage on failure), the price table and cost header, `Bounded`, hook register / revoke / short-circuit | Tests cover: a failed call also has usage, a hook hit keeps the original provenance and `cached=True`, and a revoked hook no longer takes effect |
| **S5** Release and close-out | README, `deploy/litellm/`, the v0.1.0 release wheel; the final close-out review of the whole library | A host can install from the wheel; the full set of checks is green |
| **S6+** Migration | Migrate in each of the lite-mem, lite-rag and llm-steward repos per §12 | See §12 |
| **0.3.0** Prerequisites for lite-cache (implemented 2026-11-20) | `AsyncHook`; the `Bounded` slot wraps only the real request; `Bounded.map` carries the caller's context; `cached=True` when a litellm reply carries `x-litellm-cache-key` | `tests/contract/test_bounded_hooks.py`: at concurrency 1 a hit does not wait for a running call (sync, async); a hook calling a model through the same `Bounded` does not deadlock; in `map` the hook can see `call_context`; an async hook hit short-circuits and does not block other tasks; a sync model rejects `AsyncHook`. `test_usage_and_hooks.py`: a litellm reply with the cache header gets `cached=True` with the provenance unchanged, and an empty value or another backend does not count |
| **0.5.0** Vectors and Anthropic structured output (implemented 2026-11-21) | `embed()` (§3.2): `RolePolicy(kind=)`, `EmbedReply`, `EmbedModel` / `AsyncEmbedModel`, `supports_embeddings`, batching and validation, `FakeModel(vectors=)`, `hash_vectors`; Anthropic `structured()` switched to `output_config.format` (§15) | `tests/contract/test_embed.py` (sync, async): the request body (`encoding_format: "float"`, no chat parameters); sorted back into input order by index; provenance and usage; 64 items per batch, order and usage total; a wrong count or dimension, and inconsistency between batches; different batches answered by different models; empty input sends no request; retry on the same target, and errors carry no upstream text; no hook call; a wrong role kind, an unsupported backend, and chat parameters on an embed role are rejected. `test_anthropic.py`: a structured request has only `output_config`, with no tools / `tool_choice`; the reply is read from the text JSON. `-m ollama`: the vectors of `/v1/embeddings` and `/api/embed` are equal |
| **0.4.0** Prerequisites for lite-agent (implemented 2026-11-21) | `turn()` (§3.1): `ToolSpec`, `AssistantTurn` (ordered blocks), `ToolResults`, `TurnReply`, `ToolModel` / `AsyncToolModel`; the request and reply mapping for both protocols; the `Opaque` replay rule; `supports_tools`; `FakeModel.turn` | `tests/contract/test_turns.py` (sync, async): the request body with tools; calls and results in the history (OpenAI one `tool` message per result, Anthropic one user message); the order and ids of several calls; truncation returns `stop="length"` and its calls; invalid arguments; the fields and values of a thinking block round-trip unchanged, with position unchanged and interleaved order kept; an unknown `finish_reason`; an empty `ToolResults`; a system message in the middle of the history; an `Opaque` of the other protocol is not sent; `EmptyReply` / `Truncated` / `CallFailed` / `ProvenanceUnknown` carry the right usage; no hook call; the `text()` request body is unchanged. `-m ollama`: one real tool round plus its result |

### Stage records

(After each stage is done, append here: date, what was done, the review conclusion, and rejected findings with reasons.)

- **0.8.0 (2026-11-22)**: the two embed-only types `cohere` and `gemini`, and the `rest` client they use (§3.4). An incremental change. Reason: the remaining
  two HTTP embedders in lite-rag were implemented by itself (retries, halving and error mapping each written separately). In-process embedders are not migrated (reasons in §3.4).
  - Tests: `tests/contract/test_rest_embed.py` (sync, async): both vendors' URLs, auth headers (Gemini sends no Authorization), request bodies, input_type mapping, dimensions,
    reply parsing and usage; Gemini sends no taskType when input_type is not passed; success after a retry on 503 / 429; 400 is halved and not retried; 401 is not retried and the error has only the type name;
    a wrong count is BadStructuredReply; a chat role is rejected; an injected http_client is not closed. All red when the import fails. `tests/live/test_rest_embed.py`
    (`-m cohere` / `-m gemini`) is written, there is no key, and it was not run.
  - /simplify accepted: the sync / async retry loops share `_outcome` (the verdict of each attempt: a reply, how long to wait, or an error); `rest.Raw` replaces the same `_Raw` in openai_compat;
    both vendors share `rest.connectors(auth)`, always passing an empty string when there is no key; two outdated comments ("the embed types are all OpenAI-compatible") were changed. Rejected: removing the `cast` from `BACKENDS`
    (the chat path is typed as `Backend`, and switching to a union type would push the cast to every chat call site).
  - Both Codex findings were valid, and a failing test was written before each fix: when `retry-after` was an HTTP date it was treated as unreadable and the code's own backoff was used (the SDK reads it); on a network error the original exception was left on the
    cause chain, and the original exception may contain request headers (Gemini's key is in a header). Now a network error is replaced by a `TransportFailure` with only the type name, and the cause chain is cut.
  - pytest 420 passed; mypy and ruff clean.
- **0.7.0 (2026-11-22)**: the `voyage` type, `embed(input_type=)`, `rerank()` (§3.3). All incremental changes; the `EmbedModel` protocol gains one keyword-only parameter
  with a default, and a host that implements this protocol itself has to add it. Reason: lite-rag's `VoyageEmbedder` was an HTTP call it wrote itself, and Voyage rerank had no place to go.
  - Tests: `tests/contract/test_voyage.py` (sync, async): the request URL, auth, `input_type`, `output_dimension`, `encoding_format: "float"` not sent;
    base64 vectors decoded; not sent when there is no input_type; on 400 the batch is halved and the sub-batches carry the same input_type; a type that does not accept input_type raises an error and sends no request; a chat role
    configured on voyage is rejected; rerank is sorted back into input order, outside `top_k` is None, the request body, provenance and usage; three kinds of mismatched reply; a rerank role on an unsupported
    type, and a chat/embed role calling rerank, are rejected; a rerank role is not routable; fake and Bounded pass through. All red when the import fails; for the halving and Bounded tests added later,
    the sub-batch's input_type was removed and Bounded was made not to pass the argument, they were seen to go red, and then restored. `tests/live/test_voyage.py` (`-m voyage`) is written, there is no key, and it was not run this time.
  - /simplify accepted: the profile's three booleans are merged into the `serves` set, the config check uses one `UNUSED` table (the fields each role kind does not use), and `_embed_problems` is merged into
    `_problems`; `embed_encoding_format`; one `_RerankOwner` serves both sync / async (checking whether the return value of `post` is awaitable); the rerank request also goes through
    `_transport`; the count checks are merged into one; `InputType` / `RoleKind` moved to `types.py`; the fake's rerank fills None outside `top_k`; the tests use
    `local_validate`, `Upstream.body`, `embeddings(b64=)` and `ranked()`. Rejected: sending dimensions always through `extra_body` (it would change the request body of 0.6.0 hosts, rule 13);
    sharing `Caller` among tests and merging `with_vectors` / `with_scores` (beyond the scope of this change).
  - All three Codex findings were valid, and a failing test was written first for each: non-dict entries were filtered out and not counted (with `top_k=1`, one bad entry plus one good entry was treated as valid); a boolean passed
    the index and score checks; `top_k=0` was treated as no limit. Changed so that bad entries count toward the count and are rejected, bool is excluded, and `top_k < 1` raises `ValueError` before the request is sent.
    Codex also confirmed: when the SDK is not given `encoding_format` it requests base64 and decodes it; the request bodies of other types are unchanged.
  - pytest 401 passed; `-m ollama` 6 passed.
- **0.6.0 (2026-11-22)**: the embed role's `dimensions` / `batch_size` / `max_concurrency`; batch halving on 400/413; `close()` / `aclose()`
  (§3.2.1). All incremental changes. The reason was that lite-rag S19 wants to switch its ollama / openai embedders over, and llm-steward cannot close the SDK clients on exit.
  - Tests: `test_embed.py` adds dimensions in the request body, order under batching and concurrency, the full request sequence of halving, failure when a single item is still rejected, embed fields set on a chat role
    being rejected, an error on a call after close, an injected http_client not being closed, and the close of Bounded and fake. All seen red first; after changing `TOO_LARGE` to an empty set,
    the 4 halving tests went red. pytest 378 passed; `-m ollama` 6 passed. The mypy and pytest of every host (lite-rag, lite-mem, lite-cache, lite-agent, llm-steward)
    were run with the new build (KB testing-and-ci §15).
  - /simplify accepted:
    - The thread pool and `Bounded.map` share `in_threads`.
    - After one batch fails, queued batches are no longer sent: sync uses `cancel_futures`; async switched to `TaskGroup`, unwrapping to the first error. A failing test was written first.
    - The two checks for "the role set a field it does not use" are merged into `_set`.
    - The closing logic is put into `_Clients.release()`.
  - /simplify rejected: merging the sync and async halving logic (about 6 lines each, and forcing a merge would make it messier); no longer counting 400 as "too large" (when Voyage exceeds the token limit it returns 400).
  - Codex: no findings.
- **0.5.1 (2026-11-22)**: `FakeModel(vectors=)` changed to `FakeModel.with_vectors(vectors, /, **scripts)` (the same for `AsyncFakeModel`).
  - Reason: in the cross-repo check after 0.5.0 was released, running each host's mypy with 0.5.0 reported errors at lite-rag `testing.py:93` and lite-mem `testing.py:63`. They build the fake with `FakeModel(**{role: script})`, and mypy matches the dict's values against every keyword parameter, `vectors` included. `__init__` is restored to 0.4.0's `**scripts`; the vectors go to a classmethod that takes only positional parameters, so a role named `vectors` does not conflict either.
  - Test: `test_embed.py::test_scripts_unpacked_from_a_dict_still_type_check`, checked with mypy (3 errors before the fix, passing after it); pytest 363 passed.
  - Lesson: after a library changes a public signature, the cross-repo check must also run the hosts' mypy; running only pytest is not enough.
- **0.5.0 (2026-11-21)**: `embed()` (§3.2) and Anthropic `structured()` switched to `output_config.format` (§15).
  - Checks: ruff, ruff format and mypy all pass; pytest 362 passed; `-m ollama` 6 passed (including the equal vectors of `/v1/embeddings` and `/api/embed`).
  - New tests went red then green, and afterwards the implementation was broken line by line to confirm the assertions fail: `encoding_format`, sorting by index, provenance across batches, clearing of chat settings, an embed role not being routable, the count check, the Anthropic request shape, `Bounded.embed`.
  - Old tests changed on purpose: the four tests in `test_anthropic.py` that asserted the forced tool were rewritten for the new request shape.
  - Old test deleted: `test_only_the_forced_tools_input_is_read`. That test checked "only the forced tool is read", and there is no forced tool now.
  - `/simplify` (four passes: reuse / simplification / efficiency / altitude) accepted:
    - embed is called through the `EmbedBackend` protocol, and `_send` now must be passed the endpoint owner (before, it was a patch parameter `owner=None`).
    - `_route` infers the role kind from the method name.
    - `_merged` returns directly for a single batch; the usage total uses `_total`.
    - Vectors and inputs are no longer each copied once more.
    - The routing alias check is kept in only one place.
    - `target()` really clears the embed role's chat settings, so the docstring now holds.
    - "Empty vector" is a separate check.
    - `Bounded` / `AsyncBounded` get `embed()`, which was missing before.
    - The `embeddings()` response construction moved to `tests/upstream.py`.
    - `FakeReply` does not take vectors: putting them in made the `list` scripts already in hosts' tests fail type inference (mypy reported it on `test_bounded.py`). Changed to a separate `vectors=`.
  - `/simplify` rejected:
    - Merging the embed tests into `conftest.Caller`: it would need the embed role added to the `ROLES` of all contract tests, and the anthropic / gateway config tests would get extra embed errors.
    - List scripts for `FakeModel`'s embed support: not used.
    - Sending async batches in parallel: the design says batch by batch, so other batches are not wasted on failure.
    - A `strict_schema` cache: not part of this change.
  - Codex (`codex exec -s read-only`): both findings accepted, and a red regression test was written first for each:
    - Sorting only by index without checking the index meant a duplicate or missing index would pair a vector with the wrong input. Now after sorting it must be exactly `0..n-1`.
    - When the endpoint reports no usage, `output` also became `None`. An embedding has no output tokens, and it is now fixed at 0.
    - No regression was found in the hosts' existing `text` / `structured` / `turn` / hook / `FakeModel` call paths.

- **0.4.0 (2026-11-21)**: `turn()` (§3.1). Checks: ruff, ruff format and mypy clean; pytest all green; `-m ollama` 5 passed
  (including a real tool round, qwen3:1.7b). New tests went red then green; afterwards the implementation was broken line by line to confirm the assertions fail, and the one for "order of reply blocks"
  was not red at first, so a test of interleaved order was added.
  - `/simplify` (four passes: reuse / simplification / efficiency / altitude): accepted — the code block that sends the request was extracted into `_send` /
    `_send_async` (there were four copies); the `_params` of the two backends call `_history` themselves, and `request()` no longer writes the message mapping again; the stop rule
    was extracted into `types.stop_for`, shared by the backends and the fake; the fake's `_reply` takes the record list from the caller; the tests reuse the existing `Recorder` and
    the config of `test_anthropic`. Not accepted — the three repeated error messages in `finish` / `finish_turn` (the two places check in a different order, and extracting them would be harder to read);
    re-running `json.dumps` on every call's arguments in the history each round (the cost is tiny, and the change would alter the meaning of `raw_arguments`); reusing the LiteLLM test config
    (it would change the asserted values).
  - Codex read-only review (the diff of the whole stage): 1 High, 2 Medium, 1 Low. Fixed — an unknown Chat Completions `finish_reason`
    was treated as a normal end (High, regression test red first); an empty `ToolResults` sent Anthropic a user message with empty content (Medium, regression test
    red first); a contract test for a system message in the middle of the history was added (Low); the wording "byte by byte" was changed to "fields and values unchanged (dict round trip)". Partly rejected —
    "Mappings in frozen types are mutable, and should be deep-copied and frozen at construction": no deep freezing (`Provenance.via` already does the same, and deep-copying
    the whole history every round is not worth it), and instead the type annotation and §3.1 state that only one level is frozen.
  - Left to the owner: `supports_tools` is now True for all five types, and the rejecting branch can be reached only by patching in tests (both review passes pointed this out).
    It was an item in the approved plan, so it is kept; delete it if this extension point is not needed.

**S0 (2026-11-18) Environment and verification.**
- Skeleton: `pyproject.toml` (Python 3.12, uv_build; openai 3.46.0, pydantic 2.14.3, pydantic-settings 2.16.2,
  typer 0.27.3, extra anthropic 2.6.0, the same as lite-mem), ruff / mypy strict / pytest (markers `ollama`,
  `litellm` not selected by default), CI (checks + `compose run --rm verify`, which also parses and interpolates the compose file). ruff format does not touch `*.md`
  (the code blocks in the design are formatted sketches).
- `deploy/litellm/`: compose (`verify` → `litellm`), config (all tiers Ollama), `cosign.pub`, README.
  `tests/unit/test_deploy_litellm.py` guards these configs and that `uv.lock` has no `litellm`; every assertion was seen red once by breaking the matching
  config.
- curl verification results: the §5 parameter table, §6.3, §7.2, §7.3 and §9 were rewritten per the results; the details are in `KB/global/litellm/README.md`.
  Three points differ from the original design: (1) for a plain alias the response body `model` is the alias, and the actual model is taken from the `x-litellm-model-name` header (§6.3);
  (2) `context_compaction` is on by default and returns 500 for Ollama models, so it is turned off in the config (§7.2); (3) port 4000 → 4400 (conflict with tenderted).
- KB: new `KB/global/litellm/README.md`; `tech-selection/01b` corrected (`enable_context_window_escalation` defaults to
  false, `classifier_fallback`, one supply-chain item); `machine/ports.md` registers 4400.
- Close-out review:
  - `/simplify` (4 angles): the LiteLLM image is written in only one place (the YAML anchor `x-litellm-image`; cosign accepts `tag@digest`,
    measured); the public key is compared by sha256 and the mount is checked; the loopback check was changed to "all published ports"; the fallback check lost its special case;
    the redundant `compose config` in CI, `allowed-confusables` and `restart: "no"` were removed; the run details in §7.2 now point to the README.
    Skipped: CI `concurrency` (the scale is too small), caching the signature check result (verifying at every start is exactly the guarantee wanted).
  - Codex (read-only): 1 P2 finding: the S0 record said CI runs `compose config` while the workflow had dropped it, so the record was changed (not added back:
    `compose run` itself parses the whole file). No findings on supply chain, tests, design consistency or version pinning.

**S1 (2026-11-18) Core.**
- Modules: `errors`, `types`, `profiles` (the parameter table of five types), `config` (named backends, `target()` / `resolve()`
  and all the §4 startup validation), `schema` (`strict_schema`, the same transformation as lite-mem), `backends/openai_compat` (sync
  `text` / `structured`), `client` (`build()`), `testing` (`FakeModel`), `cli` (`dump-config`).
- The owner's decision: named backends (§4). What I decided: `build(http_client=...)` (§3); `dump-config` does not run validation; a backend type that the client has not written yet
  raises `ConfigError` in `build()`; `max_output_tokens` is not sent when not configured.
- Tests: 57 unit + contract tests, 3 `-m ollama` (text, structured, `Truncated`, qwen3:1.7b, real Ollama).
  Every test was seen red: 45 single-point changes were made to the production code (turning off each validation, changing profile values, taking `answered` from the config, errors carrying upstream
  text, no retry, each branch of `FakeModel`, turning off the network guard, ...), and at least one test failed for each; for this a test that tested nothing was fixed
  (`dump-config` had no role configured when no key was given), and a test that could never fail and a piece of dead code (`JSONDecodeError`) were deleted.
- Close-out review:
  - `/simplify` (4 angles): role overrides were changed to merge with `model_copy` (removing 12 lines of `_pick` and a None check that could never trigger);
    the profile's temperature default is resolved in `target()` (`dump-config` shows the value actually sent); the "backend of a later stage"
    check was merged into `resolve()`, listed at once with the other problems; `Target.where` unifies error location (rule 8); the profile table uses keyword arguments;
    one dict for `_Model`; the backend switched to `with_raw_response`, and provenance and usage are read in `_read(target, response, headers)` — S3
    reading response headers changes only this one place; the tests share the two-vendor config, and the retry test carries `retry-after-ms: 1` (suite 1.5 s → 0.5 s).
    Skipped: usage on failed calls (an S4 acceptance item, left to S4 together with its tests), dispatch by `protocol` and the async `FakeModel` (S2), a
    `strict_schema` cache (it returns a mutable dict, and the gain is microseconds). After the changes the 44 mutations were re-run, and all were caught.
  - Codex (read-only) 3 findings:
    - P2 an empty string is read as the default by `or`: valid (the consequence is not what Codex said, bypassing the key check — `""` falling back to `OPENAI_API_KEY` is still
      checked — but that the intent of "no key" is silently rewritten). The regression test went red first, and the fix is to not accept empty strings.
    - P2 `temperature` has no range limit: valid, the regression test went red first, and the fix limits it to 0–2.
    - P1 the gateway's `reasoning_effort` should raise `IgnoredParameter` from `X-LLMGW-Ignored-Params`: **rejected (belongs to S3)**.
      Until llm-gateway is fixed, rejecting it at config time is exactly what §5 and §15 require; the gateway backend is not usable in S1; the response header check is an S3 acceptance item.
  - Undecided (not changed): an endpoint of `type: openai` cannot declare "needs no key" (something like a local vLLM). To be added when a host really needs it.
**S2 (2026-11-18) Anthropic and async.**
- Structure: one module per protocol (`openai_compat`, `anthropic_messages`), with only the `request()` / `reply()` that do not touch the network,
  plus `connect` / `connect_async` / `endpoint`; the shared provenance and error judgment are in `backends/__init__.py` (`provenance`,
  `finish`); `client.py` has only one sync executor and one async executor, looking up by `profile.protocol` in a table. `build_async()`,
  `AsyncModel`, `AsyncFakeModel`.
- The Anthropic request body is the same as what lite-mem / lite-rag send now (§5). The owner's decision: `anthropic` must be configured with `max_output_tokens`.
  A new startup check: an `anthropic` `base_url` must not end in `/v1` (found while writing tests: giving the same URL to both vendors' SDKs would request
  `/v1/v1/messages`). SDK clients are shared by (type, endpoint, key variable, timeout, retries), not across protocols.
- SDK measurements (recorded in `KB/global/llm-evals/providers/`): anthropic 2.6.0 sends the key with `authorization: Bearer`, with no
  `x-api-key`; anthropic's async `raw.parse()` is a coroutine and openai's is not (`inspect.isawaitable` in the executor);
  anyio's async transport connects inside a TaskGroup, so the network guard's error is wrapped in an `ExceptionGroup` (the tests unwrap it).
- Tests: 115 unit + contract tests (the contract tests run once sync and once async), 4 `-m ollama` (one async structured added).
  Anthropic has only contract tests (a paid API, AGENTS.md rule 11). Mutations: 21 in the new S2 code and the 44 of S1 (re-pointed after the refactor)
  were all caught.
- Close-out review (S2):
  - `/simplify` (4 angles): building the request was moved outside `try` (local errors are no longer wrapped into `CallFailed`); the two backends share
    `sdk_client()` (both vendors' SDK `Timeout` is `httpx2.Timeout`); `finish()` takes `answered` and builds the provenance itself;
    anthropic's two checks go by `profile.protocol` instead of `type`; the tests' `Caller` lost its branches, and `Connect` is a typed
    Protocol; `_causes` was replaced by `traceback.format_exception`. Skipped: a `strict_schema` cache (microseconds, and it would share a mutable
    dict), a `_Routes` base class (harmless), splitting the executor for S4's hook (to be split in S4). After the changes 65 mutations were all caught.
  - Codex (read-only) 3 findings:
    - High: only `max_tokens` counted as truncation, the partial text of `refusal` / `pause_turn` was treated as a successful reply,
      and `model_context_window_exceeded` did not count as truncation. Valid, and the same gap exists on the OpenAI side with `content_filter`. The regression tests went red first
      (14 of them), and the fix: `model_context_window_exceeded` is added to the truncation set; a stop_reason outside the complete set, and OpenAI's
      `content_filter`, report `EmptyReply("no usable reply (stop reason …)")` (carrying only the enum value, not the upstream text). No new
      exception class was added (the public API is unchanged).
    - Medium: a structured reply took the first `tool_use` without looking at the name. Valid; only the forced tool is read; the regression test went red first.
    - Medium: the SDK client should be shared by the key's value and not the variable name. **Rejected**: by variable name is intentional (no secret value goes into the sharing key); the gain
      is only one fewer client built when "two variables hold the same key". "key" in the record was changed to "key variable".

**S3 (2026-11-18) gateway and litellm.**
- The two types use the same OpenAI-compatible module, and the differences are all `BackendProfile` data (the §5 table): `answered_header`,
  `via_headers`, `ignored_params_header`, `request_headers`, `role_tag`. The owner's decision: `Provenance` gets `via`.
- Provenance: §6.2, §6.3 (litellm reads headers and raises `ProvenanceUnknown` when it cannot get them; gateway reads the profile name, and a routing alias is rejected at build).
  `IgnoredParameter` applies only to fields lite-model actually sent. Every litellm request carries `x-litellm-num-retries: 0` and
  `metadata.tags: ["role:<role>"]`. The S1 gate "only available in a later stage" was removed (the corresponding tests were replaced by two: "can build" and "a routing alias
  on the gateway is rejected" — that gate was only temporary anyway).
- Tests: 156 unit + contract tests; 4 `-m ollama`; 3 `-m litellm` (local proxy, Ollama alias, `channel: local`).
  The expectation of one live test was changed: the router sent "新西兰最大城市" (the largest city in New Zealand) to qwen3:1.7b, which answered Wellington — the test measures the path and not the model's
  knowledge, so it was changed to an arithmetic question. Mutations: 15 in S3, 43 in S1, 21 in S2 were all caught.
- Close-out review:
  - `/simplify` (4 angles): the three response-header rules (read answered, record `via`, `IgnoredParameter`) moved from `openai_compat` into the
    protocol-independent `finish()`, and use the same parameter-table data as the startup checks (otherwise an anthropic-protocol
    profile configured with `answered_header` would let routing through and then record the alias on the reply); `reply()` receives the request actually sent, and `IgnoredParameter` checks against it (no longer recomputing
    "what was sent", nor computing `strict_schema` again; S4's hook needs it too); the header names in the parameter table are all lower-case; the live tests read
    `.env` with `python-dotenv`. Skipped: moving the test config factories into `tests/configs.py` (only one file uses them).
  - Codex (read-only):
    - P1 for a plain alias, when the value of `x-litellm-model-name` is the alias itself it was also taken as the actual model: valid, the regression test went red first, and it now raises
      `ProvenanceUnknown`; the cost is that in the proxy config an alias must not have the same name as its model string, and a deployment test was added to guard it (§6.3).
    - P2 looking up the header depends on the SDK's case-insensitive mapping: valid (actual runs are not affected), `finish()` now lower-cases the header names first, and the unit test with a
      plain dict went red first.
    - Docs: §8's `ProvenanceUnknown` mentioned only routed calls, and it now covers any litellm call.
  - Ollama stopped during the close-out (not stopped by lite-model); after the owner agreed it was restarted with `brew services run ollama`, and the re-run:
    `-m ollama` 4 and `-m litellm` 3 passed; the 5 live mutations (with Ollama stopped they only fail to connect, which does not count) were all caught by the matching
    assertions.

**S4 (2026-11-18) Usage, cost, concurrency, hooks.**
- Usage: the exceptions for rejected answers all carry `usage` (including cost), and `CallFailed` has `None`. Cost: the cost header the endpoint reports > `prices` (by the model that actually
  answered) > `None`; subscription is `None`. `Price`, `ModelsConfig.prices`, `BackendProfile.cost_header`.
- `Bounded` / `AsyncBounded`; `hooks.py` (`Call`, `Hook`, `Hooks`, `Registration`); `LiteModel` /
  `AsyncLiteModel`; the reply gets `cached` (the owner's decision: put it on the reply).
- Tests: 202 unit + contract tests, 4 `-m ollama`, 3 `-m litellm`. While writing tests one test that tested nothing was found:
  `Bounded`'s concurrency limit was tested only through `map()`, and the thread pool of `map()` itself has only `concurrency` threads, so it passed even with the semaphore removed; a test with
  8 external threads sharing one `Bounded` was added. Mutations: 20 in S4, and 43 in S1, 21 in S2, 18 in S3 (after re-pointing) were all caught.
- Close-out review:
  - `/simplify` (4 angles): a hook returning a reply of the wrong type raises `TypeError` (a `cast` would have hidden the error until much later);
    `schema_hash` uses the same rules as lite-eval's `canonical.py` (NFC, `ensure_ascii=False`, `allow_nan=False`),
    copied and not imported (rule 9), with the test checking against the written-out lite-eval algorithm; `Hooks` lost the `__iter__` / `__len__` that had no basis in the design
    (the test now asserts the `KeyError` of `hooks[name]`), and `Registration` goes through `Hooks._remove`; the triple of `_before` in the executor was split into
    `call = ... if hooks else None` plus `_hit()`; `_awaited` was removed (the function of `map` is a coroutine); `_number` was
    simplified. Another test that tested nothing was found: with `concurrency=1`, `map()` is a plain loop and "one at a time" always holds — changed to 8 external
    threads, parameterized with 1 and 2 slots. Skipped: a hook snapshot, a `Call.params` cache, a shared thread pool (all microseconds). Recorded in §15:
    `Call` lacks lite-cache's scope / policy; a hit also takes a `Bounded` slot. §9 states that the `usage` of a hit is what was saved.
  - Codex (read-only) 5 findings:
    - High: `Call.request` is the very dict later sent to the SDK, so a hook that changes it can make the cache key differ from the actual request. Valid, the regression test
      went red first; a hook now gets a read-only deep copy (copied only when there is a hook).
    - High: when a hook's reply type is wrong, the `TypeError` used `hit!r`, which would include the reply content (rule 8). Valid, red first; only the type name is written.
    - Medium: `Price` accepts `inf`, and the product can overflow to `inf`. Valid, red first; `allow_inf_nan=False`, and a non-finite result is recorded as `None`.
    - Medium: when `after()` raises, or reply parsing raises a non-`ModelError`, `usage` is lost. **Rejected**: by design a hook's exception is raised as it is, and attaching an attribute to an arbitrary
      exception is not clean; `after` should not raise, which is written into §10. A shape error in reply parsing is guaranteed by the SDK's types.
    - Medium: `Call` lacks scope / policy. **Already in §15**, to be decided when lite-cache is connected.
  - After the changes 112 mutations (S4 30, S1 43, S2 21, S3 18) were all caught; `-m ollama` 4 and `-m litellm` 3 passed.

**S5 (2026-11-18) Release and whole-library close-out.**
- Packaging: version `0.1.0`; `license-files = ["LICENSE"]` (the packaging test went red first: both the wheel and the sdist lacked the LICENSE text);
  `tests/unit/test_packaging.py` checks the LICENSE, `License-Expression`, `py.typed`, the `litemodel` entry point, that `testing.py`
  is present and `tests/` is not. CI: actions are pinned by commit (the same set as lite-eval); the `package` job builds, checks that the tag version
  matches the wheel, and in a clean venv installs the wheel and `wheel[anthropic]` and runs `dump-config` and `scripts/smoke.py`; the `release` job runs only on
  `v*` tags, is the only one with `contents: write`, and publishes the very artifact the `package` job built and tested (the owner chose CI release by tag).
- The README is written in full (install, quick start, config, backend table, provenance, routing, errors, usage, concurrency, hooks, testing hosts);
  `tests/unit/test_readme.py` makes the README's config example and quick start follow the code (during mutation it was found not to test the default backend, and that was added).
- Whole-library close-out (the scope is all the code since `f13239a`, not only S5):
  - `/simplify` (4 angles) settled the public API before release (two decisions by the owner, see the decision log): `call_context()` passes per-call
    values to the hook (the `Model` protocol gets no parameter); `ModelsConfig.backend` / `model` are required. Also: `LiteModel.target(role)`;
    top-level exports were narrowed (hook types are imported from `lite_model.hooks`, and `Reply` and `call_context` are public); `Reply` is defined in only one place
    (the one in `testing` was renamed `FakeReply`); the two sets of "transport fields" were merged into one; `Call.role` comes from the target; anthropic's two
    checks became parameter-table fields (`extra`, `base_url_is_root`); comments in the code that refer to review rounds were deleted; §6.1 states how lite-mem's
    `provider:model` format is kept (the backend is named `openai` with `type: ollama`, with a test). The CLI reports a clear
    error instead of a traceback when backend / model is missing. Skipped: import time, a `strict_schema` cache, building twice in CI (all milliseconds or parallel).
  - Codex (read-only, whole library) 2 findings:
    - Medium: an unknown anthropic `stop_reason` went as it was into the message of `EmptyReply` (rule 8). Valid, red first; only the known
      `refusal` / `pause_turn` are written, and the rest are written as `unexpected`.
    - Medium: the packaging test runs `uv build` in a subprocess, which the network guard cannot see. Valid, changed to `uv build --offline` (the build backend
      is already in the cache after `uv sync`). This one has no test that went red first: this machine's cache has `uv_build`, so it would pass even without `--offline`;
      what `--offline` guarantees is failing when the cache is missing, rather than going online.
    - The status in the first line of the design was outdated: changed. No other findings on supply chain, provenance, secrets, the public API, the release process or the host migration path.
- Checks: 226 unit + contract tests, 4 `-m ollama`, 3 `-m litellm`; 136 mutations (S1 43, S2 21, S3 18, S4 30,
  S5 13+2, including 5 live ones, with Ollama online) were all caught.
- Release: when `main` was first pushed (`131c850`) the CI `package` job failed — its `dump-config` smoke step was written before
  backend / model became required, and was run locally at that time; the two variables were added (`767473d`), the CI packaging steps were re-run locally from a clean `env -i` environment
  before pushing again, and CI was all green (the lesson was recorded in the KB). Then the tag `v0.1.0` was pushed: all four jobs of CI on the tag passed, and `release` published
  `lite_model-0.1.0-py3-none-any.whl` and `lite_model-0.1.0.tar.gz`
  (https://github.com/ArthurGao/lite-model/releases/tag/v0.1.0).
- Acceptance "a host can install from the wheel": a new project points `[tool.uv.sources]` at the release wheel URL and installs
  `lite-model[anthropic]` (lite-model 0.1.0, openai 3.46.0, anthropic 2.6.0); against the local Ollama: `text` returned `pong`,
  the provenance was `ollama:qwen3:1.7b`, with usage; `structured` through `build_async` + `call_context` returned `result=42`.

**0.1.1 (2026-11-18) `keyless` backend.** Needed by the lite-mem migration (its S12): it now connects to Ollama with the OpenAI protocol, sends no temperature
and needs no key; `type: ollama` would additionally send `temperature: 0.0`, and `type: openai` needs `OPENAI_API_KEY`. The owner chose to add `BackendConfig.keyless`
(rather than accept a change in the request body). Tests went red first (config parsing, a request with the placeholder key and no temperature sent, the conflict with `api_key_env`), and both mutations were caught.
About 12 lines of production code in one file: under the global rules this is a small change, so no `/simplify` or Codex close-out was done.

**0.2.1 (2026-11-19) The source of the key is one choice.** When lite-mem switched to 0.2.0 `host_backends` it was found that: the host's `openai` is keyless, and when the config writes only
`api_key_env` the two stack, and startup reports "keyless and api_key_env cannot both be set". Now when stacking, if the config writes `api_key_env` or
`keyless` the host's two are removed; writing both in the same entry still raises an error. The tests went red first; about 6 lines of production code in one file, and under the global rules no close-out was done.

## 15. Open questions

- **`turn()` and hooks** (0.4.0): it does not go through hooks now. When a host wants to cache or observe an agent's rounds, extend the `Hook` protocol then.
- **Reasoning content of Chat Completions endpoints** (0.4.0): Ollama, DeepSeek and others return thinking content in non-standard fields, and some require it
  to be passed back between tool rounds. 0.4.0 neither collects nor passes it back (these backends produce no `Opaque`); when connecting such a model for an agent, it must first be verified and added.
- ~~**The Anthropic backend's `structured()` uses a forced tool**~~: 0.5.0 switched to `output_config.format` (json_schema, with the schema still the
  strict schema; the reply is read from the JSON made of the joined text blocks). Reason: the Claude API reference states that on Opus 5.5, Sonnet 5.5, Fable 5.1 and later
  models a forced `tool_choice` is a 400; structured output works on all current models (including the 4.6 series). The hosts' Anthropic structured request bodies therefore
  change (lite-mem `tests/unit/test_llm_requests.py` was changed to follow). Only contract tests, not verified on a real endpoint (paid APIs are not called).
- **400 for preserved thinking** (§3.1): it currently lands in `CallFailed`, and the caller cannot tell it apart from other failures. Add a dedicated error when needed.

- **Two changes to llm-gateway** (made in the llm-gateway repo, not in this repo):
  1. `reasoning_effort` pass-through (currently dropped; the `"none"` set by lite-rag graph and lite-mem eval does not take effect through the gateway,
     and results produced through the gateway actually have thinking on).
  2. Pass-through of the actual upstream model (a suggested response header `X-LLMGW-Upstream-Model`, with the response body `model` unchanged).
  Until then, lite-model for the gateway backend: rejects a configured `reasoning_effort` (§5), and rejects gateway + a routing alias (§6.2).
- Whether embedding is brought in in v2 (lite-rag and lite-mem each have an httpx2 embedding client). An embedding is tied to the index
  and is never routed.
- ~~Whether the lite-eval judge migrates later~~: decided by the owner on 2026-11-21, no migration. The judge is a measuring instrument and stays on a direct connection, written as an explicit
  exception of the workspace (lite-eval design §9, lite-eval AGENTS.md rule 9).
- ~~Hooks have only `before` / `after`, and failed calls do not go through hooks~~: decided on 2026-11-19, no `failed()` is added. lite-cache plugs in as a hook instead,
  does not count failed calls, and `skip:<reason>` comes only from its own `accept` predicate (lite-cache design §0, §3.4).
- ~~`Call` lacks lite-cache's scope / policy~~: S5 settled it as `call_context()` (§10). lite-cache's hook must not cache when there is no
  scope (otherwise one tenant's answer would be given to another tenant).
- ~~`Bounded` wraps outside the model: a hook hit also waits for a slot~~: 0.3.0 moved the slot into the executor (§10, 2026-11-20).
- ~~**Hooks for async hosts**~~: 0.3.0 added `AsyncHook` (§10, 2026-11-20).
- **LiteLLM Proxy cache hits** (2026-11-19): the Proxy's cache is not turned on (§7.2); 0.3.0 reads `x-litellm-cache-key` to mark
  `cached=True`. Verified against the source and measured against the Proxy on 2026-11-20: 1.108.9 carries this header only on a hit, and on a hit `x-litellm-model-name`
  is still the model that originally answered (§7.2).
- ~~The effect of automatic routing~~: **measured** (2026-11-18, lite-rag S14, `KB/global/llm-evals/evals/2026-11-18-lite-rag-auto-routing.md`).
  lite-rag's 32 golden questions × 2 runs, "correct" = key fact and grounded: fixed qwen3:8b 0.938 (p50 12 s), the same model through the Proxy 0.938,
  qwen3:1.7b 0.750 (3.8 s), `smart-router` (heuristic, small = 1.7b, the rest = 8b) 0.844 (5.2 s). The heuristic classifier tiers **by language and not by difficulty**:
  all Chinese questions go to 1.7b. Conclusion: the current §7.2 config is not worth turning on for mixed Chinese-English RAG; for routing, first measure the `llm` classifier or
  `keyword_tier_rules` on real traffic. The Proxy itself does not change answer quality.
- Whether each module drops its adapter and depends directly on `lite_model.Model` when it migrates (§12.1).
- ~~**A host cannot change a built-in backend**~~ (found in lite-mem S12, 2026-11-18): done in 0.2.0 (§4, the host's own defaults). lite-mem switched to
  `host_backends` on 2026-11-19 and deleted its own validator (`be5269f`).
- ~~**keyless conflicts with `api_key_env` when stacked**~~ (found when lite-mem switched to 0.2.0, 2026-11-19): done in 0.2.1 (§4).
- **Constrained decoding for structured output** (lite-rag S16, 2026-11-19): when qwen3:8b on Ollama outputs JSON freely, 7 of 10 runs on the same chunk missed a quote;
  through `structured()` (strict JSON schema) all 10 were valid. When a host wants JSON it should use `structured()` and not parse text itself.

### Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11-18 | Create the shared library lite-model, which lite modules depend on, removing the 4 duplicate LLM clients | Routing support should not be written again in every module |
| 2026-11-18 | LiteLLM is used only in Proxy mode, as an optional backend; the library does not import `litellm` | Two supply-chain attacks in 2026; the dependency would pass on to llm-steward; the routing config is shared |
| 2026-11-18 | Without LiteLLM the original direct connection is used; when LiteLLM is configured and unavailable it raises an error and does not fall back | The lite modules must be usable alone; no silent fallback |
| 2026-11-18 | llm-gateway does not depend on lite-model; the call chain is app → lite-model → llm-gateway | The gateway is a server, and lite-model is a client |
| 2026-11-18 | The name is `lite-model` (not `lite-router` / `lite-llm`) | It is mainly a client and not a router; `lite-llm` is easily confused with `litellm` |
| 2026-11-18 | GitHub `ArthurGao/lite-model`, public, Apache-2.0 | Consistent with the other lite modules |
| 2026-11-18 | After migration a module deletes its own LLM client, keeping only the role declarations, prompts, business parsing and a thin adapter (§12.1) | The purpose of a shared library is to remove duplication; having the old and new copies side by side would diverge again |
| 2026-11-18 | LiteLLM Proxy port 4400 | 4000–4010 and 4080–4090 are already registered to tenderted |
| 2026-11-18 | cosign runs as a container, as the compose `verify` service, and `litellm` depends on it exiting successfully | The owner's choice: cosign is not installed on this machine, and one `compose up` verifies the signature before starting |
| 2026-11-18 | LiteLLM's `answered` takes the response header `x-litellm-model-name`, and the response body `model` is used only for a consistency check (§6.3) | Measured in S0: the response body `model` of a plain alias is the alias, and the header is the actual model |
| 2026-11-18 | Named backends: `backends.<name>: {type, base_url, api_key_env, channel, router_aliases}`, and roles refer to them by name (§4) | The owner's choice: one host must be able to use two OpenAI-compatible vendors at the same time; the endpoint and key are written in one place |
| 2026-11-18 | `anthropic` must be configured with `max_output_tokens`, checked at startup, with no guessed default (§4) | The owner's choice: a host writes its own old value when migrating (lite-mem 2048, lite-rag 1024), and the request body is unchanged |
| 2026-11-18 | `Provenance.via`: the endpoint's own provenance headers (the allow list in the parameter table, lower-case names); `tag` unchanged | The owner's choice: §6.2 and §6.3 require recording the deployment id, alias and gateway backend separately |
| 2026-11-18 | The `cached=True` of a hook-hit reply is put on the reply, and `Provenance` is unchanged (§10) | The owner's choice: whether it was cached belongs to this call, and the provenance belongs to the model that originally answered |
| 2026-11-18 | `build()` returns `LiteModel` (`Model` + `.hooks`); the `Model` protocol has no hooks (§3, §10) | `FakeModel` and hosts' adapters need not implement hooks; `model.hooks.register` is usable as designed |
| 2026-11-18 | Call context uses a contextvar: `call_context(**values)` → `Call.context`; the `Model` protocol gets no parameter (§10) | The owner's choice: three hosts are about to write adapters against `Model`, and adding a parameter later would be a breaking change |
| 2026-11-18 | `ModelsConfig.backend` / `model` are required, with no default (§4) | The owner's choice: when a host embeds it, it does not silently inherit a default, and a lite-model version never changes a host's model |
| 2026-11-18 | The public API is narrowed: `Call`, `Hook`, `Hooks`, `Registration` are imported from `lite_model.hooks`; `Reply`, `call_context`, `LiteModel.target()` are public | After release, every exported name is a promise; a host migration needs the target before the call |
| 2026-11-18 | Heuristic automatic routing is not turned on for mixed Chinese-English RAG (§15 measured: 0.844 versus 0.938, tiering by language) | Two real runs in lite-rag S14; to route, first measure the classifier |
| 2026-11-18 | `build(..., http_client=...)` is a public parameter (§3) | The host migration contract tests need the real SDK to record the request body (§12.2); lite-model's own contract tests do the same |
| 2026-11-18 | 0.2.0: the host sets `host_backends` / `host_roles` on a `ModelsConfig` subclass, and the config is laid on top field by field; only a different `type` replaces a backend | lite-mem S12's default role was once pushed to api.openai.com by the built-in `openai`; lite-rag's graph default would be overridden wholesale by `__ROLES`. Review: writing the same `type` again must not lose the host's values; a role's null still means "use the top-level value" (the same as 0.1) |
| 2026-11-19 | For caching, lite-cache is chosen as a hook, storing in PostgreSQL; the LiteLLM Proxy cache is not turned on and Redis is not used (§7.2, lite-cache design §0) | Default calls do not go through the Proxy; on a Proxy hit lite-model cannot mark `cached` and the usage is counted wrongly; there is no required scope; Redis gives no speed benefit for 1–20 s model calls, and is new infrastructure |
| 2026-11-19 | The 0.3.0 plan: `AsyncHook`, moving the `Bounded` slot into the executor, reading `x-litellm-cache-key`; no `failed()` hook (§10, §15) | Needed before lite-cache is connected to async llm-steward and lite-rag; lite-cache does not count failed calls |
| 2026-11-20 | 0.3.0: when `Bounded` wraps `LiteModel` the slot wraps only the request (`limited(slot)`), and when it wraps another `Model` the whole call as before; a sync model rejects `AsyncHook`; `x-litellm-cache-key` decides a hit per the 1.108.9 source (§7.2, §10) | Only the `LiteModel` executor changes, and `FakeModel` and hosts' adapters need not implement slots; the L2 verifier has to call a model inside a hook through the same `Bounded`, and having the whole call take a slot would deadlock |

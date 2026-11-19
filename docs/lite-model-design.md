# lite-model 设计文档

状态：S5 完成（v0.1.0，2026-11-18）；宿主迁移完成（2026-11-19）；0.3.0 实现完成（2026-11-20：`AsyncHook`、`Bounded` 槽位只包请求、`map` 带 context、Proxy 缓存头），待发布。阶段与验收见 §14。

## 1. 概述

lite-model 是 lite 系列模块（lite-rag、lite-mem、llm-steward，以后的 lite-cache）**共用的模型调用层**：
调用点只说 role（`answer`、`extract`……），配置把 role 指向某个后端上的某个模型；lite-model 负责发请求、
处理各家参数差异、拿回结构化结果，并记录**实际是谁回答的**。路由（按请求难度自动选模型）由可选的
LiteLLM Proxy 做，lite-model 只把它当作一个后端，并在客户端补上路由需要的护栏和来源记录。

命名：repo `lite-model`，Python 包 `lite_model`，CLI `litemodel`，环境变量前缀 `LITEMODEL_`。

### 来源

- 2026-11-18 的讨论：4 个模块各有一份几乎相同的 LLM 客户端（lite-rag `llm.py` 178 行、lite-mem `llm.py`
  198 行、llm-steward `llm/client.py`、lite-eval `judges/client.py`），都在做 provider 差异、来源记录、
  错误包装和用量统计；引入 router 会再让每份各加一遍。
- 主人的决定（§15）：LiteLLM 只用 Proxy 模式，作为可选后端；lite 模块可以单独使用，没有 LiteLLM 时走原来的
  配置方式。
- 起点代码：lite-mem `src/lite_mem/llm.py`（role、strict schema、Anthropic 强制工具、来源取自响应）和
  lite-rag `src/lite_rag/llm.py`（`PROVIDERS` 数据表、`BoundedCalls`、空回复报错）。
- 规则来源：`KB/global/llm-app-architecture.md` §1（role 层）、§3.2（不做 fallback 链）、§10（共享模块的
  可插拔扩展）；`KB/global/tech-selection/01b-llm-gateway-litellm.md`（LiteLLM 选型，未核实）。

### 目标

1. 一套接口：`text()` 与 `structured(schema)`，同步和异步。
2. role 配置：默认值 + 每个 role 的覆盖；启动时校验，配置错就拒绝启动。
3. 后端：`ollama`、`openai`（任意 OpenAI-compatible 端点）、`anthropic`、`gateway`（llm-gateway）、
   `litellm`（LiteLLM Proxy）。各家差异是数据，不是分支。
4. 来源：每次回答记录 role、请求的别名、**实际回答的模型**、后端、通道（本地 / API / 订阅）。
5. 路由支持：只有声明为可路由的 role 能用路由别名；路由别名的回答记录实际模型。
6. 错误：不做 fallback；一个错误层级；截断、空回复、结构不符都报错；错误信息不带密钥。
7. 用量：input / output / reasoning token，失败的调用也记录；可选按价格表算成本。
8. 扩展点：调用前后的 hook（lite-cache 用它做缓存），注册可撤销、可按名字寻址。
9. 测试工具：`FakeModel`（按 role 脚本化回复、记录收到的请求）。

### 非目标（v1）

- **不包装 `litellm` Python 包**，也不在进程里跑路由。路由只在 LiteLLM Proxy 里做（理由见 §7.1）。
- 不做换模型的 fallback；不做同一模型简单重试以外的重试。
- 不执行工具、不跑 agent loop、不管会话状态。
- 不管 prompt 模板（留在各模块的 `prompts/`），不做缓存本身（交给 lite-cache）。
- 不做 streaming（4 个模块都没用到）。
- embedding 留到 v2（lite-rag、lite-mem 各有一份 embedding 客户端，见 §15 待确认）。
- lite-eval 的 judge 暂不迁移：它故意用 httpx2 直连、走 Ollama 原生 `/api/chat`，judge 作为测量工具要固定、
  隔离。

## 2. 位置与调用链

```
app (lite-rag / lite-mem / llm-steward / later lite-cache)
  └─ lite-model                         client lib: role -> target, provenance, guards
       ├─ HTTP → Ollama /v1             local models
       ├─ HTTP → OpenAI / Anthropic     API billing
       ├─ HTTP → llm-gateway :8787      subscription profiles (claude-sub-opus / -haiku, gpt-sub), owner-only
       └─ HTTP → LiteLLM Proxy :4400    unified aliases, auto router (optional)
                   └─ may forward to llm-gateway or vendors
```

- **依赖方向**：lite 模块 → lite-model。lite-model 不依赖任何 lite 模块，也不依赖 llm-gateway 或 LiteLLM
  的代码；它们只是 HTTP 端点。
- **llm-gateway 不 import lite-model**：gateway 是服务端（协议转换、订阅后端），lite-model 是客户端；
  互相 import 会让客户端接口的改动牵动服务端。两边只重复了一张小的参数差异表，可以接受。
- **lite-cache 反向接入**：lite-model 提供 hook 协议（§10），lite-cache 实现它；lite-model 不依赖 lite-cache。
- **可选**：不配 `gateway` / `litellm` 时，lite-model 就是直连 Ollama / OpenAI / Anthropic，与各模块现在的
  行为一致。配了 `litellm` 而 proxy 不可用时**报错**，不会退回别的后端（no silent fallback）。

## 3. 接口

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

- 调用方只传 role 和 messages；model、endpoint、key、参数都来自配置。
- `http_client`（S1 加）：替换 SDK 的 transport。宿主迁移时的契约测试（§12.2）用它记录真实 SDK 发出的请求体；
  也可以用来接代理。lite-model 自己的契约测试同样这样做（`tests/upstream.py`）。
- 各模块可以保留自己的 `LLM` Protocol（lite-rag 规则 1），用一个薄 adapter 包 lite-model；也可以直接
  依赖 `lite_model.Model`。迁移时各模块自己选（§12）。
- `RoleSpec`：宿主模块声明自己有哪些 role、哪些可路由（§7.3）、哪些需要结构化输出。例：

  ```python
  ROLES = RoleSpec({
      "answer": RolePolicy(routable=True),
      "contextual": RolePolicy(routable=False),   # part of the index identity
      "graph": RolePolicy(routable=False),
  })
  ```

## 4. 配置

lite-model 提供 pydantic 配置模型，宿主模块把它嵌进自己的设置，环境变量用宿主的前缀
（`LITERAG_LLM__...`、`LITEMEM_LLM__...`）。lite-model 自己的 `LITEMODEL_` 前缀只给 CLI 和测试用。

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
forbid_channels: []         # tests / CI set [subscription, api]
prices: {}                  # S4: "backend:model" -> {input_per_mtok, output_per_mtok}
```

**命名 backend**（S1 定，主人 2026-11-18）：`backends` 下每一项有名字和 `type`（`ollama`、`openai`、
`anthropic`、`gateway`、`litellm`），`type` 决定协议和 §5 的参数表；`base_url`、`api_key_env`、`channel`、
`router_aliases` 不写就取 `type` 的默认。role 用名字引用 backend，于是一个宿主可以同时用两个
OpenAI-compatible 厂商（DeepSeek 和 DashScope 都是 `type: openai`），endpoint 和 key 只写在一处。五个内置名字一直存在。

`keyless: true`（0.1.1）：这个 endpoint 不要 key（例如把 Ollama 的 `/v1` 当 `type: openai` 用，什么 Ollama 专有的参数
都不加），SDK 拿到占位 key；和 `api_key_env` 同时写会被拒绝。

**宿主自己的默认值**（0.2.0，lite-mem S12 发现）：宿主在 `ModelsConfig` 的子类上设 `host_backends`（盖在内置 backend
上面）和 `host_roles`（每个 role 的调用参数）。配置给的值叠在上面：同名的 backend 或 role 逐字段改它（例如只改内置
`ollama` 的 `base_url`；role 里写 null 去掉宿主给这个 role 的值，这个 role 回到顶层的值），只有写了**另一个** `type` 的 backend 才从头开始——同样的 `type` 照写一遍
（`..._OPENAI__TYPE=openai`）不会丢掉宿主的 `base_url`、`keyless`。于是配置**加**一个 backend 或改一个 role，不会把它
没提到的默认值弄丢。key 从哪里来（`api_key_env` 或 `keyless`）算一个选择（0.2.1）：配置写了其中一个，就替换宿主的那一个——
在宿主的 keyless backend 上只写 `api_key_env` 就是改成要 key，不用再写 `keyless: false`。0.1.x 里配置了任何 backend 都会让同名的内置值顶掉宿主的，lite-mem 的默认 role 因此会被送到
api.openai.com。

每个 role 解析成一个 `Target`（frozen）：backend（名字）、type、model、base_url、api_key_env、channel、
routed、reasoning_effort、max_output_tokens、temperature、timeout_s、connect_timeout_s、max_retries。endpoint
和 key 属于 backend，所以 role 换了 backend 就换了 endpoint 和 key，不会沿用默认 backend 的。
`max_output_tokens`、`reasoning_effort`、`temperature` 不配就不发（`temperature` 取 profile 默认，Ollama 0），
这样宿主迁移时请求体不变（lite-mem 不发 `max_tokens`，lite-rag 发 1024）。
字符串字段（backend、model、base_url、api_key_env、router alias）不接受空字符串：空值不会被悄悄读成
"用默认值"。`temperature` 限 0–2。

**启动时校验（`build()` 里，失败抛 `ConfigError`）**，依据 `KB/global/llm-app-architecture.md` §1.5：
- 配置里出现宿主没声明的 role；宿主声明的 role 解析不出 target（例如引用了不存在的 backend 名字）。
- 缺 key（key 环境变量没设或为空）。
- `anthropic` 后端但没装 extra `anthropic`。
- 路由别名用在不可路由的 role 上（§7.3）。
- target 的 channel 在 `forbid_channels` 里。
- 后端不支持的参数（例如 `anthropic` 配了 `temperature`，见 §5 的参数表）。
- 所有问题一次列出（一个 `ConfigError`，每条一行），不是遇到第一条就停。错误里只有 role、backend、model 和
  key 的变量名，没有 key 的值。
- `anthropic` 没配 `max_output_tokens`（Messages API 必须带 `max_tokens`；不猜默认值，宿主自己写，lite-mem 2048、
  lite-rag 1024，迁移时请求体不变；主人 2026-11-18）。
- `anthropic` 的 `base_url` 以 `/v1` 结尾：Anthropic SDK 自己拼 `/v1/messages`（OpenAI SDK 要 `/v1` 在
  `base_url` 里），照抄 OpenAI 风格的 URL 会请求 `/v1/v1/messages`，到第一次调用才失败。
- 路由别名用在没有 `answered_header` 的 backend type 上（gateway：拿不到实际模型，§6.2）。

`litemodel dump-config`：读 `LITEMODEL_*` 环境变量和 `.env`，打印合并后的配置和每个已配置 role 的 target
（key 只显示变量名），不发任何请求，也**不跑**启动校验（缺 key 时照样能看配置；KB §10.3）。

## 5. 后端

| backend | 协议 | channel 默认 | 结构化输出 | 来源（answered） |
| --- | --- | --- | --- | --- |
| `ollama` | OpenAI SDK → Ollama `/v1` | local | strict `json_schema` | 响应 `model` |
| `openai` | OpenAI SDK，任意 OpenAI-compatible 端点 | api | strict `json_schema` | 响应 `model` |
| `anthropic` | Anthropic SDK Messages（system 消息用 `\n\n` 拼进 `system` 字段；`max_tokens` / `model_context_window_exceeded` → `Truncated`；`end_turn` / `stop_sequence` / `tool_use` 是完整回复；其他 stop_reason（`refusal`、`pause_turn`……）→ `EmptyReply`；结构化回复只读被强制的那个工具的 input） | api | 一个强制工具，`input_schema` = strict schema，`tool_choice` 指定它（与 lite-mem 相同） | 响应 `model` |
| `gateway` | OpenAI SDK → llm-gateway | subscription（可按 profile 覆盖） | strict `json_schema`；gateway 对 DeepSeek / Qwen 降级为 prompted，响应头 `X-LLMGW-Structured-Output: prompted` 记进来源 | 见 §6.2 |
| `litellm` | OpenAI SDK → LiteLLM Proxy | api | strict `json_schema`（S0 经 `ollama_chat/` 核实；其他上游接入时再核实） | 见 §6.3 |

**参数差异是数据**（沿用 lite-rag `PROVIDERS` 的做法），一张 `BackendProfile` 表：

| 字段 | 说明 | 已知值（S0 2026-11-18 核实） |
| --- | --- | --- |
| `max_tokens_param` | `max_tokens` 或 `max_completion_tokens` | ollama：`max_tokens`（Ollama 不认 `max_completion_tokens`，lite-rag 2026-11-08）；openai：`max_completion_tokens`；litellm：`max_completion_tokens`（1.108.9 两个都认，经 `ollama_chat/` 实测都生效；用 OpenAI 的新名字，由 LiteLLM 按上游转换）；gateway：`max_completion_tokens`（两个都映射到 `max_tokens`，新名字优先，llm-gateway `translate.py:58-59`） |
| `default_temperature` | 未配置时发送的值；None = 不发 | ollama：0.0；openai：None（推理模型拒绝非默认值）；litellm：None（上游可能是推理模型；`temperature: 0` 经 `ollama_chat/` 被接受） |
| `supports_temperature` | | anthropic 2.6.0：否（lite-rag 2026-11-08 核实） |
| `reasoning_effort` | 怎么发：顶层参数、`extra_body`、不支持 | litellm：顶层参数；`"none"` 经 `ollama_chat/` 关掉 qwen3 的思考（实测），`"low"` 对 qwen3 **不减少**思考，未设 = 开着思考。gateway：**现在被丢弃**（`schemas.py` 没有该字段，进 `X-LLMGW-Ignored-Params`），llm-gateway 修好前配置了就报错而不是悄悄失效 |
| `key_env` | 默认 key 变量 | |
| `answered_header` | 实际回答的模型在哪个响应头（响应体 `model` 不是它时） | litellm：`x-litellm-model-name`；其他：无（读响应体） |
| `via_headers` | 记进 `Provenance.via` 的来源头（允许列表） | litellm：`x-litellm-model-id`、`-model-group`；gateway：`X-LLMGW-Backend`、`X-LLMGW-Structured-Output` |
| `ignored_params_header` | 列出被丢弃字段的响应头 | gateway：`X-LLMGW-Ignored-Params` |
| `request_headers` | 每个请求带的头 | litellm：`x-litellm-num-retries: 0` |
| `role_tag` | 把 role 作为 `metadata.tags` 发出（§7.3） | litellm：是 |

思考 token 计入输出上限：经 LiteLLM 的 qwen3 开着思考时，`max_tokens` 小会得到空 `content` 和
`finish_reason: "length"`（应报 `Truncated`，不是 `EmptyReply`）。

gateway 后端在响应头里看到 `X-LLMGW-Ignored-Params` 列出 lite-model **实际发出**的字段（`max_*_tokens`、
`temperature`、`reasoning_effort`、`response_format`）时，抛 `IgnoredParameter`，不把被丢掉的设置当成生效；
列出的其他字段（不是 lite-model 发的）不算错。错误里只有 lite-model 自己的字段名，不回显 gateway 的列表。

## 6. 来源（provenance）

### 6.1 规则

- 每次回答都带 `Provenance`。`answered` 取**响应里实际回答的模型**，不取配置；端点没说时为 `None`，
  `tag` 里用 `requested?` 表示"未确认"，不冒充。
- 宿主存 `tag`（`backend:answered`），需要时另存 `requested` 和 `channel`。lite-mem 现在存的 `provider:model`
  格式在迁移时保持兼容：`tag` 的前缀是 **backend 的名字**，所以宿主把 backend 命名成它现在存的 provider
  （lite-mem 对本机 Ollama 存 `openai:qwen3:8b`：`backends: {openai: {type: ollama}}`），存下来的格式不变；
  也可以在 adapter 里映射。
- 一次操作固定一个 target（KB §1.4）：调用期间配置变了也不中途换。

### 6.2 经过 llm-gateway

gateway 的响应 `model` 字段写的是 **profile 名**（`translate.py:194,256`），不转发上游的响应头。所以：
- 固定 profile（一个 profile 对应一个模型）：`answered` = profile 名（响应体 `model`），`via` 记
  `x-llmgw-backend`（如 `claude-code`）和有时的 `x-llmgw-structured-output: prompted`，`routed=False`。
- 路由别名配在 gateway 上（`backends.<gateway>.router_aliases`）：在 llm-gateway 透传上游模型之前（§15），
  `build()` **拒绝**（gateway profile 没有 `answered_header`，拿不到实际模型就不允许路由）。

### 6.3 经过 LiteLLM

S0 实测（1.108.9，2026-11-18，`KB/global/litellm/README.md` §2–§3）：
- **普通别名**（`workhorse`）的响应体 `model` 永远是别名本身；实际模型只在响应头
  `x-litellm-model-name`（deployment 的 LiteLLM 模型串，如 `ollama_chat/qwen3:8b`）。
- **路由别名**打开 `return_raw_model_name: true` 后，响应体 `model` 是被选中 deployment 的 LiteLLM 模型串
  （`ollama_chat/qwen3:1.7b`）；不打开时被改回别名，但 `x-litellm-model-name` 头仍是实际模型。
- 每个成功响应都有 `x-litellm-model-name`、`x-litellm-model-id`（选中的 deployment）、`x-litellm-model-group`
  （调用的别名）；错误响应没有。

规则（S3 实现，`openai_compat._answered`）：
- `answered` = 响应头 `x-litellm-model-name`（profile 的 `answered_header`）。路由别名的响应体 `model` 必须与
  头一致（`return_raw_model_name` 打开时一致），不一致抛 `ProvenanceUnknown`。头缺失时，普通别名和路由别名都
  抛 `ProvenanceUnknown`（拿不到实际模型就不冒充）。头的值只是请求的别名本身时同样抛（区分不了别名和模型；
  因此 proxy 配置里别名不能与它的模型串同名，`tests/unit/test_deploy_litellm.py` 守着）。
- `via` 记 `x-litellm-model-id` 和 `x-litellm-model-group`（只记允许列表里的头，名字小写）。
- `-m litellm` 实测（2026-11-18）：`workhorse` → `ollama_chat/qwen3:8b`；`smart-router` 对 "hi" →
  `ollama_chat/qwen3:1.7b`，对一个长的多步编程问题 → `ollama_chat/qwen3:8b`；`routed=True`。
- proxy 配置仍要求路由别名打开 `return_raw_model_name`（`tests/unit/test_deploy_litellm.py` 守着），让日志和
  响应体也指向实际模型。
- 成本：`x-litellm-response-cost` 在 Ollama 模型的成功响应里**不存在**（没有价格），错误响应里是 `0`。有这个头
  且是成功响应时才写进 `Usage.cost_usd`；接入有价格的上游时再核实。

## 7. 路由

### 7.1 为什么只用 Proxy

- litellm 包 2026 年两次被投毒：3 月 1.82.7 / 1.82.8；10 月 21 日 1.107.0（PyPI 全部 7 个 wheel，以及
  Docker `latest`、`main-stable` 等 tag），载荷在 Python 启动时运行、窃取凭证。装进进程，宿主能碰到的数据库
  连接和所有 key 都暴露；放在容器里，它只拿到挂进去的那几个 key。
- llm-steward 直接 import lite-rag，库依赖会传过去；steward 2026-11-08 已因依赖过重去掉过 LiteLLM。
- 路由配置（model_list、分档、分类器）放在 proxy，所有模块共用一份。

### 7.2 Proxy 配置（`deploy/litellm/`，S0 起）

`deploy/litellm/config.yaml` 是实际配置；S0 的分档全是本机 Ollama（`small` = `qwen3:1.7b`，`workhorse` =
`qwen3:8b`），测试和核实不花钱。接入付费上游时按下面的形状加 deployment：

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

`context_compaction`（1.108.9 默认开）：请求接近选中 deployment 的输入上限时，路由器另调一次模型把历史压成
摘要再发出去。这既是一次看不见的模型调用，又改写了宿主构造的请求，所以关掉。它还要读 deployment 的
`max_input_tokens`，Ollama 模型不在 LiteLLM 价格表里，开着时**每次**路由调用都 500（S0 实测）。

运行：版本、digest、公钥、升级步骤在 `deploy/litellm/README.md`。设计上的要求：
- 镜像按 digest 锁定；**永远不用 1.107.0、1.82.7、1.82.8**，不用浮动 tag。LiteLLM 用 cosign **密钥**签名
  （不是 keyless），公钥锁定在仓库里。
- 验签在 compose 里：`verify` 服务先跑 `cosign verify`，`litellm` 依赖它成功退出，验签失败就不启动；本机不装
  cosign，CI 跑同一个 `verify`。
- 不接数据库（没有数据库时 `max_budget` 不生效）；只绑 127.0.0.1:**4400**（4000 已登记给 tenderted）；只挂
  需要的 key。
- **不开 Proxy 的 cache**（`litellm_settings.cache`，2026-11-19）：缓存是 lite-cache 的事，它作为 hook 接在
  lite-model 上（§10）。Proxy 命中时 lite-model 现在认不出来：回复不会标 `cached`，省下的 token 会被记成这次花的
  （§9）。0.3.0：读到非空的 `x-litellm-cache-key` 时标 `cached=True`（profile 的 `cached_header`），这样即使
  有人开了，用量也不会算错。依据 1.108.9 源码：`cache_key` 只在命中分支写进 hidden params（`caching/caching_handler.py`），
  空值的头不发（`proxy/common_request_processing.py`）；Proxy 上的实测待做（会启动 Proxy，要 owner 同意）。

**订阅额度不进自动路由**：分档里不放指向 llm-gateway 订阅 profile 的 deployment。要用订阅，在 role 上
显式配 `backend: gateway`。确实要按难度在订阅模型间路由时，另起一个别名（如 `sub-router`），只有显式选它
才生效，并且受 `forbid_channels` 约束。

### 7.3 客户端护栏

- `RolePolicy.routable`：宿主声明。lite-rag 的 `contextual`、`graph` 的模型 id 是索引身份的一部分
  （lite-rag `config.py:155,217`），必须 `routable=False`；judge 一类测量用途也必须 `False`。
- `backends.litellm.router_aliases` 列出哪些名字是路由别名；不可路由的 role 配了其中一个，`build()` 拒绝。
- 每次请求带上 role：`metadata: {tags: ["role:<role>"]}`（S0：`user` 和 `metadata.tags` 都被接受；没有数据库或
  日志 callback 时 proxy 不在任何地方记录它们，接日志时再核实能看到哪个）。用 `tags` 而不用 `user`，`user`
  留给宿主表示终端用户。
- 每次请求带 `x-litellm-num-retries: 0`。S0 实测：配置 `num_retries: 2` 时，对不可达的 deployment 不带头重试
  2 次（4.8 s），带头 0 次（0.06 s）。重试只在 lite-model 一处。

## 8. 错误与重试

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

- **不做 fallback**：失败就抛，不换模型、不换后端。
- 重试：只重试同一个 target，只用 SDK 的 `max_retries`（默认 1）。LiteLLM 的重试关掉（§7.3），避免
  SDK × proxy × 宿主任务队列叠成几十次。
- 错误信息只含 role、`backend:model` 和异常类型，**不含上游原文**（SDK 和代理可能回显 key、auth 头、带凭证
  的 URL；沿用 lite-mem `_failed`），原始异常留在 `__cause__`。

## 9. 用量与成本

- `Usage`：input、output、reasoning（`completion_tokens_details.reasoning_tokens`，有就记；经 LiteLLM →
  Ollama 时是 null，思考 token 已含在 `completion_tokens` 里，S0 实测）、cost_usd。
- 失败和截断的调用也报告用量：端点回答了但答案不能用时（`Truncated`、`EmptyReply`、`BadStructuredReply`、
  `IgnoredParameter`、`ProvenanceUnknown`），异常的 `usage` 是这次花掉的 token 和成本；`CallFailed`、
  `ConfigError` 的 `usage` 是 `None`（没有回答，SDK 拿不到用量）。
- 成本：优先用端点自己报的成本（profile 的 `cost_header`，litellm：`x-litellm-response-cost`，能读成非负有限
  数才用）；否则用配置里的 `prices`（键 `"backend:model"`，model 是**实际回答**的模型，与 `Provenance.tag`
  一致；USD / 百万 token）；都没有就是 `None`，不估算。订阅通道的成本永远是 `None`。`prices` 的键里 backend
  不存在时，`build()` 拒绝。
- hook 命中（`cached=True`）的回复带的是**当初那次**调用的 `usage`：这次省下的，不是这次花的。汇总花费的宿主
  先看 `cached`。

## 10. 并发与 hook

- `Bounded(model, concurrency)` / `AsyncBounded`：沿用 lite-rag `BoundedCalls`，包一个 `Model` / `AsyncModel`，
  所有经过它的调用（包括别的线程 / task 发起的）同时最多 `concurrency` 个；`map()` 并发跑独立调用、按输入
  顺序返回，第一个错误抛出。
- Hook（KB §10：注册、可撤销、可寻址）：

  ```python
  class Hook(Protocol):
      name: str
      def before(self, call: Call) -> TextReply | StructuredReply | None: ...   # a reply = short-circuit (cache hit)
      def after(self, call: Call, reply: TextReply | StructuredReply) -> None: ...

  model.hooks.register(hook) -> Registration   # Registration.remove() unregisters
  ```

  `Call` 含 role、解析后的 `Target`、messages、`request`（实际发出的 SDK 参数）、`params`（去掉 model、
  messages 和传输层的 `extra_headers`）、schema、`schema_name`、`schema_hash`（strict schema 规范化 JSON 的
  sha256）：正好是 lite-cache L1 的缓存键需要的字段（lite-cache 设计 §3.1）。
- 顺序：每次调用取一份 hook 快照；`before` 按注册顺序，**第一个**返回回复的 hook 结束这次调用，不发请求，
  返回的回复 `cached=True`（放在回复上，不在 `Provenance` 上：来源保持原来那次真正回答的模型；主人 2026-11-18）。
  `after` 只对刚从端点拿到的回复运行（命中、失败的调用都不运行，缓存不会重复写）。
- 没有 hook 时不构造 `Call`。hook 是同步的，同步和异步模型共用（lite-cache 的 store 是同步的）。
- 名字唯一（重复注册报 `ConfigError`），`model.hooks[name]` 取回，`Registration.remove()` 撤销（重复撤销无害）。
- 调用上下文：`with call_context(scope="acme/u42", cache=True): model.text(...)`，hook 从 `Call.context`（只读）
  读到；嵌套的块叠加并覆盖外层。是一个 contextvar：`Model` 协议不加参数，宿主 adapter、`FakeModel`、`Bounded`
  都不变；同步调用和块内创建的 asyncio task 都看得到。lite-model 自己从不读这些值（主人 2026-11-18）。
- `LiteModel.target(role)` / `AsyncLiteModel.target(role)`：调用前取得某个 role 解析后的 target（宿主把它记进
  索引身份，§12.1）。
- hook 抛错 = 调用失败，原样抛出（不吞掉，不包装）；`before` 抛错时不发请求。`after` 抛错时这次已经计费的回复
  和它的 `usage` 一起丢掉：`after` 不应抛错（lite-cache 写失败时照样返回回复；回复上没有 `store_error` 字段，失败记在日志和 `CacheHook.store_errors` 里，lite-cache 设计 §3.5）。
- hook 看到的 `Call.request` 是实际请求的只读深拷贝：hook 改不了发出去的请求，缓存键描述的就是发出去的那个。
- hook 返回的回复类型不对（`text()` 拿到 `StructuredReply`，`structured()` 拿到 `TextReply` 或别的 schema 的
  值）时抛 `TypeError`，写明是哪个 hook——在出错的地方失败，而不是调用方读值时。
- `build()` / `build_async()` 返回 `LiteModel` / `AsyncLiteModel`（满足 `Model` / `AsyncModel`，多一个
  `.hooks`）；`Model` 协议本身不含 hooks，`FakeModel` 和宿主的 adapter 不必实现。
- **0.3.0（2026-11-19 定，2026-11-20 实现）**：
  - `AsyncHook`（`async def before / after`，`lite_model.hooks`），`AsyncLiteModel` 会 await 它；同步 `Hook` 在异步
    模型上照旧可用。同步模型注册 `AsyncHook` 时 `TypeError`（它永远不会被 await）。原因：llm-steward 是 async 的，
    lite-cache 的同步 psycopg 查询放在同步 hook 里会阻塞 event loop。
  - `Bounded` / `AsyncBounded` 包住 `LiteModel` / `AsyncLiteModel` 时，槽位在执行器里只包住真正发出的请求
    （结构化地认 `Limitable`：有 `limited(slot)` 的模型，共享 routes 和 hooks；宿主 adapter 转发它即可）：hook 命中不再排队，hook 也可以经同一个 `Bounded` 调模型
    （concurrency 1 时以前会死锁；L2 verifier 就是这个形状）。包住别的 `Model`（`FakeModel`、宿主 adapter）时照旧
    整个调用占一个槽位。
  - `Bounded.map` 的工作线程用调用方 context 的一份拷贝（每项一份）运行每一项，`call_context()` 的值能到 hook
    （lite-rag S18 发现，lite-rag 在自己的 `BoundedLLM.map` 里先绕过；2026-11-19）。
  - `cached=True` 的意思是"这次没有再花钱"，来源有两个：hook 命中（不经过 `after`），或 Proxy 自己的缓存命中
    （`x-litellm-cache-key`，§7.2；会经过 `after`）。hook 在 `after` 里看到 `reply.cached` 为真时，不该把它的
    `usage` 算成这次花掉的（lite-cache 的 `after` 待跟进）。
  - 不加 `failed(call, error)` hook：lite-cache 不统计失败的调用（它的设计 §0），失败调用的用量在异常上（§9）。

## 11. 测试工具

- `lite_model.testing.FakeModel`：按 role 脚本化回复（文本或结构化对象，也可以是要抛的异常；列表按次消费，
  函数按调用计算），记录每次请求（role、messages、schema），实现 `Model`。`AsyncFakeModel` 同样的脚本、同一份
  `calls`，实现 `AsyncModel`（一个类不能让同名方法既同步又异步）。来源为 `backend="fake"`。
- 契约测试同步、异步各跑一遍（`tests/contract/conftest.py` 的 `connect` fixture）。
- 契约测试：**真的 SDK**，只把 transport 换成记录请求的假 endpoint（`build(..., http_client=...)`，
  `tests/upstream.py`），检查每个后端实际发出的请求体（参数名、strict schema、强制工具、header）。
- `tests/conftest.py` 在 `-m ollama` / `-m litellm` 以外拒绝一切 socket 连接：忘了注入 transport 的测试会失败，
  而不是悄悄调用真的服务。
- 标记：`-m ollama`（真实本地 Ollama，免费）；`-m litellm`（需要本机 proxy，**只用指向 Ollama 的别名**）；
  永远没有调用订阅或付费 API 的测试。测试配置设 `forbid_channels: [subscription, api]`。

## 12. 迁移

每个模块在**自己的 repo 里**作为一个 stage 迁移，验收都包含：迁移前后发出去的请求体一致（契约测试）、
eval 数字一致、来源格式兼容。

| 顺序 | 模块 | 要点 |
| --- | --- | --- |
| 1 | lite-mem | 功能最全（strict schema、Anthropic 强制工具、按 role 覆盖），最能检验接口。`produced_by` 继续存 `provider:model`。**已迁移**（2026-11-18，lite-mem S12，用 0.1.1 的 `keyless`；12 个请求体前后相同） |
| 2 | lite-rag | 三处 LLM 配置变成三个 role；`contextual`、`graph` 不可路由；`Completion.model_id` 改成实际回答的模型；`literag eval` 前后一致。**已迁移**（2026-11-18，lite-rag S15，用 0.2.0 的 `host_roles`；5 个请求体前后相同，索引身份不变） |
| 3 | llm-steward | 异步；顺带修它没整理过的 strict schema（raw `model_json_schema()`）。**已迁移**（2026-11-19，llm-steward P9；5 个请求体前后相同——它的 schema 本来就是 strict 形状，`strict_schema` 结果不变；锁定的 lite-rag、lite-mem 一起升到迁移后的版本） |
| — | lite-eval judge | 暂不迁移（§1 非目标） |

lite-model 以 GitHub release wheel 发布，宿主在 `[tool.uv.sources]` 里锁定版本（与 lite-eval 相同的做法）。

### 12.1 迁移后删什么、留什么

迁移的目的是消除重复：一个模块迁移完，它自己那份 LLM 客户端代码**删掉**。模块之间可以一个一个迁，
没迁的照常工作；但已迁的模块里不能同时留着旧客户端。

| 模块 | 删掉（改由 lite-model 提供） | 留在模块里 |
| --- | --- | --- |
| lite-mem | `llm.py` 的 `strict_schema`、`_failed`、`_parsed`、`_VendorLLM`、`OpenAICompatibleLLM`、`AnthropicLLM`、`RoutedLLM`；`config.py` 的 `RoleOverride`、`ModelTarget`、`LLMConfig`（改为嵌入 lite-model 的配置）；`testing.py` 的 FakeLLM；对应的单元测试 | role 名（`extract`、`resolve`、`consolidate`）及其 `RoleSpec`；把 payload 序列化成 user 消息；`prompts/*.j2`；`Provenance` → `produced_by` 的存储格式；embedding 客户端（v2 前） |
| lite-rag | `llm.py` 的 `PROVIDERS`、`ProviderProfile`、`_Client`、`OpenAICompatLLM`、`AnthropicLLM`、`BoundedCalls`、`build_llm`；`config.py` 的 `LLMConfig` 及其 provider 校验；`testing.py` 的 FakeLLM；对应的单元测试 | `answer`、`contextual`、`graph` 三个 role 的声明（后两个 `routable=False`）；graph 从文本里解析 JSON（业务逻辑）；索引身份里的 LLM id（改为取自 lite-model 的 target）；`prompts/*.j2`；embedding、rerank |
| llm-steward | `llm/client.py`；`llm/roles.py` 里解析 role 和调模型的部分；各处自己生成 schema 的代码（改用 lite-model 的 `strict_schema`）；`tests/fakes/llm.py` | role 名（`memory.extract`、`memory.resolve`、`kb.risk`、`kb.worthiness`、`kb.draft`）；mem0 的 LLM 配置（mem0 内部调模型，不经过 lite-model） |
| lite-eval | 不迁移（§1 非目标） | judge 客户端整份 |

每个模块保留一个**薄 adapter**（十几行）：把模块自己的 `LLM` Protocol 接到 `lite_model.Model` 上，模块内部
的调用点不用改，也满足"每个组件都是 Protocol"的规则。是否改为直接依赖 `lite_model.Model`、去掉 adapter，
在各模块迁移时决定。

### 12.2 每个模块的迁移步骤

1. 先写契约测试，固定**现在**实际发出去的请求体（参数名、schema、强制工具、header）。
2. 改为调用 lite-model，跑同一组契约测试：请求体一致。
3. 跑模块自己的 eval（lite-rag `literag eval`，lite-mem 的 eval）：数字一致；前后数字记进 KB。
4. 一致之后**删除**旧客户端、旧 fake 和只测旧客户端的单元测试；FakeLLM 换成 `lite_model.testing.FakeModel`。
5. 更新该模块的设计文档和 `AGENTS.md`：LLM 相关规则（Protocol、no fallback、来源记录、Tests never call a real
   model）改为指向 lite-model；依赖列表里去掉不再直接使用的 `openai` / `anthropic`（如果别处也不用）。
6. 按该模块的阶段收尾流程（/simplify → Codex review → 修复 → 重跑全部检查）结束。

## 13. 项目结构

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

## 14. 阶段与验收

每个阶段结束：ruff、ruff format、mypy、pytest 全绿；新测试都见过一次红；然后按 `AGENTS.md` Workflow 第 4 步
收尾（/simplify → Codex 只读 review → 修复 → 重跑全部检查）。

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| **S0** 环境与核实 | repo 骨架（uv、ruff、mypy、pytest、CI）；版本锁定（openai、anthropic、pydantic 与 lite-mem 相同）；起 LiteLLM Proxy（digest 锁定、cosign 验签、无数据库、127.0.0.1:4400（原定 4000，与 tenderted 冲突）、§7.2 配置，分档先全用 Ollama 模型以免花钱）；用 curl 核实：`max_tokens` / `max_completion_tokens`、`temperature`、`reasoning_effort` 到 `ollama_chat` 的转换、strict `json_schema`、`return_raw_model_name`、响应头（`x-litellm-model-id`、`-model-group`、`-response-cost`）、`x-litellm-num-retries`、role 透传字段 | 核实结果写进 KB（`KB/global/litellm/README.md`，并修正 `tech-selection/01b` 过时的两处：`enable_context_window_escalation` 默认 false、`classifier_fallback`）；§5 参数表和 §6.3、§7.3 按结果更新；端口登记到 `KB/global/machine/` |
| **S1** 核心 | types、config（含启动校验）、errors、`strict_schema`、openai-compatible 后端（ollama / openai），同步 `text` / `structured`，来源取自响应，`FakeModel`，`dump-config` | 单元 + 契约测试覆盖 §4 每条校验、§5 ollama/openai 参数、§8 每种错误；`-m ollama` 跑通 text 与 structured |
| **S2** Anthropic 与异步 | anthropic 后端（强制工具、system 字段、`stop_reason`）；`build_async` 与两个异步后端 | 契约测试检查 Anthropic 请求体；同步与异步共用一套契约测试 |
| **S3** gateway 与 litellm | 两个后端；channel 与 `forbid_channels`；`router_aliases` 与 `RolePolicy` 护栏；§6.2 / §6.3 来源；`IgnoredParameter`；`x-litellm-num-retries: 0` | 契约测试覆盖：路由别名在不可路由 role 上被拒、`return_raw_model_name` 未开时 `ProvenanceUnknown`、gateway 忽略参数时报错；`-m litellm` 用 Ollama 别名跑通，来源是实际模型 |
| **S4** 用量、并发、hook | `Usage`（含 reasoning、失败时的用量）、价格表与成本头、`Bounded`、hook 注册 / 撤销 / 短路 | 测试覆盖：失败调用也有用量、hook 命中保留原来源并 `cached=True`、撤销后不再生效 |
| **S5** 发布与收尾 | README、`deploy/litellm/`、v0.1.0 release wheel；整个库的最终收尾 review | 宿主可以从 wheel 安装；全量检查绿 |
| **S6+** 迁移 | 在 lite-mem、lite-rag、llm-steward 各自 repo 里按 §12 迁移 | 见 §12 |
| **0.3.0** lite-cache 的前置（2026-11-20 实现） | `AsyncHook`；`Bounded` 槽位只包住真正的请求；`Bounded.map` 带调用方 context；litellm 回复带 `x-litellm-cache-key` 时 `cached=True` | `tests/contract/test_bounded_hooks.py`：concurrency 1 时命中不等正在跑的调用（sync、async）；hook 经同一个 `Bounded` 调模型不死锁；`map` 里 hook 看得到 `call_context`；async hook 命中短路且不阻塞其它 task；同步模型拒绝 `AsyncHook`。`test_usage_and_hooks.py`：带缓存头的 litellm 回复 `cached=True`、来源不变，空值或别的后端不算 |

### 阶段记录

（每个阶段完成后在这里追加：日期、做了什么、review 结论、被拒的 finding 及理由。）

**S0（2026-11-18）环境与核实。**
- 骨架：`pyproject.toml`（Python 3.12，uv_build；openai 3.46.0、pydantic 2.14.3、pydantic-settings 2.16.2、
  typer 0.27.3、extra anthropic 2.6.0，与 lite-mem 相同）、ruff / mypy strict / pytest（marker `ollama`、
  `litellm` 默认不选）、CI（检查 + `compose run --rm verify`，它也解析并插值 compose 文件）。ruff format 不管 `*.md`
  （设计里的代码块是排过版的示意）。
- `deploy/litellm/`：compose（`verify` → `litellm`）、config（分档全 Ollama）、`cosign.pub`、README。
  `tests/unit/test_deploy_litellm.py` 守住这些配置和 `uv.lock` 里没有 `litellm`；每条断言都改坏对应配置看过
  一次红。
- curl 核实结果：§5 参数表、§6.3、§7.2、§7.3、§9 已按结果改写；细节在 `KB/global/litellm/README.md`。
  与原设计不同的三处：（1）普通别名的响应体 `model` 是别名，实际模型取 `x-litellm-model-name` 头（§6.3）；
  （2）`context_compaction` 默认开且对 Ollama 模型 500，配置里关掉（§7.2）；（3）端口 4000 → 4400（与 tenderted
  冲突）。
- KB：新建 `KB/global/litellm/README.md`；修正 `tech-selection/01b`（`enable_context_window_escalation` 默认
  false、`classifier_fallback`、供应链一条）；`machine/ports.md` 登记 4400。
- 收尾 review：
  - `/simplify`（4 个角度）：LiteLLM 镜像只写一处（YAML anchor `x-litellm-image`；cosign 接受 `tag@digest`，
    实测）；公钥改为比对 sha256 并检查挂载；loopback 检查改为"所有对外端口"；fallback 检查去掉特例；
    删去 CI 里多余的 `compose config`、`allowed-confusables`、`restart: "no"`；§7.2 运行细节改为指向 README。
    跳过：CI `concurrency`（规模太小）、缓存验签结果（每次启动验签正是要的保证）。
  - Codex（只读）：1 条 P2，S0 记录写着 CI 跑 `compose config` 而 workflow 已去掉，已改记录（不加回：
    `compose run` 本身解析整份文件）。供应链、测试、设计一致性、版本锁定四个方面无 finding。

**S1（2026-11-18）核心。**
- 模块：`errors`、`types`、`profiles`（五个 type 的参数表）、`config`（命名 backend、`target()` / `resolve()`
  与 §4 全部启动校验）、`schema`（`strict_schema`，与 lite-mem 相同的变换）、`backends/openai_compat`（同步
  `text` / `structured`）、`client`（`build()`）、`testing`（`FakeModel`）、`cli`（`dump-config`）。
- 主人的决定：命名 backend（§4）。我定的：`build(http_client=...)`（§3）；`dump-config` 不跑校验；client 还没写的
  backend type 在 `build()` 报 `ConfigError`；`max_output_tokens` 不配不发。
- 测试：单元 + 契约 57 个，`-m ollama` 3 个（text、structured、`Truncated`，qwen3:1.7b，真实 Ollama）。
  每个测试都见过红：对生产代码做了 45 处单点改动（关掉每条校验、改 profile 值、从配置取 `answered`、错误带上游
  原文、不重试、`FakeModel` 各分支、关掉网络守卫……），每处至少一个测试失败；为此修了一个测不到东西的测试
  （`dump-config` 不带 key 时没配 role），删了一个不可能失败的测试和一段死代码（`JSONDecodeError`）。
- 收尾 review：
  - `/simplify`（4 个角度）：role 覆盖改用 `model_copy` 合并（去掉 12 行 `_pick` 和永远不会触发的 None 检查）；
    profile 的 temperature 默认值在 `target()` 里解析（`dump-config` 显示实际发出的值）；"后续阶段的 backend"
    检查并进 `resolve()`，与其他问题一次列出；`Target.where` 统一错误定位（规则 8）；profile 表改用关键字参数；
    `_Model` 一个 dict；后端改用 `with_raw_response`，来源和用量在 `_read(target, response, headers)` 里读——S3
    读响应头只改这一处；测试共用双厂商配置，重试测试带 `retry-after-ms: 1`（套件 1.5 s → 0.5 s）。
    跳过：失败调用带用量（S4 验收项，留给 S4 连同测试）、按 `protocol` 分发和异步 `FakeModel`（S2）、
    `strict_schema` 缓存（返回可变 dict，收益是微秒级）。改完重跑 44 处变异，全部被抓住。
  - Codex（只读）3 条：
    - P2 空字符串被 `or` 读成默认值：成立（后果不是 Codex 说的绕过 key 检查——`""` 回落到 `OPENAI_API_KEY` 仍被
      检查——而是"无 key"的意图被悄悄改写）。回归测试先红，修为不接受空字符串。
    - P2 `temperature` 不限范围：成立，回归测试先红，修为 0–2。
    - P1 gateway 的 `reasoning_effort` 应由 `X-LLMGW-Ignored-Params` 报 `IgnoredParameter`：**拒绝（属于 S3）**。
      llm-gateway 修好前在配置时拒绝正是 §5、§15 的要求；gateway 后端 S1 不可用；响应头检查是 S3 验收项。
  - 待定（未改）：`type: openai` 的 endpoint 没法声明"不需要 key"（本地 vLLM 一类）。真有宿主需要时再加。

**S2（2026-11-18）Anthropic 与异步。**
- 结构：每个协议一个模块（`openai_compat`、`anthropic_messages`），只有不碰网络的 `request()` / `reply()`
  加 `connect` / `connect_async` / `endpoint`；共用的来源和错误判定在 `backends/__init__.py`（`provenance`、
  `finish`）；`client.py` 只有一个同步执行器和一个异步执行器，按 `profile.protocol` 查表。`build_async()`、
  `AsyncModel`、`AsyncFakeModel`。
- Anthropic 请求体与 lite-mem / lite-rag 现在的相同（§5）。主人的决定：`anthropic` 必须配 `max_output_tokens`。
  新的启动检查：`anthropic` 的 `base_url` 不能以 `/v1` 结尾（写测试时发现：同一个 URL 给两家 SDK 用会请求
  `/v1/v1/messages`）。SDK client 按 (type, endpoint, key 变量, 超时, 重试) 共享，不跨协议。
- SDK 实测（记进 `KB/global/llm-evals/providers/`）：anthropic 2.6.0 用 `authorization: Bearer` 发 key，没有
  `x-api-key`；anthropic 的异步 `raw.parse()` 是协程，openai 的不是（执行器里 `inspect.isawaitable`）；
  anyio 的异步 transport 在 TaskGroup 里连接，网络守卫的错误包在 `ExceptionGroup` 里（测试会展开它）。
- 测试：单元 + 契约 115 个（契约测试同步、异步各一遍），`-m ollama` 4 个（新增一个异步 structured）。
  Anthropic 只有契约测试（付费 API，AGENTS.md 规则 11）。变异：S2 新代码 21 处、S1 的 44 处（重构后重新
  指向）全部被抓住。
- 收尾 review（S2）：
  - `/simplify`（4 个角度）：构造请求移到 `try` 外（本地错误不再被包成 `CallFailed`）；两个后端共用
    `sdk_client()`（两家 SDK 的 `Timeout` 都是 `httpx2.Timeout`）；`finish()` 收 `answered`，自己建来源；
    anthropic 的两条检查按 `profile.protocol` 而非 `type`；测试的 `Caller` 去掉分支，`Connect` 是有类型的
    Protocol；`_causes` 换成 `traceback.format_exception`。跳过：`strict_schema` 缓存（微秒级，且会共享可变
    dict）、`_Routes` 基类（无害）、为 S4 hook 拆执行器（S4 再拆）。改后 65 处变异全部被抓住。
  - Codex（只读）3 条：
    - High：只有 `max_tokens` 算截断，`refusal` / `pause_turn` 的部分文字被当成功回复，
      `model_context_window_exceeded` 不算截断。成立，而且 OpenAI 侧 `content_filter` 同样的缺口。回归测试先红
      （14 个），修为：截断集合加 `model_context_window_exceeded`；完整集合以外的 stop_reason 和 OpenAI 的
      `content_filter` 报 `EmptyReply("no usable reply (stop reason …)")`（只带枚举值，不带上游文字）。没有加新的
      异常类（公共 API 不变）。
    - Medium：结构化回复取第一个 `tool_use`，不看名字。成立，只读被强制的那个工具；回归测试先红。
    - Medium：SDK client 应按 key 的值共享而不是变量名。**拒绝**：按变量名是有意的（共享键里不放密钥值）；收益
      只是"两个变量存了同一个 key"时少建一个 client。记录里的"key"改为"key 变量"。

**S3（2026-11-18）gateway 与 litellm。**
- 两个 type 用同一个 OpenAI-compatible 模块，差异全是 `BackendProfile` 数据（§5 表）：`answered_header`、
  `via_headers`、`ignored_params_header`、`request_headers`、`role_tag`。主人的决定：`Provenance` 加 `via`。
- 来源：§6.2、§6.3（litellm 读头、拿不到抛 `ProvenanceUnknown`；gateway 读 profile 名，路由别名在 build 拒绝）。
  `IgnoredParameter` 只对 lite-model 实际发出的字段。每个 litellm 请求带 `x-litellm-num-retries: 0` 和
  `metadata.tags: ["role:<role>"]`。去掉了 S1 的"后续阶段才有"关卡（对应的测试换成"能 build"和"gateway
  上的路由别名被拒"两条——那个关卡本来就是临时的）。
- 测试：单元 + 契约 156 个；`-m ollama` 4 个；`-m litellm` 3 个（本机 proxy，Ollama 别名，`channel: local`）。
  一个 live 测试的期望改过：路由器把"新西兰最大城市"分给 qwen3:1.7b，它答 Wellington——测的是路径不是模型
  知识，改为算术题。变异：S3 15 处、S1 43 处、S2 21 处全部被抓住。
- 收尾 review：
  - `/simplify`（4 个角度）：三条响应头规则（读 answered、记 `via`、`IgnoredParameter`）从 `openai_compat` 移进
    与协议无关的 `finish()`，和启动检查用同一份参数表数据（否则一个配了 `answered_header` 的 anthropic 协议
    profile 会放行路由、回复时却记别名）；`reply()` 收到实际发出的请求，`IgnoredParameter` 对照它（不再重算
    一遍"发了什么"，也不再重复算 `strict_schema`；S4 的 hook 也要它）；参数表里的头名全小写；live 测试读
    `.env` 用 `python-dotenv`。跳过：测试配置工厂挪进 `tests/configs.py`（只有一个文件用）。
  - Codex（只读）：
    - P1 普通别名时，`x-litellm-model-name` 的值就是别名本身也被当作实际模型：成立，回归测试先红，改为抛
      `ProvenanceUnknown`；代价是 proxy 配置里别名不能与它的模型串同名，加了部署测试守着（§6.3）。
    - P2 查头依赖 SDK 的大小写不敏感映射：成立（实际运行不受影响），`finish()` 先把头名转小写，单元测试用
      普通 dict 先红。
    - 文档：§8 的 `ProvenanceUnknown` 只写了路由调用，改为包括任何 litellm 调用。
  - Ollama 在收尾期间停了（不是 lite-model 停的）；主人同意后用 `brew services run ollama` 重新启动，重跑：
    `-m ollama` 4 个、`-m litellm` 3 个通过；5 条 live 变异（Ollama 停着时只是连接失败，不算数）全部被对应的
    断言抓住。

**S4（2026-11-18）用量、成本、并发、hook。**
- 用量：答案被拒的异常都带 `usage`（含成本），`CallFailed` 为 `None`。成本：端点报的成本头 > `prices`（按实际
  回答的模型）> `None`；订阅为 `None`。`Price`、`ModelsConfig.prices`、`BackendProfile.cost_header`。
- `Bounded` / `AsyncBounded`；`hooks.py`（`Call`、`Hook`、`Hooks`、`Registration`）；`LiteModel` /
  `AsyncLiteModel`；回复加 `cached`（主人的决定：放回复上）。
- 测试：单元 + 契约 202 个，`-m ollama` 4 个、`-m litellm` 3 个。写测试时发现一个测不到东西的测试：
  `Bounded` 的并发上限只经 `map()` 测，而 `map()` 的线程池本身就只有 `concurrency` 个线程，去掉信号量也过；加了
  8 个外部线程共用一个 `Bounded` 的测试。变异：S4 20 处，S1 43、S2 21、S3 18 处（重新指向后）全部被抓住。
- 收尾 review：
  - `/simplify`（4 个角度）：hook 返回的回复类型不对时抛 `TypeError`（`cast` 本会把错藏到很后面）；
    `schema_hash` 改用与 lite-eval `canonical.py` 相同的规则（NFC、`ensure_ascii=False`、`allow_nan=False`），
    照抄不 import（规则 9），测试对照写出的 lite-eval 算法；`Hooks` 去掉没有设计依据的 `__iter__` / `__len__`
    （测试改为断言 `hooks[name]` 的 `KeyError`），`Registration` 走 `Hooks._remove`；执行器里 `_before` 的
    三元组拆成 `call = ... if hooks else None` 加 `_hit()`；去掉 `_awaited`（`map` 的函数是协程）；`_number`
    简化。又发现一个测不到东西的测试：`concurrency=1` 时 `map()` 是普通循环，"一次一个"永远成立——改为外部
    8 线程，参数化 1 和 2 个槽位。跳过：hook 快照、`Call.params` 缓存、共享线程池（都是微秒级）。记进 §15：
    `Call` 缺 lite-cache 的 scope / policy；命中也占 `Bounded` 槽位。§9 写明命中的 `usage` 是省下的。
  - Codex（只读）5 条：
    - High：`Call.request` 就是随后发给 SDK 的那个 dict，hook 改它就能让缓存键和实际请求不一致。成立，回归测试
      先红；hook 拿到只读深拷贝（只在有 hook 时拷贝）。
    - High：hook 回复类型不对时 `TypeError` 用了 `hit!r`，会带上回复内容（规则 8）。成立，先红；只写类型名。
    - Medium：`Price` 接受 `inf`、乘积可能溢出成 `inf`。成立，先红；`allow_inf_nan=False`，非有限的结果记 `None`。
    - Medium：`after()` 抛错或回复解析抛非 `ModelError` 时丢了 `usage`。**拒绝**：hook 异常按设计原样抛出，给任意
      异常挂属性不干净；`after` 不应抛错，写进 §10。回复解析的形状错误由 SDK 的类型保证。
    - Medium：`Call` 缺 scope / policy。**已在 §15**，lite-cache 接入时定。
  - 改后变异 112 处（S4 30、S1 43、S2 21、S3 18）全部被抓住；`-m ollama` 4、`-m litellm` 3 通过。

**S5（2026-11-18）发布与整库收尾。**
- 打包：版本 `0.1.0`；`license-files = ["LICENSE"]`（打包测试先红：wheel 和 sdist 都缺 LICENSE 原文）；
  `tests/unit/test_packaging.py` 检查 LICENSE、`License-Expression`、`py.typed`、`litemodel` 入口、`testing.py`
  在、`tests/` 不在。CI：action 按 commit 锁定（与 lite-eval 同一组）；`package` 任务构建、tag 版本与 wheel
  一致、干净 venv 里装 wheel 和 `wheel[anthropic]` 跑 `dump-config` 与 `scripts/smoke.py`；`release` 任务只在
  `v*` tag 上、只有它有 `contents: write`，发布 `package` 任务构建并测过的那一份（主人选择 CI 按 tag 发布）。
- README 写全（安装、快速上手、配置、后端表、来源、路由、错误、用量、并发、hook、测试宿主）；
  `tests/unit/test_readme.py` 让 README 的配置示例和快速上手跟着代码走（变异时发现它没测默认 backend，补上）。
- 整库收尾（范围是 `f13239a` 以来的全部代码，不只 S5）：
  - `/simplify`（4 个角度）定了发布前的公共 API（主人的两个决定，见决定记录）：`call_context()` 传每次调用的
    值给 hook（`Model` 协议不加参数）；`ModelsConfig.backend` / `model` 必填。另外：`LiteModel.target(role)`；
    顶层导出收窄（hook 类型从 `lite_model.hooks` 导入，`Reply`、`call_context` 公开）；`Reply` 只定义一处
    （`testing` 里的改名 `FakeReply`）；两份"传输字段"集合合成一份；`Call.role` 取自 target；anthropic 的两条
    检查改为参数表字段（`extra`、`base_url_is_root`）；删去代码里引用 review 轮次的注释；§6.1 写明 lite-mem 的
    `provider:model` 格式怎么保持（backend 起名 `openai`、`type: ollama`，有测试）。CLI 缺 backend / model 时报清楚
    的错而不是 traceback。跳过：导入耗时、`strict_schema` 缓存、CI 里构建两次（都是毫秒级或并行）。
  - Codex（只读，整库）2 条：
    - Medium：未知的 anthropic `stop_reason` 原样进了 `EmptyReply` 的消息（规则 8）。成立，先红；只写已知的
      `refusal` / `pause_turn`，其余写 `unexpected`。
    - Medium：打包测试在子进程里 `uv build`，网络守卫看不到子进程。成立，改为 `uv build --offline`（构建后端
      在 `uv sync` 后已在缓存）。这一条没有先红的测试：本机缓存里有 `uv_build`，去掉 `--offline` 也会通过；
      `--offline` 保证的是缓存缺失时失败而不是联网。
    - 设计首行状态过期：已改。供应链、来源、密钥、公共 API、发布流程、宿主迁移路径其余均无 finding。
- 检查：单元 + 契约 226 个，`-m ollama` 4 个，`-m litellm` 3 个；变异 136 处（S1 43、S2 21、S3 18、S4 30、
  S5 13+2，含 live 5 处，Ollama 在线）全部被抓住。
- 发布：第一次推 main（`131c850`）时 CI 的 `package` 任务失败——它的 `dump-config` 冒烟步骤是在
  backend / model 改为必填之前写的、本机也是那时跑的；补上两个变量（`767473d`），本机从 `env -i` 的干净环境重跑
  CI 的打包步骤后再推，CI 全绿（教训记进 KB）。然后推 tag `v0.1.0`：tag 上的 CI 四个任务全过，`release` 发布了
  `lite_model-0.1.0-py3-none-any.whl` 和 `lite_model-0.1.0.tar.gz`
  （https://github.com/ArthurGao/lite-model/releases/tag/v0.1.0）。
- 验收"宿主可以从 wheel 安装"：一个新项目用 `[tool.uv.sources]` 指向 release 的 wheel URL，装
  `lite-model[anthropic]`（lite-model 0.1.0、openai 3.46.0、anthropic 2.6.0），对本机 Ollama：`text` 回 `pong`，
  来源 `ollama:qwen3:1.7b`，有用量；`build_async` + `call_context` 的 `structured` 回 `result=42`。

**0.1.1（2026-11-18）`keyless` backend。** lite-mem 迁移（它的 S12）需要：它现在用 OpenAI 协议连 Ollama，不发 temperature、
不要 key；`type: ollama` 会多发 `temperature: 0.0`，`type: openai` 要 `OPENAI_API_KEY`。主人选择加 `BackendConfig.keyless`
（而不是接受请求体变化）。测试先红（配置解析、占位 key 和不发 temperature 的请求、与 `api_key_env` 冲突），变异 2 处都被抓住。
生产代码约 12 行、一个文件：按全局规则这是小改动，不做 `/simplify` 和 Codex 收尾。

**0.2.1（2026-11-19）key 的来源是一个选择。** lite-mem 改用 0.2.0 `host_backends` 时发现：宿主的 `openai` 是 keyless，配置只写
`api_key_env` 时两者叠在一起，启动报 "keyless and api_key_env cannot both be set"。现在叠加时，配置写了 `api_key_env` 或
`keyless` 就去掉宿主的这两个；同一个条目里两个都写仍然报错。测试先红；生产代码约 6 行、一个文件，按全局规则不做收尾。

## 15. 待确认

- **llm-gateway 的两处修改**（在 llm-gateway repo 里做，不在本 repo）：
  1. `reasoning_effort` 透传（现在被丢弃；lite-rag graph 和 lite-mem eval 设的 `"none"` 经过 gateway 时不生效，
     经过 gateway 跑出的结果其实开着思考）。
  2. 透传上游实际模型（建议响应头 `X-LLMGW-Upstream-Model`，响应体 `model` 不变）。
  在这之前，lite-model 对 gateway 后端：配置了 `reasoning_effort` 就拒绝（§5），gateway + 路由别名拒绝（§6.2）。
- embedding 是否在 v2 收进来（lite-rag、lite-mem 各有一份 httpx2 embedding 客户端）。embedding 和索引绑定，
  永远不路由。
- lite-eval 的 judge 以后是否迁移。
- ~~hook 只有 `before` / `after`，失败的调用不经过 hook~~：2026-11-19 定，不加 `failed()`。lite-cache 改为 hook
  接入，失败的调用不计数，`skip:<reason>` 只来自它自己的 `accept` 谓词（lite-cache 设计 §0、§3.4）。
- ~~`Call` 缺 lite-cache 的 scope / policy~~：S5 定为 `call_context()`（§10）。lite-cache 的 hook 必须在没有
  scope 时不缓存（否则会把一个租户的答案给另一个租户）。
- ~~`Bounded` 包在模型外面：hook 命中也要等一个槽位~~：0.3.0 把槽位移进执行器（§10，2026-11-20）。
- ~~**异步宿主的 hook**~~：0.3.0 加了 `AsyncHook`（§10，2026-11-20）。
- **LiteLLM Proxy 的 cache 命中**（2026-11-19）：Proxy 的 cache 不开（§7.2）；0.3.0 读 `x-litellm-cache-key` 标
  `cached=True`。2026-11-20 源码核实：1.108.9 只在命中时带这个头（§7.2）。仍待 Proxy 实测：命中时
  `x-litellm-model-name` 是否仍是当初回答的模型。
- ~~自动路由的效果~~：**已测**（2026-11-18，lite-rag S14，`KB/global/llm-evals/evals/2026-11-18-lite-rag-auto-routing.md`）。
  lite-rag 32 道金标题 × 2 次，"对" = 关键事实且 grounded：固定 qwen3:8b 0.938（p50 12 s），经 Proxy 的同一模型 0.938，
  qwen3:1.7b 0.750（3.8 s），`smart-router`（启发式，small = 1.7b，其余 = 8b）0.844（5.2 s）。启发式分类器**按语言而不是难度**
  分档：中文题全部去 1.7b。结论：现在的 §7.2 配置不值得在中英混合的 RAG 上打开；要路由，先在实际流量上测 `llm` 分类器或
  `keyword_tier_rules`。Proxy 本身不改变答案质量。
- 各模块迁移时是否去掉 adapter、直接依赖 `lite_model.Model`（§12.1）。
- ~~**宿主改不了内置 backend**~~（lite-mem S12 发现，2026-11-18）：0.2.0 已做（§4 宿主自己的默认值）。lite-mem 2026-11-19 已改用
  `host_backends`、删掉自己的 validator（`be5269f`）。
- ~~**keyless 与 `api_key_env` 叠加时冲突**~~（lite-mem 改用 0.2.0 时发现，2026-11-19）：0.2.1 已做（§4）。
- **结构化输出约束解码**（lite-rag S16，2026-11-19）：qwen3:8b 在 Ollama 上自由输出 JSON 时，同一个 chunk 10 次里 7 次漏引号；
  经 `structured()`（strict JSON schema）10 次全部合法。宿主要 JSON 时应当用 `structured()`，不要自己解析文本。

### 决定记录

| 日期 | 决定 | 理由 |
| --- | --- | --- |
| 2026-11-18 | 新建共享库 lite-model，lite 模块依赖它，消除 4 份重复的 LLM 客户端 | 引入路由支持时不再每个模块各写一遍 |
| 2026-11-18 | LiteLLM 只用 Proxy 模式，作为可选后端；库不 import `litellm` | 2026 年两次供应链攻击；依赖会传给 llm-steward；路由配置共用 |
| 2026-11-18 | 没有 LiteLLM 时走原来的直连方式；配了 LiteLLM 而它不可用时报错，不退回 | lite 模块要能单独使用；no silent fallback |
| 2026-11-18 | llm-gateway 不依赖 lite-model；调用链是 app → lite-model → llm-gateway | gateway 是服务端，lite-model 是客户端 |
| 2026-11-18 | 名字 `lite-model`（不叫 `lite-router` / `lite-llm`） | 主要是客户端而非路由；`lite-llm` 与 `litellm` 易混 |
| 2026-11-18 | GitHub `ArthurGao/lite-model`，公开，Apache-2.0 | 与其他 lite 模块一致 |
| 2026-11-18 | 模块迁移后删除自己的 LLM 客户端，只留 role 声明、prompt、业务解析和一个薄 adapter（§12.1） | 共享库的目的就是消除重复；新旧两份并存会再次分叉 |
| 2026-11-18 | LiteLLM Proxy 端口 4400 | 4000–4010、4080–4090 已登记给 tenderted |
| 2026-11-18 | cosign 以容器运行，作为 compose 的 `verify` 服务，`litellm` 依赖它成功退出 | 主人的选择：本机不装 cosign，一条 `compose up` 先验签再启动 |
| 2026-11-18 | LiteLLM 的 `answered` 取响应头 `x-litellm-model-name`，响应体 `model` 只做一致性检查（§6.3） | S0 实测：普通别名的响应体 `model` 是别名，头才是实际模型 |
| 2026-11-18 | 命名 backend：`backends.<name>: {type, base_url, api_key_env, channel, router_aliases}`，role 用名字引用（§4） | 主人的选择：一个宿主要能同时用两个 OpenAI-compatible 厂商；endpoint 和 key 只写一处 |
| 2026-11-18 | `anthropic` 必须配 `max_output_tokens`，启动时检查，不猜默认值（§4） | 主人的选择：宿主迁移时写自己的旧值（lite-mem 2048、lite-rag 1024），请求体不变 |
| 2026-11-18 | `Provenance.via`：端点自己的来源头（参数表里的允许列表，名字小写）；`tag` 不变 | 主人的选择：§6.2、§6.3 要求另记 deployment id、别名、gateway backend |
| 2026-11-18 | hook 命中的回复 `cached=True` 放在回复上，`Provenance` 不变（§10） | 主人的选择：是否缓存属于这次调用，来源属于当初回答的模型 |
| 2026-11-18 | `build()` 返回 `LiteModel`（`Model` + `.hooks`）；`Model` 协议不含 hooks（§3、§10） | `FakeModel` 和宿主 adapter 不必实现 hook；`model.hooks.register` 照设计可用 |
| 2026-11-18 | 调用上下文用 contextvar：`call_context(**values)` → `Call.context`；`Model` 协议不加参数（§10） | 主人的选择：三个宿主马上要照 `Model` 写 adapter，加参数以后就是破坏性改动 |
| 2026-11-18 | `ModelsConfig.backend` / `model` 必填，没有默认值（§4） | 主人的选择：宿主嵌入时不会悄悄继承默认值，lite-model 的版本不会改宿主的模型 |
| 2026-11-18 | 公开 API 收窄：`Call`、`Hook`、`Hooks`、`Registration` 从 `lite_model.hooks` 导入；`Reply`、`call_context`、`LiteModel.target()` 公开 | 发布后导出的名字都是承诺；宿主迁移要在调用前拿到 target |
| 2026-11-18 | 不在中英混合的 RAG 上打开启发式自动路由（§15 实测：0.844 对 0.938，按语言分档） | lite-rag S14 两次真实运行；要路由先测分类器 |
| 2026-11-18 | `build(..., http_client=...)` 是公开参数（§3） | 宿主迁移的契约测试要用真 SDK 记录请求体（§12.2）；lite-model 自己的契约测试也这样做 |
| 2026-11-18 | 0.2.0：宿主在 `ModelsConfig` 子类上设 `host_backends` / `host_roles`，配置逐字段叠在上面；只有另一个 `type` 才替换 backend | lite-mem S12 的默认 role 曾被内置 `openai` 顶到 api.openai.com；lite-rag 的 graph 默认值会被 `__ROLES` 整个顶掉。审查：同样的 `type` 照写一遍不能丢宿主的值；role 的 null 仍是"用顶层的值"（和 0.1 一样） |
| 2026-11-19 | 缓存选 lite-cache 作为 hook、存 PostgreSQL；LiteLLM Proxy 的 cache 不开，不用 Redis（§7.2，lite-cache 设计 §0） | 默认调用不经过 Proxy；Proxy 命中时 lite-model 标不出 `cached`，用量算错；没有必填 scope；Redis 对 1–20 s 的模型调用没有速度上的意义，又是新的基础设施 |
| 2026-11-19 | 0.3.0 计划：`AsyncHook`、`Bounded` 槽位移进执行器、读 `x-litellm-cache-key`；不加 `failed()` hook（§10、§15） | lite-cache 接入 async 的 llm-steward 和 lite-rag 之前需要；lite-cache 不统计失败的调用 |
| 2026-11-20 | 0.3.0：`Bounded` 包 `LiteModel` 时槽位只包请求（`limited(slot)`），包别的 `Model` 时照旧整个调用；同步模型拒绝 `AsyncHook`；`x-litellm-cache-key` 按 1.108.9 源码判命中（§7.2、§10） | 只改 `LiteModel` 的执行器，`FakeModel` 和宿主 adapter 不必实现槽位；L2 verifier 要在 hook 里经同一个 `Bounded` 调模型，整个调用占槽位会死锁 |

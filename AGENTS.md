# lite-model — Agent Guide

lite-model is the **shared model-call layer for the lite modules** (lite-rag, lite-mem, llm-steward, later
lite-cache): call sites name a role, configuration maps each role to a model on a backend (`ollama`, `openai` —
any OpenAI-compatible endpoint —, `anthropic`, `gateway` — llm-gateway —, `litellm` — LiteLLM Proxy, `voyage` — embeddings and rerank only, `cohere` / `gemini` — embeddings only), and every
reply says which model actually answered. Routing by request difficulty is done by an optional LiteLLM Proxy;
lite-model treats it as one more backend and adds the client-side guards and provenance routing needs.

This file is the shared guide for every coding agent. `CLAUDE.md` imports it.

Naming: repo `lite-model`, Python package `lite_model`, CLI `litemodel`, env prefix `LITEMODEL_`.

## Sources of truth (highest first)

1. `docs/lite-model-design.md` — the design. §2 position and call chain, §3 interface, §4 config and startup
   validation, §5 backends and parameter table, §6 provenance, §7 routing (LiteLLM Proxy, guards), §8 errors,
   §9 usage, §10 hooks, §11 test tools, §12 migration, §14 stages and acceptance, §15 open questions and decisions.
2. Project KB: `/Users/arthurgao/Docs/Knowledgebase/projects/lite-model/` — decisions, findings, lessons.
   Library facts shared across projects (LiteLLM, OpenAI / Anthropic SDKs, Ollama) go to `KB/global/`
   (LiteLLM: `KB/global/litellm/`, created in S0). Write with the `kb-maintain` skill.

If a change must contradict the design, update the design in the same change and say why in the commit.

## Current stage

**v0.8.0 (2026-11-22): `cohere` and `gemini` embed-only backend types over a small `rest` client** (design §3.4; additive; `-m cohere` / `-m gemini` opt-in and paid).

**v0.7.0 (2026-11-22): the `voyage` backend type (embeddings with `input_type` / `output_dimension`, `rerank()` with `RolePolicy(kind="rerank")`)** (design §3.3; additive; `-m voyage` is opt-in and paid).

**v0.6.0 (2026-11-22): embed role settings (`dimensions`, `batch_size`, `max_concurrency`), 400/413 batch halving, `close()` / `aclose()`** (design §3.2.1; additive).

**v0.5.1 (2026-11-22): `FakeModel.with_vectors(...)` replaces 0.5.0's `FakeModel(vectors=...)`**, which broke mypy in hosts that build a fake from a dict (`FakeModel(**scripts)`; design §14 0.5.1 record). Use 0.5.1, not 0.5.0.

**v0.5.0 (2026-11-21): `embed()` and Anthropic structured output via `output_config.format`** (design §3.2, §15; `tests/contract/test_embed.py`, `test_anthropic.py`). `embed()` is additive (an embed role: `RolePolicy(kind="embed")`, never routable; ollama / openai / litellm). The Anthropic change alters the structured request body for hosts on the `anthropic` backend: newer Claude models reject the forced `tool_choice` it replaced.

**v0.4.0 (2026-11-21): `turn()`, one model-chosen tool turn** (design §3.1; `tests/contract/test_turns.py`). Additive: `text()` / `structured()`, `Model` and hooks are unchanged. lite-agent depends on it; the other hosts stay on 0.3.0 until they need it.

**S5 done 2026-11-18; host migrations (design §12) done 2026-11-19. v0.3.0 (2026-11-20)** (design §14): `AsyncHook`, `Bounded` slot around the request only, `Bounded.map` carries the caller's context, `x-litellm-cache-key` → `cached` (design §10, §7.2).

## Rules

1. **lite-model never imports `litellm`.** LiteLLM is reached only over HTTP as the `litellm` backend
   (design §7.1: two 2026 supply-chain compromises; the dependency would spread to every host). Never use
   LiteLLM 1.107.0, 1.82.7 or 1.82.8, never a floating image tag; pin the image by digest and verify it with
   cosign.
2. **LiteLLM and llm-gateway are optional.** With neither configured, lite-model talks to Ollama / OpenAI /
   Anthropic directly, exactly as the hosts do today. A configured backend that is down **raises**.
3. **No fallback chain.** A failed call raises after retries on the same target only. No switching models or
   backends, no LiteLLM `fallbacks`, LiteLLM retries forced to 0 per request (design §7.3, §8).
4. **Every reply carries provenance**: role, requested alias, the model that **actually answered** (read from
   the response, never from config), backend, channel. When the answering model cannot be read on a routed call,
   raise `ProvenanceUnknown`; never record a guess (design §6).
5. **Routing is opt-in per role.** Hosts declare which roles are routable; a routing alias on a non-routable role
   is refused at `build()`. Anything that is part of an index's identity, and every judge, is non-routable.
6. **Validate at startup, not at the first call** (design §4): unknown role, missing key, missing extra,
   unsupported parameter, routing alias on a pinned role, forbidden channel.
7. **Backend differences are data** (`profiles.py`), not branches in the clients. A parameter an endpoint
   reports as ignored is an error (`IgnoredParameter`), never a silent no-op.
8. **Error messages never carry upstream text** (keys, auth headers, credentialed URLs can be echoed): role,
   `backend:model` and the exception type only; the cause stays chained.
9. **No host dependency.** lite-model depends on no lite module and no service code. llm-gateway never imports
   lite-model. lite-cache plugs in through hooks (design §10).
10. **Secrets only from environment variables.** Never commit `.env*` except `.env.example`.
11. **Tests never call a paid API or a subscription.** Unit and contract tests use `FakeModel` or fake SDK
    clients. Real models run only under opt-in markers: `-m ollama` (local Ollama), `-m litellm` (local proxy,
    Ollama-backed aliases only), and `-m voyage` / `-m cohere` / `-m gemini` (paid vendor APIs: only when the owner
    asks, with the key read inline from `~/.config/lite-ai/<vendor>.key`). Test config sets `forbid_channels: [subscription, api]`. Never call llm-gateway
    subscription profiles from tests or on your own initiative.
12. **Pinned dependencies**: exact versions for the SDKs (`openai`, `anthropic`, `pydantic`), the same as lite-mem
    unless a stage records why not. An upgrade re-runs the contract tests and `-m ollama`.
13. **Measure, don't guess**: a host migration (design §12) proves request bodies and eval numbers unchanged.

## Tech stack

Python 3.12, uv, pydantic / pydantic-settings, OpenAI SDK (OpenAI-compatible endpoints: Ollama `/v1`,
llm-gateway, LiteLLM Proxy, vendors), Anthropic SDK (extra `anthropic`), Typer (CLI), pytest, ruff, mypy.
Ollama on the host (`127.0.0.1:11434`). LiteLLM Proxy in Docker (`127.0.0.1:4400`, design §7.2). llm-gateway on
the host (`127.0.0.1:8787`, owner-only).

## Commands

```
uv run pytest                              # unit + contract, fakes only
uv run pytest -m ollama                    # opt-in: real local Ollama
uv run pytest -m litellm                   # opt-in: local LiteLLM Proxy, Ollama-backed aliases only
uv run ruff check && uv run ruff format --check && uv run mypy
uv run litemodel dump-config               # the merged config; makes no calls
docker compose -f deploy/litellm/compose.yaml up -d   # LiteLLM Proxy (from S0)
```

Free checks before an opt-in run: `curl -s http://127.0.0.1:11434/api/tags` (Ollama),
`curl -s http://127.0.0.1:4400/health/liveliness` (LiteLLM), `curl -s http://127.0.0.1:8787/healthz` (gateway).

## Workflow

1. Work stage by stage (design §14). Each stage ends with every check green; every new test seen red once.
2. Definition of done: ruff, ruff format, mypy, tests pass; acceptance items covered by tests; design updated if
   behaviour changed.
3. Conventional Commits (`feat(backends): …`). All code-facing text and docs (design doc, README, plans, comments) in English; Chinese only as examples or test data.
4. **Stage close-out — after every stage's implementation passes its checks, before its commit and before the
   next stage starts:**
   1. Run `/simplify` on the stage's full diff (reuse, simplification, efficiency, altitude) and apply the fixes.
   2. Independent read-only Codex review of the same diff (command in `CLAUDE.md`).
   3. Fix every finding that holds up — a bug gets a failing regression test first. A finding you reject is
      written down with the reason (design §14 stage record or the KB); never silently dropped.
   4. Re-run all checks so they cover the final code, then report the review outcome with the stage.
   S5 is the final close-out over the whole library before any host migrates.

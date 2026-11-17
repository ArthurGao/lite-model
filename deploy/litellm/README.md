# LiteLLM Proxy for lite-model

Optional. lite-model reaches it over HTTP as the `litellm` backend and never imports the `litellm` package
(design §7.1). It does difficulty routing (`smart-router`) behind one OpenAI-compatible endpoint.

## Run

```bash
cp deploy/litellm/.env.example deploy/litellm/.env       # then set LITELLM_MASTER_KEY
lsof -ti:4400 || echo free                                 # the port must be free
docker compose -f deploy/litellm/compose.yaml up -d        # verify runs first; litellm starts only if it passed
curl -s http://127.0.0.1:4400/health/liveliness
docker compose -f deploy/litellm/compose.yaml down
```

- Listens on `127.0.0.1:4400` only (4000 is tenderted's `api-gateway`; `KB/global/machine/ports.md`).
- No database: only the OpenAI-compatible API and the master key. Budgets (`max_budget`) are not enforced
  without one.
- Every tier is a local Ollama model (`qwen3:1.7b`, `qwen3:8b`), reached at `host.docker.internal:11434`;
  Ollama must be running on the host. No vendor key is mounted.
- Never `source` `.env` in the shell that runs `docker compose`: an exported shell variable beats `.env`.

## The pinned images

| Image | Tag | Digest (index) | Chosen |
| --- | --- | --- | --- |
| `ghcr.io/berriai/litellm` | `v1.108.9` | `sha256:d817e8e4…3d6608` | 2026-11-18: the 1.108 line, released 2026-11-14; the two later patches (1.108.10, .11) were a day old and claim no security fix |
| `ghcr.io/sigstore/cosign/cosign` | `v3.1.4` | `sha256:0f061548…f69c2d` | 2026-11-18: latest cosign release (2026-10-20) |

Never 1.107.0, 1.82.7 or 1.82.8 (compromised, design §7.1); never a floating tag (`latest`, `main-stable`).
`tests/unit/test_deploy_litellm.py` checks digest pinning, the verify-before-run order, the pinned key file,
loopback-only ports, no retries, no fallbacks and the router settings. It also rejects the banned tags, but a tag
is only a label: the signature on the pinned digest is what proves the image.

### Signature

LiteLLM signs its GHCR images with a cosign **key** (not keyless). `cosign.pub` here is the key at
BerriAI/litellm commit `e82a3c646908ba0a6ee0beddc4310d7a6cac4065` (the key rotated on 2026-10-23 after the
2026-10-21 incident; LiteLLM's "Docker Image Security" page). The `v1.108.9` and `v1.108.11` tags carry the same
key. The image is named once, in `x-litellm-image` (a YAML anchor), and the `verify` service runs:

```
cosign verify --key=/cosign.pub ghcr.io/berriai/litellm:<tag>@sha256:<digest>
```

and CI runs the same service (`docker compose run --rm verify`).

### Upgrading

1. Pick a release at least a few days old; read its notes. Never a banned version.
2. `docker buildx imagetools inspect ghcr.io/berriai/litellm:<tag> --format '{{json .Manifest}}' | jq -r .digest`
3. Check the key did not change: `curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/<tag>/cosign.pub | diff - deploy/litellm/cosign.pub`.
   A different key needs LiteLLM's announcement before it is trusted; then replace `cosign.pub` and the
   hash in `tests/unit/test_deploy_litellm.py`.
4. Put `<tag>@<digest>` in `x-litellm-image` in `compose.yaml`, update the table above, run `uv run pytest` and
   `docker compose -f deploy/litellm/compose.yaml run --rm verify`, then re-run `-m litellm`.

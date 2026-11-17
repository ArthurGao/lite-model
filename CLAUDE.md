@AGENTS.md

# Claude Code notes

- At the start of a session: check §14 of `docs/lite-model-design.md` for the current stage and its acceptance,
  and skim the project KB at `/Users/arthurgao/Docs/Knowledgebase/projects/lite-model/`.
- Never answer a LiteLLM / OpenAI SDK / Anthropic SDK / Ollama / llm-gateway API question from memory: check
  context7 or the pinned version's docs (LiteLLM moves fast and has renamed config keys), and record surprises in
  the KB (`KB/global/litellm/` for LiteLLM facts). llm-gateway's behaviour is in its repo
  (`/Users/arthurgao/Code/AI/llm-gateway`, `docs/DESIGN.md`) and `KB/global/llm-gateway/README.md`.
- Relevant user-level skills: `llm-app-architecture` (role layer, no fallback chain, provenance, pluggable
  capabilities §10, gateways), `llm-model-eval` (comparing models), `testing-and-ci`, `secrets-hygiene` (keys for
  the proxy), `dev-machine` (ports, Ollama, credentials), `kb-maintain`.
- Ollama runs on the host; check it for free with `curl -s http://127.0.0.1:11434/api/tags` before `-m ollama`.
  Check port 4400 with `lsof -ti:4400` before starting the proxy (`deploy/litellm/README.md`). Do not start long-lived services unasked.
- **Stage close-out, for every stage, in this order** (`AGENTS.md` Workflow step 4):
  1. `/simplify` on the stage's full diff; apply its fixes.
  2. Codex, from the repo root, read-only, on the stage's whole diff:
     `codex exec -s read-only -o <out.md> "Review \`git diff <commit before the stage>\` (includes uncommitted
     changes) … focus … report severity, file:line, scenario, fix" < /dev/null`.
     The `< /dev/null` matters: with a piped stdin, `codex exec` waits for stdin to close and hangs in a
     background shell (codex-cli 0.160.1, 2026-11-08).
  3. Fix every finding that holds up (a bug gets a failing regression test first); write down the ones you
     reject, with the reason.
  4. Re-run every check (read summary lines, e.g. `rg 'passed|failed'`, never a bare `| tail -1`), then report
     and propose the commit. Do not start the next stage before this is done. After each push, read the CI
     result (`gh run list --limit 3 --json headSha,conclusion`).

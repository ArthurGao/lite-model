# lite-model

The shared model-call layer for the lite modules (lite-rag, lite-mem, llm-steward): call sites name a role,
configuration maps each role to a model on a backend, and every reply says which model actually answered.

Work in progress: the design is [`docs/lite-model-design.md`](docs/lite-model-design.md), the current stage is in
its §14. The optional LiteLLM Proxy is in [`deploy/litellm/`](deploy/litellm/README.md).

License: Apache-2.0.

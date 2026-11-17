"""Guards on deploy/litellm/ and the lock file (AGENTS.md rules 1 and 3, design §7.2, §8)."""

import hashlib
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy" / "litellm"
IMAGE_REF = re.compile(r"(?P<repo>[^:@]+)(?::(?P<tag>[^@]+))?@sha256:[0-9a-f]{64}")
# Compromised releases (design §7.1) and floating tags. Only a label check: the cosign
# signature on the pinned digest is what proves the image.
BANNED_TAGS = {"v1.107.0", "v1.82.7", "v1.82.8", "latest", "main-stable", "main-latest"}
# sha256 of cosign.pub at BerriAI/litellm commit e82a3c6 (key rotated 2026-10-23); README.md.
COSIGN_PUB_SHA256 = "e3fe3e08c44b16407fe0b37c43b4f8b603b3a0f4ee74c29e647bc022d762c2c0"


def _load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((DEPLOY / name).read_text())
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return _load("compose.yaml")


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return _load("config.yaml")


def test_every_image_is_pinned_by_digest(compose: dict[str, Any]) -> None:
    images = {name: svc["image"] for name, svc in compose["services"].items()}
    assert set(images) == {"verify", "litellm"}
    for name, ref in images.items():
        match = IMAGE_REF.fullmatch(ref)
        assert match, f"{name}: {ref} is not pinned by digest"
        assert match["tag"] not in BANNED_TAGS, f"{name}: banned tag {match['tag']}"


def test_litellm_starts_only_after_its_signature_verifies(compose: dict[str, Any]) -> None:
    verify, litellm = compose["services"]["verify"], compose["services"]["litellm"]
    assert litellm["depends_on"] == {"verify": {"condition": "service_completed_successfully"}}
    assert verify["command"] == ["verify", "--key=/cosign.pub", litellm["image"]]
    assert verify["volumes"] == ["./cosign.pub:/cosign.pub:ro"]


def test_signing_key_is_the_pinned_one() -> None:
    digest = hashlib.sha256((DEPLOY / "cosign.pub").read_bytes()).hexdigest()
    assert digest == COSIGN_PUB_SHA256


def test_published_ports_bind_loopback_only(compose: dict[str, Any]) -> None:
    ports = [p for svc in compose["services"].values() for p in svc.get("ports", [])]
    assert ports, "the proxy publishes no port"
    assert all(p.startswith("127.0.0.1:") for p in ports), ports


def _keys(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def test_proxy_does_not_retry_or_fall_back(config: dict[str, Any]) -> None:
    assert config["litellm_settings"]["num_retries"] == 0
    assert not [k for k in _keys(config) if "fallback" in k]


def test_router_aliases_return_the_chosen_model(config: dict[str, Any]) -> None:
    models = config["model_list"]
    names = {m["model_name"] for m in models}
    [router] = [m for m in models if m["litellm_params"]["model"].startswith("auto_router/")]
    assert router["model_name"] == "smart-router"
    params = router["litellm_params"]
    cfg = params["complexity_router_config"]
    assert cfg["return_raw_model_name"] is True
    assert cfg["classifier_type"] == "heuristic"
    # Default-on compaction rewrites the history with an extra model call (S0, 1.108.9).
    assert cfg["context_compaction"] is False
    assert set(cfg["tiers"]) == {"SIMPLE", "MEDIUM", "COMPLEX", "REASONING"}
    assert set(cfg["tiers"].values()) | {params["complexity_router_default_model"]} <= names


def test_litellm_is_never_a_python_dependency() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = {p["name"] for p in lock["package"]}
    assert "lite-model" in packages
    assert "litellm" not in packages

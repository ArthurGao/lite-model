"""What the published wheel and sdist must carry (design §14 S5): hosts install the release
wheel by URL (design §12), so a missing file shows up only in a host."""

import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dist(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("dist")
    # --offline: the build backend comes from uv's cache (`uv sync` put it there); the unit suite
    # never reaches the network, the socket guard in conftest.py cannot see a subprocess.
    subprocess.run(
        ["uv", "build", "--offline", "--quiet", "--out-dir", str(out)],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
    return next(out.glob("lite_model-*.whl")), next(out.glob("lite_model-*.tar.gz"))


def test_the_wheel_carries_the_license_types_cli_and_fakes(dist: tuple[Path, Path]) -> None:
    wheel, _ = dist
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    info = f"lite_model-{version}.dist-info"
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        assert f"{info}/licenses/LICENSE" in names
        assert "Apache License" in zf.read(f"{info}/licenses/LICENSE").decode()
        assert "License-Expression: Apache-2.0" in zf.read(f"{info}/METADATA").decode()
        assert "litemodel = lite_model.cli:app" in zf.read(f"{info}/entry_points.txt").decode()
        assert "lite_model/py.typed" in names  # hosts type-check against it (mypy strict)
        assert "lite_model/testing.py" in names  # hosts' tests use FakeModel
        assert not [n for n in names if n.startswith("tests/")]


def test_the_sdist_carries_the_license(dist: tuple[Path, Path]) -> None:
    _, sdist = dist
    with tarfile.open(sdist) as tf:
        assert f"{sdist.name.removesuffix('.tar.gz')}/LICENSE" in tf.getnames()

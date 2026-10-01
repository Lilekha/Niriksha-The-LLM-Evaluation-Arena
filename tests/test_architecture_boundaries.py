import ast
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "src" / "niriksha" / "core"

FORBIDDEN = (
    "niriksha.providers",
    "niriksha.scorers",
    "httpx",
    "httpcore",
    "requests",
    "urllib3",
    "aiohttp",
    "urllib.request",
    "http.client",
    "socket",
)


def imported_modules(source: str, package: str) -> set[str]:
    """All absolute module names a source file imports; ``package`` resolves relative imports."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join(anchor + ([base] if base else []))
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def violations(modules: set[str]) -> set[str]:
    return {m for m in modules for bad in FORBIDDEN if m == bad or m.startswith(bad + ".")}


@pytest.mark.parametrize(
    "source",
    [
        "import niriksha.providers.fake",
        "from niriksha.providers import fake",
        "from niriksha import providers",
        "from .. import providers",
        "from ..scorers import exact",
        "import httpx",
        "from urllib import request",
        "import socket",
    ],
)
def test_checker_detects_forbidden_imports(source):
    assert violations(imported_modules(source, "niriksha.core"))


def test_checker_allows_core_and_stdlib_imports():
    source = "import math\nfrom niriksha.core import generation\nfrom . import provider"
    assert not violations(imported_modules(source, "niriksha.core"))


def test_core_has_source_files():
    assert list(CORE.rglob("*.py"))


def test_core_imports_no_providers_scorers_or_network_libraries():
    for path in CORE.rglob("*.py"):
        package = ".".join(path.relative_to(CORE.parents[1]).with_suffix("").parts[:-1])
        found = violations(imported_modules(path.read_text(encoding="utf-8"), package))
        assert not found, f"{path.name} imports forbidden modules: {sorted(found)}"


def test_fake_provider_imports_only_core_and_stdlib():
    path = CORE.parent / "providers" / "fake.py"
    modules = imported_modules(path.read_text(encoding="utf-8"), "niriksha.providers")
    external = {m.split(".")[0] for m in modules if m} - {"niriksha"}
    assert external <= {"collections"}, f"unexpected imports: {sorted(external)}"
    assert not violations(modules - {"niriksha.providers"}), "fake provider uses network or scorers"


def test_boundary_scan_covers_the_dataset_and_run_modules():
    scanned = {path.stem for path in CORE.rglob("*.py")}
    expected = {
        "generation",
        "provider",
        "runner",
        "dataset",
        "provenance",
        "runstore",
        "execution",
    }
    assert expected <= scanned

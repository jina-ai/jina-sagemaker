"""Lightweight notebook-import validation.

The customer-facing notebooks under ``notebooks/`` are the primary
onboarding surface for this package — if a public API rename leaves them
referencing a no-longer-exported symbol the customer's copy-paste blows
up on first run with a confusing error.

This module parses every notebook, extracts every Python ``import``
statement that targets ``jina_sagemaker`` (or a sub-module), and verifies
the import resolves under the currently-installed package. It does NOT
execute notebook cells (most of them call real SageMaker, which needs
AWS credentials and a deployed endpoint — well out of scope for unit
tests). The check is intentionally narrow: it catches the rename / move
/ removed-export class of bug and nothing else.

Cells whose source does not parse as Python (typical for cells holding
raw AWS CLI snippets or shell commands without an ``!`` prefix) are
skipped with a warning rather than failing the test, so a pre-existing
broken cell does not block this check from running on the rest of the
notebook.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
TARGET_PACKAGE = "jina_sagemaker"


def _notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _extract_target_imports(source: str) -> list[str]:
    """Return the dotted module names this cell imports from ``jina_sagemaker``.

    Returns an empty list both when no relevant imports exist AND when the
    cell does not parse — non-Python cells are not this test's problem.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == TARGET_PACKAGE:
                    modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == TARGET_PACKAGE:
                modules.append(node.module)
                # The `from jina_sagemaker import X` form also requires X
                # to be a real attribute of the module — verify per name
                # so a renamed export is caught even when the parent
                # module itself still imports cleanly.
                for alias in node.names:
                    modules.append(f"{node.module}:{alias.name}")
    return modules


@pytest.mark.parametrize(
    "nb_path",
    _notebook_paths(),
    ids=lambda p: p.name,
)
def test_notebook_jina_sagemaker_imports_resolve(nb_path: Path) -> None:
    nb = nbformat.read(str(nb_path), as_version=4)
    seen: set[str] = set()

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for target in _extract_target_imports(cell.source):
            if target in seen:
                continue
            seen.add(target)

            if ":" in target:
                module_name, attr = target.split(":", 1)
                module = importlib.import_module(module_name)
                assert hasattr(module, attr), (
                    f"{nb_path.name} imports {attr!r} from {module_name}, "
                    f"but {module_name} has no such attribute. The notebook "
                    f"likely references a renamed or removed export."
                )
            else:
                importlib.import_module(target)


def test_notebook_directory_is_non_empty() -> None:
    """Guard against silently losing every notebook (e.g. accidental rm)."""
    assert _notebook_paths(), "no notebooks found under notebooks/"

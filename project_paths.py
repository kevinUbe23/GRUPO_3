from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
GENERATION_DIR = ARTIFACTS_ROOT / "01_generacion"
EDA_DIR = ARTIFACTS_ROOT / "02_eda"
PREPARATION_DIR = ARTIFACTS_ROOT / "03_preparacion"


def ensure_artifact_dirs() -> None:
    for path in (ARTIFACTS_ROOT, GENERATION_DIR, EDA_DIR, PREPARATION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def iter_repo_roots(start: Path | None = None):
    current = (start or Path.cwd()).resolve()
    yield current
    yield from current.parents


def find_repo_root(start: Path | None = None) -> Path:
    for candidate in iter_repo_roots(start):
        if (candidate / "project_paths.py").exists():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto.")


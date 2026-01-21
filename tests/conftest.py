import sys
from pathlib import Path


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_root = repo_root / "python"
    sys.path.insert(0, str(python_root))

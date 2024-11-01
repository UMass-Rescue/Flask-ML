from pathlib import Path
from .MLServer import MLServer

__all__ = ["MLServer"]  # for flake8 unused import error

def load_file_as_string(file_path: str) -> str:
    fp = Path(file_path)
    if not fp.is_file():
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, "r") as f:
        return f.read()

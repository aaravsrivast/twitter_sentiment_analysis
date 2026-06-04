"""Load project configuration from config.yaml."""

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str, config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    return ROOT_DIR / cfg["paths"][key]

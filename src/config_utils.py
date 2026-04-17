from __future__ import annotations

from pathlib import Path
import yaml


def project_root() -> Path:
    # Returns the root directory of the project by going two levels up
    # from the current file location.
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    # Constructs the full path to the config.yaml file
    # located at the project root.
    return project_root() / 'config.yaml'


def load_config() -> dict:
    # Opens the config.yaml file and safely parses it into a dictionary.
    # yaml.safe_load prevents execution of arbitrary YAML code.
    with open(config_path(), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_path(value: str | None) -> Path | None:
    # Handles optional paths:
    # - If None → return None directly
    # - If absolute path → return as is
    # - If relative path → resolve it relative to project root
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root() / p
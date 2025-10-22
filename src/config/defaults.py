from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``updates`` into ``base``."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Load a YAML configuration file and optionally apply overrides."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if overrides:
        data = _deep_update(data, overrides)
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Persist a configuration mapping to disk."""
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)

"""Load YAML configs whose leaves are ``{value, unit, source}`` records.

Keeping the unit and source next to every number makes the "honest physics"
rule checkable: a test walks the file and fails on any unlabelled parameter.
Code only ever sees the plain values, through :func:`load_config`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CONFIG = REPO_ROOT / "configs" / "env_default.yaml"


def _is_param(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node


def _strip(node: Any) -> Any:
    if _is_param(node):
        return copy.deepcopy(node["value"])
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items()}
    return copy.deepcopy(node)


def load_raw(path: str | Path = DEFAULT_ENV_CONFIG) -> dict:
    """Return the config with unit/source metadata intact."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(
    path: str | Path = DEFAULT_ENV_CONFIG, overrides: dict[str, Any] | None = None
) -> dict:
    """Return a nested dict of plain values.

    ``overrides`` uses dotted keys, e.g. ``{"mismatch.scale": 0.0}``. Unknown
    keys raise, so a typo cannot silently leave a default in place.
    """
    cfg = _strip(load_raw(path))
    for dotted, value in (overrides or {}).items():
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            if key not in node:
                raise KeyError(f"Unknown config section '{key}' in override '{dotted}'")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown config key '{dotted}'")
        node[leaf] = value
    return cfg


def iter_params(raw: dict, prefix: str = ""):
    """Yield ``(dotted_name, record)`` for every parameter record in a raw config."""
    for key, node in raw.items():
        name = f"{prefix}{key}"
        if _is_param(node):
            yield name, node
        elif isinstance(node, dict):
            yield from iter_params(node, prefix=f"{name}.")

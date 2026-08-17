"""极简 YAML 配置加载, 支持 _base_ 字段一层继承."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_update(base_cfg, cfg)
    return cfg

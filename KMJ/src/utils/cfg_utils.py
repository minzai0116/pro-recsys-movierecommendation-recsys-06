from __future__ import annotations

from typing import Any, Optional


def cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    """
    Safe config selector for mixed config types:
    - OmegaConf DictConfig/ListConfig
    - plain dict (nested)
    - simple objects with attributes

    Args:
        cfg: config object
        key: dot-path, e.g. "train.run_dir"
        default: fallback value
    """
    # OmegaConf path (preferred)
    try:
        from omegaconf import OmegaConf  # type: ignore

        val = OmegaConf.select(cfg, key, default=default)
        return val
    except Exception:
        pass

    # dict path fallback
    cur = cfg
    for part in key.split("."):
        if cur is None:
            return default
        try:
            # dict-like
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                # attribute-like
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    # mapping-like
                    try:
                        cur = cur.get(part, default)  # type: ignore[attr-defined]
                    except Exception:
                        return default
        except Exception:
            return default
    return cur


def cfg_bool(cfg: Any, key: str, default: bool = False) -> bool:
    v = cfg_select(cfg, key, default=default)
    try:
        return bool(v)
    except Exception:
        return default


def cfg_str(cfg: Any, key: str, default: str = "") -> str:
    v = cfg_select(cfg, key, default=default)
    if v is None:
        return default
    return str(v)



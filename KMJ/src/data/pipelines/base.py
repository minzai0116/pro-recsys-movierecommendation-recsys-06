# src/data/pipelines/base.py (핵심 부분만)
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.data.data_bundle import DataBundle
from src.data.transforms.registry import GLOBAL_TRANSFORM_REGISTRY, POST_TRANSFORM_REGISTRY


def _as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


class DataPipelineBase:
    name: str = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    # --- required hooks implemented by concrete pipelines ---
    def load_raw(self, cfg: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def prepare_data(
        self, cfg: Any, raw: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
        raise NotImplementedError

    def to_bundle(
        self,
        cfg: Any,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame],
        test_df: pd.DataFrame,
        meta: Dict[str, Any],
    ) -> DataBundle:
        raise NotImplementedError

    # ------------------------------
    # builder helpers (NEW)
    # ------------------------------
    def _get_transform_cfg(self, cfg: Any) -> Dict[str, Any]:
        """
        supports:
          cfg.data.transforms.global: list[{name, args}]
          cfg.data.transforms.post: list[{name, args}]
        """
        d = getattr(cfg, "data", None)
        if d is None:
            return {"global": [], "post": []}

        t = getattr(d, "transforms", None)
        if t is None:
            # allow legacy keys if you want:
            return {"global": [], "post": []}

        # OmegaConf DictConfig도 dict처럼 접근 가능하게 최대한 방어
        def _get(obj, key, default):
            try:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                if hasattr(obj, "get"):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            except Exception:
                return default

        return {
            "global": _get(t, "global", []) or [],
            "post": _get(t, "post", []) or [],
        }

    def _build_global_transforms(self, cfg: Any):
        items = self._get_transform_cfg(cfg)["global"]
        items = _as_list(items)

        out = []
        for spec in items:
            # spec can be "name" or {"name":..., "args":...}
            if isinstance(spec, str):
                name, args = spec, {}
            else:
                name = str(spec.get("name"))
                args = dict(spec.get("args", {}) or {})

            if name not in GLOBAL_TRANSFORM_REGISTRY:
                raise ValueError(f"Unknown global transform: {name}. Available={list(GLOBAL_TRANSFORM_REGISTRY.keys())}")

            cls = GLOBAL_TRANSFORM_REGISTRY[name]
            out.append(cls(**args))
        return out

    def _build_post_transforms(self, cfg: Any):
        items = self._get_transform_cfg(cfg)["post"]
        items = _as_list(items)

        out = []
        for spec in items:
            if isinstance(spec, str):
                name, args = spec, {}
            else:
                name = str(spec.get("name"))
                args = dict(spec.get("args", {}) or {})

            if name not in POST_TRANSFORM_REGISTRY:
                raise ValueError(f"Unknown post transform: {name}. Available={list(POST_TRANSFORM_REGISTRY.keys())}")

            cls = POST_TRANSFORM_REGISTRY[name]
            out.append(cls(**args))
        return out

    # ------------------------------
    # fixed build (FINAL)
    # ------------------------------
    def build(self, cfg: Any) -> DataBundle:
        """
        fixed protocol:
          1) load_raw
          2) global transforms (raw dict)
          3) prepare_data -> train/valid/test/meta
          4) post transforms: fit on train, transform on train/valid/test
          5) to_bundle
        """
        raw = self.load_raw(cfg)

        # (2) global transforms (stateless-ish)
        for t in self._build_global_transforms(cfg):
            raw = t(cfg, raw)  # GlobalTransform.__call__(cfg, raw)->raw

        train_df, valid_df, test_df, meta = self.prepare_data(cfg, raw)

        # (4) post transforms (stateful)
        for t in self._build_post_transforms(cfg):
            state = t.fit(cfg, raw, train_df)
            train_df = t.transform(cfg, raw, train_df, state)
            if valid_df is not None:
                valid_df = t.transform(cfg, raw, valid_df, state)
            test_df = t.transform(cfg, raw, test_df, state)

        return self.to_bundle(cfg, train_df, valid_df, test_df, meta)

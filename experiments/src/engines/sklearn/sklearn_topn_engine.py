# src/engines/sklearn/sklearn_topn_engine.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.data.data_bundle import DataBundle
from src.engines.sklearn.sklearn_base import SklearnEngineBase

class SklearnTopNEngine(SklearnEngineBase):
    """
    Task-family engine for Top-N / Seq-Top-N using sklearn-style (non-deep) models.

    Currently supports:
      - cfg.model == "EASE_SciPy" : closed-form EASE using scipy/numpy (ported from the experiments workspace)

    Output contract:
      - for schema.task in {"topn","seq_topn"}: preds is List[List[int]] with len == len(meta['submission']['users'])
    """

    def fit(self, bundle: DataBundle) -> Dict[str, Any]:
        self._guard_topn(bundle)

        model_name = self._model_name()
        if model_name != "EASE_SciPy":
            raise ValueError(f"SklearnTopNEngine supports only model=EASE_SciPy for now, got {model_name}")

        # read model args (SSoT: cfg.model_args[model])
        mcfg = self._model_args(model_name)
        lambda_reg = float(mcfg.get("lambda_reg", 100.0))
        # NewFrame-compatible options
        val_items_per_user = int(mcfg.get("val_items_per_user", 10))
        compute_val = bool(mcfg.get("compute_val_recall", True))
        retrain_on_full = bool(mcfg.get("retrain_on_full", True))

        user_col = str(bundle.schema["user_col"])
        item_col = str(bundle.schema["item_col"])
        time_col = str(bundle.schema.get("time_col") or "time")

        from src.models.sklearn.topn.ease_scipy import EASEScipy, EASECheckpoint
        from src.models.sklearn.topn.split_utils import user_sequence_split, create_ground_truth, recall_at_k

        # (NewFrame-like) Train/Validation split + Recall@K sanity metric
        base_df = bundle.train.copy()
        train_df = base_df
        val_df = None
        val_recall = None
        if compute_val and (time_col in base_df.columns):
            train_df, val_df = user_sequence_split(
                base_df,
                user_col=user_col,
                item_col=item_col,
                time_col=time_col,
                val_items_per_user=val_items_per_user,
                seed=int(getattr(self.cfg, "seed", 42)),
            )

        # fit on train split
        X, user_id_to_idx, item_id_to_idx, idx_to_item_id, popular_items = EASEScipy.build_implicit_matrix(
            train_df, user_col=user_col, item_col=item_col
        )
        model = EASEScipy(lambda_reg=lambda_reg)
        B = model.fit(X, verbose=bool(getattr(self.cfg, "verbose", True)))

        # validation recall (NewFrame metric definition)
        if compute_val and (val_df is not None) and (len(val_df) > 0):
            gt = create_ground_truth(val_df, user_col=user_col, item_col=item_col)
            val_users = [u for u in gt.keys() if u in user_id_to_idx]
            if val_users:
                k_eval = int(getattr(getattr(self.cfg, "train", None), "topk", 10))
                user_indices = [user_id_to_idx[int(u)] for u in val_users]
                recs = model.recommend_batch(
                    X,
                    user_indices=user_indices,
                    idx_to_item_id=idx_to_item_id,
                    k=k_eval,
                    popular_items=popular_items,
                )
                preds_map = {int(u): r for u, r in zip(val_users, recs)}
                val_recall = recall_at_k(preds_map, gt, k=k_eval)

        ckpt = EASECheckpoint(
            lambda_reg=lambda_reg,
            B=B,
            user_id_to_idx=user_id_to_idx,
            item_id_to_idx=item_id_to_idx,
            idx_to_item_id=idx_to_item_id,
            popular_items=popular_items,
        )

        # (NewFrame-like) Retrain on full data for final model
        if retrain_on_full:
            X_full, user_id_to_idx_full, item_id_to_idx_full, idx_to_item_id_full, popular_items_full = (
                EASEScipy.build_implicit_matrix(base_df, user_col=user_col, item_col=item_col)
            )
            model_full = EASEScipy(lambda_reg=lambda_reg)
            B_full = model_full.fit(X_full, verbose=False)
            ckpt = EASECheckpoint(
                lambda_reg=lambda_reg,
                B=B_full,
                user_id_to_idx=user_id_to_idx_full,
                item_id_to_idx=item_id_to_idx_full,
                idx_to_item_id=idx_to_item_id_full,
                popular_items=popular_items_full,
            )

        ckpt_path = self._default_ckpt_path(f"{model_name}.joblib")
        self._save_checkpoint(
            {
                "model": model_name,
                "ckpt": ckpt,
                "schema": dict(bundle.schema),
                "meta": {
                    "n_users": int(X.shape[0]),
                    "n_items": int(X.shape[1]),
                    "lambda_reg": float(lambda_reg),
                    "val_items_per_user": int(val_items_per_user),
                    "val_recall_at_topk": float(val_recall) if val_recall is not None else None,
                    "retrain_on_full": bool(retrain_on_full),
                },
            },
            ckpt_path,
        )

        self._log_train(
            {
                "engine_family": "sklearn_topn",
                "model": model_name,
                "checkpoint_saved": ckpt_path,
                "n_train": int(len(bundle.train)),
                "n_users": int(X.shape[0]),
                "n_items": int(X.shape[1]),
                "lambda_reg": float(lambda_reg),
                "val_recall": float(val_recall) if val_recall is not None else None,
                "val_items_per_user": int(val_items_per_user),
                "retrain_on_full": bool(retrain_on_full),
            }
        )

        return {"checkpoint_path": ckpt_path}

    def predict(self, bundle: DataBundle, checkpoint: Optional[str] = None):
        self._guard_topn(bundle)

        model_name = self._model_name()
        if model_name != "EASE_SciPy":
            raise ValueError(f"SklearnTopNEngine supports only model=EASE_SciPy for now, got {model_name}")

        # submission users order
        sub = (bundle.meta or {}).get("submission", {}) or {}
        users: List = sub.get("users") or []
        if not users:
            raise ValueError("topn/seq_topn requires bundle.meta['submission']['users']")

        # K: cfg.train.topk > meta.submission.k > default 10
        k = 10
        try:
            k = int(getattr(getattr(self.cfg, "train", None), "topk", 10))
        except Exception:
            pass
        try:
            if sub.get("k") is not None:
                k = int(sub["k"])
        except Exception:
            pass

        # batch size for scoring (avoid huge dense [n_users, n_items])
        mcfg = self._model_args(model_name)
        batch_size = int(mcfg.get("predict_batch_size", 2048))

        ckpt_path = self._resolve_checkpoint(checkpoint, f"{model_name}.joblib")
        obj = self._load_checkpoint(ckpt_path)
        ckpt = obj["ckpt"]

        from scipy import sparse
        from src.models.sklearn.topn.ease_scipy import EASEScipy

        # rebuild X on the fly from train to score (simple & consistent)
        user_col = str(bundle.schema["user_col"])
        item_col = str(bundle.schema["item_col"])
        X, user_id_to_idx, item_id_to_idx, idx_to_item_id, popular_items = EASEScipy.build_implicit_matrix(
            bundle.train, user_col=user_col, item_col=item_col
        )

        # use stored B if available
        model = EASEScipy(lambda_reg=float(getattr(ckpt, "lambda_reg", 100.0)))
        model.B = np.asarray(getattr(ckpt, "B"), dtype=np.float32)

        # Prepare predictions aligned to submission users
        preds: List[List[int]] = []
        # For missing users (cold), output popular items
        popular_fallback = list(getattr(ckpt, "popular_items", popular_items)) or popular_items

        # Build list of user indices (with -1 for cold)
        user_indices: List[int] = []
        for u in users:
            try:
                uid = int(u)
            except Exception:
                uid = u
            idx = user_id_to_idx.get(uid, -1)
            user_indices.append(idx)

        # score warm users in batches
        i = 0
        while i < len(users):
            j = min(i + batch_size, len(users))
            batch = user_indices[i:j]

            # separate warm/cold positions
            warm_pos = [p for p, ui in enumerate(batch) if ui >= 0]
            cold_pos = [p for p, ui in enumerate(batch) if ui < 0]

            batch_out: List[Optional[List[int]]] = [None] * len(batch)

            # warm
            if warm_pos:
                warm_user_idx = [batch[p] for p in warm_pos]
                recs = model.recommend_batch(
                    X,
                    user_indices=warm_user_idx,
                    idx_to_item_id=idx_to_item_id,
                    k=k,
                    popular_items=popular_fallback,
                )
                for p, r in zip(warm_pos, recs):
                    batch_out[p] = r

            # cold
            for p in cold_pos:
                batch_out[p] = popular_fallback[:k]

            preds.extend([r or popular_fallback[:k] for r in batch_out])
            i = j

        self._validate_preds(preds, bundle)

        self._log_predict(
            {
                "engine_family": "sklearn_topn",
                "model": model_name,
                "checkpoint_used": ckpt_path,
                "n_users_submit": int(len(users)),
                "topk": int(k),
                "batch_size": int(batch_size),
            }
        )
        return preds

    # ---------- helpers ----------
    @staticmethod
    def _guard_topn(bundle: DataBundle) -> None:
        task = bundle.schema.get("task")
        if task not in ("topn", "seq_topn"):
            raise ValueError("SklearnTopNEngine supports only schema.task in {'topn','seq_topn'}.")

    def _model_name(self) -> str:
        m = getattr(self.cfg, "model", None)
        if m is None:
            return "model"
        name = getattr(m, "name", None)
        return str(name) if name is not None else str(m)

    def _model_args(self, model_name: str) -> Dict[str, Any]:
        # cfg.model_args[model_name] (normalize_config ensures only selected exists)
        margs = getattr(self.cfg, "model_args", {}) or {}
        if isinstance(margs, dict) and model_name in margs:
            return dict(margs[model_name] or {})
        # OmegaConf / DictConfig fallback
        try:
            if model_name in margs:  # type: ignore[operator]
                return dict(margs.get(model_name) or {})  # type: ignore[attr-defined]
        except Exception:
            pass
        return {}

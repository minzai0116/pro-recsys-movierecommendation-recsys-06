from __future__ import annotations

"""
RecBole 엔진 래퍼.

입력:
- cfg.recbole.*: dataset/work_dir/overrides 등
- cfg.train.*: epochs/train_batch_size/eval_batch_size/learning_rate/topk 등
- bundle: DataBundle (schema/meta 포함, 특히 submission.users, user_seq)

출력:
- fit(): dict (best_valid_score, best_valid_result, test_result, checkpoint_path)
- predict(): List[List[int]] (seq_topn 제출용)

부작용:
- recbole용 atomic dataset export
- overrides json 저장(재현성)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

from src.data.data_bundle import DataBundle
from src.engines.core.engine_base import EngineBase
from src.models.recbole.recipes.registry import build_recbole_recipe
from .runner import RecBoleRunner
from omegaconf import OmegaConf

def to_jsonable(obj):
    return OmegaConf.to_container(OmegaConf.create(obj), resolve=True)



class RecBoleEngine(EngineBase):
    """
    RecBole 래퍼 엔진 (Context 계층 없음).

    원칙:
      - 정책/값은 cfg에서만 온다 (Engine이 기본값 "발명" 금지)
      - Engine: cfg 검증 + 디렉토리 생성(setting) + recipe/runner 조립 + logger 요약 기록
      - Recipe: atomic files 생성 + overrides 생성 (모델별)
      - Runner: RecBole 내부(Config/Dataset/Trainer) 캡슐화
    """

    def __init__(self, cfg: Any, logger, setting):
        super().__init__(cfg, logger, setting)
        self.runner = RecBoleRunner()

        # recipe는 models/recbole에서 관리
        self.recipe = build_recbole_recipe(cfg)

        self._validate_cfg()

    # -------------------------
    # cfg validation
    # -------------------------
    def _validate_cfg(self) -> None:
        # 필수: 모델명
        if not getattr(self.cfg, "model", None):
            raise ValueError("cfg.model is required for RecBoleEngine")

        # 필수: seed (Setting.seed_everything은 main에서 호출되는 게 원칙)
        if not hasattr(self.cfg, "seed"):
            raise ValueError("cfg.seed is required for RecBoleEngine")

        # 필수: recbole 섹션
        if not hasattr(self.cfg, "recbole"):
            raise ValueError("cfg.recbole is required for RecBoleEngine")

        # work_dir can be inferred from Setting.run_dir (default: <run_dir>/recbole)
        # so we don't hard-require it here.

        if not getattr(self.cfg.recbole, "dataset", None):
            raise ValueError("cfg.recbole.dataset is required")

        # 필수: 학습 하이퍼 (strict: cfg.train only)
        tcfg = getattr(self.cfg, "train", None)
        if tcfg is None:
            raise ValueError("cfg.train is required (train.* is the single source of truth)")
        for k in ["epochs", "train_batch_size", "eval_batch_size", "learning_rate"]:
            if not hasattr(tcfg, k):
                raise ValueError(f"Missing train hyperparam: {k} (expected under cfg.train.*)")

        # model_args는 prune 전/후 모두 올 수 있으니 여기서는 존재만 강제
        if not hasattr(self.cfg, "model_args"):
            raise ValueError("cfg.model_args is required (even if empty dict)")

    # -------------------------
    # helpers
    # -------------------------
    def _resolve_paths(self) -> Dict[str, str]:
        """
        cfg 기반으로 경로만 계산.
        (Context 없음 — 로컬 dict로만 관리)
        """
        work_dir = getattr(self.cfg.recbole, "work_dir", None)
        if not work_dir:
            base = getattr(self.setting, "run_dir", None) or "."
            work_dir = os.path.join(str(base), "recbole")
        work_dir = str(work_dir)
        data_root = os.path.join(work_dir, "data")
        dataset = str(self.cfg.recbole.dataset)
        dataset_dir = os.path.join(data_root, dataset)

        return {
            "work_dir": work_dir,
            "data_root": data_root,
            "dataset": dataset,
            "dataset_dir": dataset_dir,
        }

    # -------------------------
    # engine API
    # -------------------------
    def fit(self, data_bundle: DataBundle) -> Dict[str, Any]:
        paths = self._resolve_paths()

        # dirs
        self.setting.ensure_dir(paths["work_dir"])
        self.setting.ensure_dir(paths["data_root"])
        self.setting.ensure_dir(paths["dataset_dir"])

        # 1) atomic export (모델별: recipe가 결정)
        ds_spec = self.recipe.prepare_dataset(
            data_bundle,
            data_root=paths["data_root"],
            dataset=paths["dataset"],
            setting=self.setting,
        )

        # 2) overrides (모델별)
        overrides = self.recipe.build_overrides(
            data_bundle,
            data_root=paths["data_root"],
            dataset=paths["dataset"],
        )

        # 3) reproducibility artifact: overrides dump
        # logger.log_dir 아래에 저장
        overrides_path = os.path.join(self.logger.log_dir, "recbole_overrides.json")
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(overrides), f, ensure_ascii=False, indent=2)
        self.logger.log_artifact(overrides_path, name="recbole_overrides")

        # 4) run recbole
        tr = self.runner.fit(
            model_name=str(self.cfg.model),
            dataset_name=paths["dataset"],
            overrides=overrides,
        )

        # 5) 요약 로깅 (step=0)
        self.logger.log_train_metrics(
            {"engine": "recbole", "model": str(self.cfg.model), "dataset": paths["dataset"]},
            step=0,
        )
        self.logger.log_valid_metrics(
            {
                "best_valid_score": tr.get("best_valid_score"),
                **{f"best_valid/{k}": v for k, v in (tr.get("best_valid_result") or {}).items()},
                **{f"test/{k}": v for k, v in (tr.get("test_result") or {}).items()},
            },
            step=0,
        )

        ckpt = tr.get("checkpoint_path")
        if ckpt:
            self.logger.log_artifact(ckpt, name="recbole_best_checkpoint")

        return tr

    def predict(self, bundle: DataBundle, checkpoint: Optional[str] = None) -> Any:
        """
        제출용 preds 생성:
        - 체크포인트 로드
        - full-sort topK 추천
        - user별 item_id 리스트 반환 (problem.save_submission이 받는 형식)
        """
        if not checkpoint:
            raise ValueError("RecBoleEngine.predict requires checkpoint path.")

        self.logger.log_predict_info({
            "engine": "recbole",
            "model": self.cfg.model,
            "checkpoint": checkpoint,
            "stage": "start",
        })

        paths = self._resolve_paths()
        self.setting.ensure_dir(paths["work_dir"])
        self.setting.ensure_dir(paths["data_root"])
        self.setting.ensure_dir(paths["dataset_dir"])

        ds_spec = self.recipe.prepare_dataset(
            bundle,
            data_root=paths["data_root"],
            dataset=paths["dataset"],
            setting=self.setting,
        )

        overrides = self.recipe.build_overrides(
            bundle,
            data_root=paths["data_root"],
            dataset=paths["dataset"],
        )

        objs = self.runner.build(
            model_name=str(self.cfg.model),
            dataset_name=paths["dataset"],
            overrides=overrides,
        )

        config = objs["config"]
        dataset = objs["dataset"]
        test_data = objs["test_data"]
        model = objs["model"]
        trainer = objs["trainer"]

        # 3) 체크포인트 로드
        self.runner.load(trainer=trainer, checkpoint_path=checkpoint)

        # 4) 제출 대상 user 목록 가져오기
        #    pipeline에서 meta["submission"]["users"] 형태로 만든다고 했던 흐름 기준
        sub = (bundle.meta or {}).get("submission", {})
        user_tokens: List[str] = sub.get("users") or sub.get("user_tokens") or []
        user_tokens = [str(u) for u in user_tokens]
        if not user_tokens:
            raise ValueError("bundle.meta['submission']['users'] is required for seq_topn submission.")

        # 5) topK 결정 (cfg 우선, 없으면 10)
        k = 10
        try:
            tcfg = getattr(self.cfg, "train", None)
            k = int(getattr(tcfg, "topk", 10))
        except Exception:
            pass
        try:
            # recbole overrides에 topk 있으면 우선
            user_over = getattr(self.cfg.recbole, "overrides", None)
            if user_over and ("topk" in user_over):
                k = int(user_over["topk"])
        except Exception:
            pass

        # 6) full-sort TopK 추천
        topk_item_tokens = self.runner.fullsort_topk(
            dataset=dataset,
            eval_data=test_data,
            model=model,
            user_tokens=user_tokens,
            k=k,
            device=str(config["device"]),
        )

        # 7) token -> int 변환 (문제 제출 포맷이 int item_id를 기대하는 경우가 대부분)
        preds = []
        for row in topk_item_tokens:
            preds.append([int(x) for x in row])

        self.logger.log_predict_info({
            "engine": "recbole",
            "model": self.cfg.model,
            "stage": "done",
            "users": len(preds),
            "topk": k,
        })
        return preds
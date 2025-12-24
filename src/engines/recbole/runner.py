from __future__ import annotations

"""
RecBole 실행 어댑터(캡슐화 레이어).

입력:
- model_name: str
- dataset_name: str
- overrides: dict (RecBole Config에 주입할 값들)

출력:
- build(): RecBole 내부 객체(config/dataset/dataloader/model/trainer) dict
- fit(): 학습 결과 요약 dict + checkpoint_path
- fullsort_topk(): 외부 user token -> 외부 item token topK
"""

from typing import Any, Dict, List, Optional, Tuple


class RecBoleRunner:
    """
    RecBole 실행 어댑터 (최소·견고).

    책임:
      - RecBole 내부 객체 구성(Config/Dataset/DataLoader/Model/Trainer)
      - 학습(fit) + 체크포인트 경로 반환
      - 체크포인트 로드(버전 차이 흡수)
      - 예측 primitive 제공
          * full-sort TopK (현재 SeqTopN/TopN)
          * pairwise score (향후 regression/rerank)

    비책임:
      - bundle/meta/submission
      - setting/logger
      - task 분기
    """

    # --------------------------------------------------
    # build: 공통 객체 구성
    # --------------------------------------------------
    def build(
        self,
        *,
        model_name: str,
        dataset_name: str,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed, init_logger, get_model, get_trainer

        config = Config(model=model_name, dataset=dataset_name, config_dict=overrides)

        # seed / logger (RecBole 내부 logger는 보조)
        init_seed(config["seed"], _cfg_get(config, "reproducibility", True))
        init_logger(config)

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        return {
            "config": config,
            "dataset": dataset,
            "train_data": train_data,
            "valid_data": valid_data,
            "test_data": test_data,
            "model": model,
            "trainer": trainer,
        }

    # --------------------------------------------------
    # fit: 학습 + 결과 요약
    # --------------------------------------------------
    def fit(
        self,
        *,
        model_name: str,
        dataset_name: str,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        objs = self.build(
            model_name=model_name,
            dataset_name=dataset_name,
            overrides=overrides,
        )

        trainer = objs["trainer"]
        train_data = objs["train_data"]
        valid_data = objs["valid_data"]
        test_data = objs["test_data"]

        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=True,
        )

        test_result = trainer.evaluate(test_data, load_best_model=True)

        ckpt = getattr(trainer, "saved_model_file", None)

        return {
            "best_valid_score": best_valid_score,
            "best_valid_result": dict(best_valid_result) if best_valid_result else {},
            "test_result": dict(test_result) if test_result else {},
            "checkpoint_path": ckpt,
        }

    # --------------------------------------------------
    # load: 체크포인트 로드 (버전 차이 흡수)
    # --------------------------------------------------
    def load(self, *, trainer, checkpoint_path: str) -> None:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # RecBole/Lightning/순수 state_dict 등 방어
        state = (
            ckpt.get("state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("model")
            or ckpt
        )

        trainer.model.load_state_dict(state, strict=False)

    # --------------------------------------------------
    # predict primitive A: full-sort TopK
    # --------------------------------------------------
    def fullsort_topk(
        self,
        *,
        dataset,
        eval_data,
        model,
        user_tokens: List[str],
        k: int,
        device: Optional[str] = None,
    ) -> List[List[str]]:
        """
        외부 user token -> 외부 item token TopK

        요구사항:
          - eval_data가 FullSortEvalDataLoader여야 함
            (recipe에서 eval_args.mode='full' 보장)
        """
        import numpy as np
        from recbole.utils.case_study import full_sort_topk

        uid_field = dataset.uid_field
        iid_field = dataset.iid_field

        # external token -> internal id
        uid_arr = np.asarray(dataset.token2id(uid_field, user_tokens))

        # internal iid 반환
        _, topk_iid = full_sort_topk(
            uid_arr,
            model,
            eval_data,
            k=k,
            device=device,
        )

        # internal id -> external token
        topk_tokens = dataset.id2token(iid_field, topk_iid.cpu())

        return [list(row) for row in topk_tokens]

    # --------------------------------------------------
    # predict primitive B: pairwise score (향후용)
    # --------------------------------------------------
    def score_pairs(
        self,
        *,
        dataset,
        model,
        user_tokens: List[str],
        item_tokens: List[str],
        device: Optional[str] = None,
    ) -> List[float]:
        """
        (u,i) 쌍 점수 예측.
        - regression / rerank 대비
        """
        import torch
        import numpy as np

        uid_field = dataset.uid_field
        iid_field = dataset.iid_field

        uids = np.asarray(dataset.token2id(uid_field, user_tokens))
        iids = np.asarray(dataset.token2id(iid_field, item_tokens))

        # RecBole 모델은 (uids, iids) -> score 지원하는 경우가 대부분
        with torch.no_grad():
            scores = model.predict(uids, iids)

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        return scores.tolist()

def _cfg_get(config, key, default=None):
    try:
        return config[key]
    except Exception:
        return default

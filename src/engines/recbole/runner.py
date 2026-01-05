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
        from recbole.data import create_dataset
        from recbole.data.utils import create_samplers, get_dataloader
        from recbole.data.dataloader import FullSortEvalDataLoader
        from recbole.sampler import KGSampler
        from recbole.utils import init_seed, init_logger, get_model, get_trainer
        from recbole.utils.enum_type import ModelType

        overrides = _normalize_recbole_config_dict(overrides)
        
        # SSoT: Centralized logging. Disable RecBole's default tensorboard to avoid extra folders.
        overrides.setdefault("enable_tensorboard", False)
        overrides.setdefault("enable_scaler", False) # Often causes extra logging
        
        config = Config(model=model_name, dataset=dataset_name, config_dict=overrides)

        # seed / logger (RecBole 내부 logger는 보조)
        init_seed(config["seed"], _cfg_get(config, "reproducibility", True))
        init_logger(config)

        dataset = create_dataset(config)

        # RecBole 기본 data_preparation은 valid/test split이 "0건"인 경우
        # FullSortEvalDataLoader 내부에서 예외가 날 수 있어(특히 split=[1,0,0]),
        # 여기서 직접 구성하면서 안전장치를 둔다.
        built_datasets = dataset.build()
        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

        model_type = config["MODEL_TYPE"]
        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )

        # valid/test는 비어있을 수 있으므로 None 허용
        valid_data = None
        if len(valid_dataset) > 0:
            valid_data = get_dataloader(config, "valid")(
                config, valid_dataset, valid_sampler, shuffle=False
            )

        test_data_is_fallback = False
        test_data = None
        if len(test_dataset) > 0:
            test_data = get_dataloader(config, "test")(
                config, test_dataset, test_sampler, shuffle=False
            )
        else:
            # train-only(split에서 test가 0건)일 때도 제출용 full-sort는 필요함.
            # test_data를 train_dataset 기반 FullSortEvalDataLoader로 대체한다.
            #
            # 중요: FullSortEvalDataLoader는 sampler.used_ids를 "history(이미 본 아이템)"로 사용한다.
            # test_dataset이 비어있는 상황에서 test_sampler(phase=test)를 넘기면 used_ids가 비어
            # seen 마스킹이 깨져 추천에 학습 아이템이 그대로 섞일 수 있다.
            # 따라서 fallback에서는 train_sampler(phase=train)을 우선 사용한다.
            sampler_for_fallback = train_sampler or valid_sampler or test_sampler
            test_data = FullSortEvalDataLoader(config, train_dataset, sampler_for_fallback, shuffle=False)
            test_data_is_fallback = True

        model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        return {
            "config": config,
            "dataset": dataset,
            "train_data": train_data,
            "valid_data": valid_data,
            "test_data": test_data,
            "test_data_is_fallback": test_data_is_fallback,
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
        test_data_is_fallback = bool(objs.get("test_data_is_fallback", False))

        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=True,
        )

        # train-only(fallback test_data)에서는 '평가'가 의미 없고(ground truth 없음),
        # 일부 설정에선 evaluator가 에러를 낼 수 있어 생략한다.
        test_result = {}
        if (not test_data_is_fallback) and test_data:
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


def _normalize_recbole_config_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    RecBole Config(config_dict=...)에 들어가기 직전 dict를 정규화한다.

    - OmegaConf(DictConfig/ListConfig)가 섞여도 순수 python dict/list로 변환
    - RecBole이 dict를 요구하는 키(eval_args 등)가 실수로 list([dict])로 들어오면 unwrap
    """

    def _to_py(obj):
        # 1) OmegaConf가 있으면 가장 안전하게 container로 변환
        try:
            from omegaconf import OmegaConf  # type: ignore

            return OmegaConf.to_container(OmegaConf.create(obj), resolve=True)
        except Exception:
            pass

        # 2) 기본 재귀 변환
        if isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_py(x) for x in obj]
        # DictConfig류는 items()가 있으니 dict()로 시도
        try:
            if hasattr(obj, "items"):
                return {k: _to_py(v) for k, v in dict(obj).items()}
        except Exception:
            pass
        return obj

    out = _to_py(d)
    if not isinstance(out, dict):
        # 방어: config_dict는 dict여야 함
        return d

    # 흔한 실수 방어: list([dict]) -> dict
    for key in ["eval_args", "train_neg_sample_args", "valid_neg_sample_args", "test_neg_sample_args"]:
        v = out.get(key)
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
            out[key] = v[0]

    # nested eval_args도 방어 (eval_args 자체는 dict여야 함)
    if isinstance(out.get("eval_args"), list) and len(out["eval_args"]) == 1 and isinstance(out["eval_args"][0], dict):
        out["eval_args"] = out["eval_args"][0]

    return out

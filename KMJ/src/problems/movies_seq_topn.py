from __future__ import annotations

"""
MovieLens Seq Top-K 제출용 Problem 구현.

입력:
- bundle.meta["submission"]["users"]: 제출 대상 user 순서 (sample_submission 기반)
- preds: List[List[int]] (len(preds)==len(users), 각 row는 topK item_id)

출력:
- save_submission(): `<train.submit_dir>/<model>.csv` 형태로 저장 후 경로 반환
- evaluate_preds(): (선택) 간단 sanity metric(dict) 반환

주의:
- 이 대회는 sample_submission 템플릿을 “채워 제출”하는 형태를 전제로 합니다.
"""

from typing import Any, Optional, List
import pandas as pd

from src.data.data_bundle  import DataBundle
from src.problems.base import ProblemBase
from src.factories.pipeline_factory import PipelineFactory
from src.problems.registry import register_problem

@register_problem("movies_seq_topn", pipelines=["seq_topn_v1"])
class MoviesSeqTopNProblem(ProblemBase):
    name = "movies_seq_topn"

    def build_data_bundle(self) -> DataBundle:
        pipeline_name = self.cfg.data.get("pipeline") or self.cfg.problem.get("pipeline")
        if not pipeline_name:
            raise ValueError("cfg.data.pipeline is required for movies_seq_topn")
        pipeline = PipelineFactory.build(self.cfg)
        return pipeline.build(self.cfg)

    def save_submission(
        self,
        preds: List[List[int]],
        cfg: Any,
        setting,
        bundle: DataBundle,
    ) -> Optional[str]:

        # validate_bundle에서 이미 meta['submission']['users']를 강제한다고 가정
        bundle_users = bundle.meta["submission"]["users"]
        # 유저 수가 다르면 에러 대신 경고만 출력하고 함수 종료
        if len(preds) != len(bundle_users):
            print(f"\n[Warning] 예측된 유저 수({len(preds)})가 전체 유저 수({len(bundle_users)})와 다릅니다.")
            print("         LightGBM 학습용 데이터(Partial) 생성 중이라면 이 메시지는 정상입니다.")
        
        # 데이터 타입 자동 감지 (첫 번째 유저의 첫 번째 아이템 확인)
        is_score_mode = False
        if len(preds) > 0 and len(preds[0]) > 0:
            first_item = preds[0][0]
            if isinstance(first_item, (tuple, list)):
                is_score_mode = True

        result = []
        if is_score_mode:
            # --- [Mode A] 점수/랭킹 포함 (Feature Engineering용) ---
            for idx, user_preds in enumerate(preds):
                u = bundle_users[idx]
                for rank, (item, score) in enumerate(user_preds):
                    result.append((u, item, rank + 1, score))
            columns = ["user", "item", "rank", "score"]
            
        else:
            # --- [Mode B] 기본 제출용 (item만 있음) ---
            for idx, items in enumerate(preds):
                u = bundle_users[idx]
                for item in items:
                    result.append((u, item))
            columns = ["user", "item"]

        sub_df = pd.DataFrame(result, columns=columns)
        train_cfg = cfg.get("train", {})  # train 없으면 {}
        submit_dir = train_cfg.get("submit_dir", "saved/submit")

        out_path = setting.get_submit_path(
            base_dir=submit_dir,
            model=cfg.model,
            run_name=cfg.get("run_name"),
        )
        sub_df.to_csv(out_path, index=False)
        return out_path

    def evaluate_preds(self, preds: List[List[int]], cfg: Any, bundle: DataBundle) -> dict | None:
        """
        Competition-friendly metric for this problem:
        - Recall@K computed as hit-rate of the held-out last item per user.

        NOTE:
        - We don't have public labels for eval in the competition setting.
        - This provides a lightweight, always-available sanity metric using train history.
        """
        meta = bundle.meta or {}
        sub = meta.get("submission", {}) or {}
        users = sub.get("users") or []
        user_seq = meta.get("user_seq") or {}

        # 평가 함수도 타입 감지하여 처리
        is_score_mode = False
        if len(preds) > 0 and len(preds[0]) > 0:
            if isinstance(preds[0][0], (tuple, list)):
                is_score_mode = True

        if not users or not user_seq:
            return None

        # K: prefer cfg.train.topk, fallback 10
        k = 10
        try:
            k = int(cfg.get("train", {}).get("topk", 10))
        except Exception:
            pass

        hits = 0
        total = 0
        for u, recs in zip(users, preds):
            seq = user_seq.get(u)
            if not seq:
                continue
            gt = seq[-1]
            try:
                gt = int(gt)
            except Exception:
                gt = gt
            total += 1

            # 아이템 ID 추출
            if is_score_mode:
                # (item, score) 튜플에서 item만 꺼냄
                rec_items = [x[0] for x in recs[:k]]
            else:
                rec_items = recs[:k]

            if gt in rec_items:
                hits += 1

        if total == 0:
            return None

        return {f"Recall@{k}": hits / total}
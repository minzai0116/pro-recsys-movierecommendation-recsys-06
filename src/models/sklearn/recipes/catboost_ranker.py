# src/models/sklearn/recipes/catboost_ranker.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from src.models.sklearn.recipes.base import SklearnRegressionRecipeSpec
from src.models.sklearn.recipes.registry import register_sklearn_recipe
from src.utils.cfg_utils import cfg_select
from src.models.sklearn.recipes.ranker_utils import add_ranker_features, merge_candidate_files

@register_sklearn_recipe("catboost_ranker")
class CatBoostRankerRecipe(SklearnRegressionRecipeSpec):
    """
    2-stage CatBoost Ranker(재랭킹).
    - 여러 후보 CSV를 병합해 후보 풀을 만들고
    - item/user 통계 + 메타데이터 피처를 생성한 뒤
    - valid(마스킹으로 분리된 정답) 기준으로 라벨링하여 학습합니다.

    유출 방지 원칙:
    - label=1(정답)은 bundle.valid에서만 정의
    - bundle.train(이미 본 아이템)은 후보에서 제거
    """
    name = "catboost_ranker"

    def fit(self, bundle: Any, engine: Any) -> Dict[str, Any]:
        margs = engine._model_args("CatBoostRanker")
        user_col = str(bundle.schema.get("user_col", "user"))
        item_col = str(bundle.schema.get("item_col", "item"))

        # 1) 후보 로드 (여러 모델 CSV 병합)
        candidate_files = cfg_select(self.cfg, "recipe_args.candidate_files", default={})
        if not candidate_files:
            print("[WARN] recipe_args.candidate_files가 비어있음: bundle.train을 후보로 사용(비권장)")
            candidate_df = bundle.train.copy()
        else:
            candidate_df = merge_candidate_files(candidate_files)

        # 2) 라벨링: valid(마스킹된 정답)을 ground truth로 사용
        # - 후보 CSV에 정답이 하나도 없으면 학습이 불가능하므로(valid positive를 후보에 강제로 포함)
        if bundle.valid is None or len(bundle.valid) == 0:
            raise ValueError("CatBoost 학습에는 bundle.valid(마스킹 정답)가 필요합니다. (valid가 비어있음)")

        print(f"[Ranker] 라벨링 | valid_interactions={len(bundle.valid)}")
        valid_pos = bundle.valid[[user_col, item_col]].drop_duplicates().copy()
        valid_pos["label"] = 1.0

        # 후보 U 정답(positive) 합치기: label=1이 반드시 존재하도록 보장
        candidate_df = pd.concat([candidate_df, valid_pos[[user_col, item_col]]], ignore_index=True)
        candidate_df = candidate_df.drop_duplicates(subset=[user_col, item_col])

        # 벡터화 라벨링(merge): 정답이면 1, 아니면 0
        candidate_df = candidate_df.merge(valid_pos, on=[user_col, item_col], how="left")
        candidate_df["label"] = candidate_df["label"].fillna(0.0).astype("float32")

        # 3) 마스킹: train(이미 본 아이템)은 후보에서 제거
        train_set = set(zip(bundle.train[user_col], bundle.train[item_col]))
        candidate_df = candidate_df[~candidate_df.apply(lambda row: (row[user_col], row[item_col]) in train_set, axis=1)]

        # 3.5) 안전장치: 라벨이 한쪽(전부 0 또는 전부 1)이면 CatBoost가 학습 불가
        n_total = int(len(candidate_df))
        n_pos = int((candidate_df["label"] > 0.0).sum()) if n_total else 0
        if n_total == 0:
            raise ValueError("마스킹 후 후보가 0건입니다. candidate_files / split 설정을 확인하세요.")
        if n_pos == 0:
            raise ValueError("positive(label=1)가 0건입니다. user/item 키 정합성을 확인하세요.")
        if n_pos == n_total:
            print("[WARN] 라벨이 전부 1입니다. 간단 negative 샘플을 추가합니다.")
            rng = np.random.default_rng(int(cfg_select(self.cfg, "seed", default=42)))
            neg_per_user = int(margs.get("neg_per_user", 20))

            item_pop = bundle.train[item_col].value_counts()
            popular_items = item_pop.index.to_numpy()

            valid_set = set(zip(valid_pos[user_col].to_list(), valid_pos[item_col].to_list()))
            cand_set = set(zip(candidate_df[user_col].to_list(), candidate_df[item_col].to_list()))

            users = valid_pos[user_col].unique()
            neg_rows = []
            for u in users:
                trials = 0
                added = 0
                while added < neg_per_user and trials < neg_per_user * 50:
                    it = int(rng.choice(popular_items))
                    key = (u, it)
                    trials += 1
                    if key in train_set or key in valid_set or key in cand_set:
                        continue
                    neg_rows.append({user_col: u, item_col: it, "label": 0.0})
                    cand_set.add(key)
                    added += 1

            if neg_rows:
                candidate_df = pd.concat([candidate_df, pd.DataFrame(neg_rows)], ignore_index=True)

        # 4) 피처 생성
        data_dir = cfg_select(self.cfg, "dataset.data_path", default="data/movielens/train")
        train_df = add_ranker_features(
            candidate_df=candidate_df,
            train_ratings=bundle.train,
            data_dir=data_dir,
            verbose=True
        )
        
        # 학습 피처 컬럼 식별
        exclude_cols = ['group_id', 'group', user_col, 'label', item_col, 'user_last_time', 'time']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        cat_features = [c for c in feature_cols if train_df[c].dtype.name in ('category', 'object')]
        
        group_col = 'group_id'
        train_df[group_col] = train_df[user_col].astype('int32')
        train_df_sorted = train_df.sort_values(group_col).reset_index(drop=True)
        
        train_pool = Pool(
            data=train_df_sorted[feature_cols],
            label=train_df_sorted['label'],
            group_id=train_df_sorted[group_col],
            cat_features=cat_features if cat_features else None
        )
        
        # 5) CatBoost 하이퍼
        params = {
            'loss_function': margs.get('loss_function', 'QueryRMSE'),
            'iterations': margs.get('iterations', 1000),
            'learning_rate': margs.get('learning_rate', 0.05),
            'depth': margs.get('depth', 6),
            'random_seed': 42,
            'verbose': 100,
            'task_type': 'GPU' if margs.get('use_gpu', True) else 'CPU',
        }
        
        model = CatBoostRanker(**params)
        model.fit(train_pool)
        
        return {
            "model": model,
            "feature_cols": feature_cols,
            "metrics": {"train_loss": model.get_best_score().get("learn", {})}
        }

    def predict(self, model: CatBoostRanker, bundle: Any, engine: Any) -> List[List[int]]:
        user_col = str(bundle.schema.get("user_col", "user"))
        item_col = str(bundle.schema.get("item_col", "item"))
        
        # 1. Load Test Candidates (External CSVs)
        candidate_files = cfg_select(self.cfg, "recipe_args.candidate_files", default={})
        if not candidate_files:
            candidate_df = bundle.test if bundle.test is not None else bundle.train
        else:
            # We assume for prediction we use the same CSVs or special 'test' CSV paths
            # Usually, you'd provide paths to 'test' candidates in config
            candidate_df = merge_candidate_files(candidate_files)

        # 2. Feature Engineering
        data_dir = cfg_select(engine.cfg, "dataset.data_path", default="data/movielens/train")
        test_df = add_ranker_features(
            candidate_df=candidate_df,
            train_ratings=bundle.train,
            data_dir=data_dir,
            verbose=True
        )
        
        exclude_cols = ['group_id', 'group', user_col, 'label', item_col, 'user_last_time', 'time']
        feature_cols = [c for c in test_df.columns if c not in exclude_cols]
        
        group_col = 'group_id'
        test_df[group_col] = test_df[user_col].astype('int32')
        test_df_sorted = test_df.sort_values(group_col).reset_index(drop=True)
        cat_features = [c for c in feature_cols if test_df_sorted[c].dtype.name in ('category', 'object')]
                
        test_pool = Pool(
            data=test_df_sorted[feature_cols],
            group_id=test_df_sorted[group_col],
            cat_features=cat_features if cat_features else None
        )
        
        scores = model.predict(test_pool)
        test_df_sorted['score'] = scores
        
        # 3. Final Re-ranking
        k = int(cfg_select(engine.cfg, "train.topk", default=10))
        sub_users = bundle.meta['submission']['users']
        user_groups = test_df_sorted.groupby(user_col)
        
        preds: List[List[int]] = []
        for uid in sub_users:
            if uid in user_groups.groups:
                grp = user_groups.get_group(uid)
                top_items = grp.sort_values('score', ascending=False).head(k)[item_col].tolist()
                preds.append(top_items)
            else:
                preds.append([]) 
        return preds

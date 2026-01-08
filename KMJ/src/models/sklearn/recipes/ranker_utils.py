# src/models/sklearn/recipes/ranker_utils.py
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

def load_ml_metadata(data_dir: str) -> Dict[str, pd.DataFrame]:
    """MovieLens 메타데이터 TSV를 로드합니다. (genres/directors/writers/years)"""
    meta = {}
    files = {"genres": "genres.tsv", "directors": "directors.tsv", "writers": "writers.tsv", "years": "years.tsv"}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            if key == "years":
                meta[key] = df.drop_duplicates('item').set_index('item')
            else:
                grouped = df.groupby('item').agg(lambda x: '|'.join(map(str, x)))
                meta[key] = grouped
    return meta

def merge_candidate_files(file_paths: Dict[str, str], verbose: bool = True) -> pd.DataFrame:
    """
    여러 모델의 후보 CSV를 (user,item) 기준으로 병합합니다.
    - file_paths: {모델명: csv경로}
    - 입력 CSV 컬럼: user,item (long format)
    - 각 모델별 rank_<model>, exists_<model> 피처를 생성합니다.
    """
    merged_df = pd.DataFrame()
    
    for model_name, path in file_paths.items():
        if not os.path.exists(path):
            if verbose:
                print(f"[WARN] 후보 파일 없음 | model={model_name}, path={path}")
            continue
            
        df = pd.read_csv(path)
        # CSV는 (user,item) long format을 가정하며,
        # user별 등장 순서(cumcount)를 rank로 사용합니다.
        df['rank_' + model_name] = df.groupby('user').cumcount() + 1
        df['exists_' + model_name] = 1
        
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=['user', 'item'], how='outer')
            
    # 특정 모델이 추천하지 않은 후보는 rank를 큰 값으로(패널티), exists=0으로 채웁니다.
    for model_name in file_paths.keys():
        if 'rank_' + model_name in merged_df.columns:
            merged_df['rank_' + model_name] = merged_df['rank_' + model_name].fillna(200)
            merged_df['exists_' + model_name] = merged_df['exists_' + model_name].fillna(0)
            
    if verbose:
        print(f"[Ranker] 후보 병합 완료 | models={len(file_paths)}, candidates={len(merged_df)}")
    return merged_df

def add_ranker_features(
    candidate_df: pd.DataFrame, 
    train_ratings: pd.DataFrame, 
    data_dir: str,
    verbose: bool = True
) -> pd.DataFrame:
    """CatBoost 재랭킹용 피처를 생성합니다."""
    df = candidate_df.copy()
    
    user_col, item_col = 'user', 'item'
    if verbose:
        print(f"[Ranker] 피처 생성 | candidates={len(df)}")

    # 1. Item Profile
    item_pop = train_ratings[item_col].value_counts()
    df['item_popularity_log'] = np.log1p(df[item_col].map(item_pop).fillna(0)).astype('float32')
    
    meta = load_ml_metadata(data_dir)
    if "years" in meta:
        df = df.merge(meta["years"], left_on=item_col, right_index=True, how='left')
        # DataFrame.rename(..., inplace=True)는 None을 반환하므로 체이닝 금지.
        df.rename(columns={'year': 'item_year'}, inplace=True)
        df['item_year'] = df['item_year'].fillna(0).astype('int16')

    for key in ["genres", "directors", "writers"]:
        if key in meta:
            df = df.merge(meta[key], left_on=item_col, right_index=True, how='left')
            col_name = f'item_{key[:-1]}'
            df.rename(columns={key[:-1]: col_name}, inplace=True)
            df[col_name] = df[col_name].fillna('Unknown').apply(lambda x: x.split('|')[0]).astype('category')

    # 2. User Profile
    user_counts = train_ratings.groupby(user_col)[item_col].nunique()
    df['user_total_watched'] = df[user_col].map(user_counts).fillna(0).astype('int32')
    
    top_20_items = set(item_pop.head(int(len(item_pop) * 0.2)).index)
    user_ms = train_ratings.assign(is_ms=train_ratings[item_col].isin(top_20_items)).groupby(user_col)['is_ms'].mean()
    df['user_mainstream_ratio'] = df[user_col].map(user_ms).fillna(0.5).astype('float32')

    # 3. Contextual
    if 'time' in train_ratings.columns:
        last_interactions = train_ratings.sort_values([user_col, 'time']).groupby(user_col).tail(1)
        if "genres" in meta:
            last_item_info = last_interactions.merge(meta["genres"], left_on=item_col, right_index=True, how='left')
            last_item_info['first_genre'] = last_item_info['genre'].fillna('Unknown').apply(lambda x: x.split('|')[0])
            user_last_genre = last_item_info.set_index(user_col)['first_genre']
            df['user_last_item_genre'] = df[user_col].map(user_last_genre).fillna('Unknown').astype('category')

    # 4. Fill NaNs for any model scores/ranks
    score_cols = [c for c in df.columns if 'rank_' in c or 'exists_' in c or 'score' in c]
    for c in score_cols:
        df[c] = df[c].fillna(0.0).astype('float32')

    return df

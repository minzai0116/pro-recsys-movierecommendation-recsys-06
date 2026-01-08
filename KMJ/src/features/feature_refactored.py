"""피처 엔지니어링 (2-Stage CatBoost Ranker용 - 최종 모듈화 버전)."""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# =============================================================================
# [Category 1] Model Score Features (1단계 모델 점수 및 교차 피처)
# =============================================================================
def add_model_score_features(df: pd.DataFrame, scores_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """EASE/VAE 점수 및 하이브리드 병합 시 생성된 교차 정보를 통합합니다."""
    if verbose: print("  [1/4] 모델 점수 및 교차 피처 통합 중...")
    
    # 필수 점수 및 병합 피처 컬럼 식별
    target_cols = [col for col in scores_df.columns if 
                   col.startswith('ease_score_norm_lambda') or 
                   col in ['vae_score', 'is_hybrid', 'ease_vae_score_diff']]
    
    df = df.merge(scores_df[['user_id', 'item_id'] + target_cols], on=['user_id', 'item_id'], how='left')
    
    # 결측치 0.0 처리 및 타입 고정
    for col in target_cols:
        df[col] = df[col].fillna(0.0).astype('float32')
        
    return df

# =============================================================================
# [Category 2] Item Profile Features (아이템 메타데이터 및 통계)
# =============================================================================
def add_item_profile_features(df: pd.DataFrame, train_df: pd.DataFrame, metadata_path: Optional[str], verbose: bool = True) -> pd.DataFrame:
    """아이템 인기도, 트렌드, 개봉 연도 및 제작진 정보를 처리합니다."""
    if verbose: print("  [2/4] 아이템 프로필 피처 계산 중...")
    
    # 1. 인기도 및 트렌드
    item_pop = train_df['item'].value_counts().to_dict()
    df['item_popularity_log'] = np.log1p(df['item_id'].map(item_pop).fillna(0)).astype('float32')
    
    if 'time' in train_df.columns:
        max_time = train_df['time'].max()
        pop_30d = train_df[train_df['time'] >= (max_time - 30 * 86400)]['item'].value_counts().to_dict()
        df['item_pop_recent_ratio'] = (df['item_id'].map(pop_30d).fillna(0) / 
                                       (df['item_id'].map(item_pop).fillna(0) + 1)).astype('float32')

    # 2. 메타데이터 (Year, Director, Writer)
    if metadata_path:
        meta_df = pd.read_csv(metadata_path)
        df = df.merge(meta_df[['item', 'director', 'writer', 'year']], 
                      left_on='item_id', right_on='item', how='left').drop(columns=['item'])
        df = df.rename(columns={'director': 'item_director', 'writer': 'item_writer', 'year': 'item_year'})
        
        # 타입 최적화
        df['item_director'] = df['item_director'].fillna('Unknown').astype('category')
        df['item_writer'] = df['item_writer'].fillna('Unknown').astype('category')
        df['item_year'] = df['item_year'].fillna(0).astype('int16')
        
    return df

# =============================================================================
# [Category 3] User Profile Features (유저 활동성 및 취향 성향)
# =============================================================================
def add_user_profile_features(df: pd.DataFrame, train_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """유저의 총 시청량 및 대중성 선호도를 계산합니다."""
    if verbose: print("  [3/4] 유저 프로필 피처 계산 중...")
    
    # 1. 활동량 (숫자 피처)
    user_counts = train_df.groupby('user')['item'].nunique().to_dict()
    df['user_total_watched'] = df['user_id'].map(user_counts).fillna(0).astype('int32')
    
    # 2. 메인스트림 비율
    item_counts = train_df['item'].value_counts()
    top_20_items = set(item_counts.head(int(len(item_counts) * 0.2)).index)
    user_mainstream = train_df.assign(is_ms=train_df['item'].isin(top_20_items)).groupby('user')['is_ms'].mean().to_dict()
    df['user_mainstream_ratio'] = df['user_id'].map(user_mainstream).fillna(0.5).astype('float32')
    
    return df

# =============================================================================
# [Category 4] Contextual & Sequential Features (실시간 맥락 및 장르 유사도)
# =============================================================================
def add_contextual_features(df: pd.DataFrame, train_df: pd.DataFrame, metadata_path: Optional[str], verbose: bool = True) -> pd.DataFrame:
    """마지막 시청 영화 정보와 장르 유사도 피처를 복구하여 반영합니다."""
    if verbose: print("  [4/4] 컨텍스트 및 시퀀셜 피처 계산 중...")
    
    if 'time' in train_df.columns and metadata_path:
        meta_df = pd.read_csv(metadata_path).set_index('item')
        item_genres = meta_df['genres'].to_dict()
        
        # 1. 마지막 아이템 장르 (순차 정보)
        last_items = train_df.sort_values(['user', 'time']).groupby('user').tail(1).set_index('user')['item'].to_dict()
        def get_top_genre(g_str): return g_str.split('|')[0].split(':')[0] if pd.notna(g_str) and g_str != '' else 'Unknown'
        
        last_item_genres = {u: get_top_genre(item_genres.get(i, '')) for u, i in last_items.items()}
        df['user_last_item_genre'] = df['user_id'].map(last_item_genres).fillna('Unknown').astype('category')
        
        # 2. 장르 유사도 복구 (Last Item & User Preference)
        # (유사도 계산 로직은 기존 calculate_last_item_similarity 참조하여 경량화 구현 필요)
        # 현재는 핵심 범주형 피처인 user_last_item_genre 유지에 집중
        
    return df

# =============================================================================
# [Main Entry] Dataset Builder
# =============================================================================
def create_ranker_dataset(scores_df: pd.DataFrame, train_df: pd.DataFrame, val_ground_truth: Dict[int, Set[int]], 
                         metadata_path: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    """4대 카테고리 함수를 결합하여 최종 Ranker 학습용 데이터셋을 생성합니다."""
    
    df = scores_df[['user_id', 'item_id']].copy()
    
    # 순차적 피처 주입
    df = add_model_score_features(df, scores_df, verbose)
    df = add_item_profile_features(df, train_df, metadata_path, verbose)
    df = add_user_profile_features(df, train_df, verbose)
    df = add_contextual_features(df, train_df, metadata_path, verbose)
    
    # 레이블링 및 최종 정리
    if val_ground_truth:
        gt_df = pd.DataFrame([{'user_id': u, 'item_id': i, 'label': 1} for u, items in val_ground_truth.items() for i in items])
        df = df.merge(gt_df, on=['user_id', 'item_id'], how='left')
        df['label'] = df['label'].fillna(0).astype('int8')
        
    df['group_id'] = df['user_id'].astype('int32')
    if verbose: print(f"🚀 최종 피처셋 생성 완료: {len(df.columns)}개 피처")
    
    return df
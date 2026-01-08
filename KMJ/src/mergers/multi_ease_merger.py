"""Multi-EASE 후보 병합 모듈 (메모리/시간 최적화)."""
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm


def merge_multi_ease_candidates(
    candidate_dfs: List[pd.DataFrame],
    lambda_values: List[float],
    k_final: int = 200,
    verbose: bool = True
) -> pd.DataFrame:
    """
    여러 EASE 모델의 후보를 병합 (메모리/시간 최적화).
    
    전략:
    1. 각 모델의 점수를 유저별로 MinMax 정규화
    2. Lambda 값에 따른 가중치 적용:
       - λ 작을수록 (공격적) → 가중치 낮음
       - λ 클수록 (보수적) → 가중치 높음
       - 인기작(λ↑)과 롱테일(λ↓)의 균형 유지
    3. Outer Join으로 병합 (중복 제거)
    4. Combined Score로 정렬 후 Top-K 선택
    
    Args:
        candidate_dfs: 각 EASE 모델의 후보 DataFrame 리스트
                      (columns: user_id, item_id, ease_score, ease_rank)
        lambda_values: 각 모델의 lambda 값 리스트
        k_final: 병합 후 최종 후보 수 (유저당)
        verbose: 진행 상황 출력
    
    Returns:
        병합된 후보 DataFrame (columns: user_id, item_id, ease_score_norm, ...)
    """
    if verbose:
        print("\n" + "="*70)
        print(f"🔄 Multi-EASE 후보 병합 (메모리 최적화)")
        print("="*70)
        print(f"  모델 수: {len(candidate_dfs)}개")
        print(f"  Lambda 값: {lambda_values}")
        print(f"  최종 후보 수: {k_final}개/유저")
    
    # ========================================
    # Step 1: 각 모델의 후보에 lambda 정보 추가
    # ========================================
    if verbose:
        print("\n[1/4] Lambda 정보 추가 중...")
    
    for i, (df, lambda_val) in enumerate(zip(candidate_dfs, lambda_values)):
        df[f'ease_score_lambda{int(lambda_val)}'] = df['ease_score'].astype('float32')
        df[f'ease_rank_lambda{int(lambda_val)}'] = df['ease_rank'].astype('int16')
    
    # ========================================
    # Step 2: Outer Join 병합 (메모리 효율적)
    # ========================================
    if verbose:
        print("\n[2/4] 후보 병합 중 (Outer Join)...")
    
    merged = candidate_dfs[0][['user_id', 'item_id', 
                                f'ease_score_lambda{int(lambda_values[0])}',
                                f'ease_rank_lambda{int(lambda_values[0])}']].copy()
    
    # 순차적 병합 (메모리 최적화)
    for i in range(1, len(candidate_dfs)):
        lambda_val = lambda_values[i]
        right_df = candidate_dfs[i][['user_id', 'item_id',
                                      f'ease_score_lambda{int(lambda_val)}',
                                      f'ease_rank_lambda{int(lambda_val)}']]
        
        merged = merged.merge(
            right_df,
            on=['user_id', 'item_id'],
            how='outer'
        )
        
        if verbose:
            print(f"  병합 {i}/{len(candidate_dfs)-1} 완료 | 현재 후보: {len(merged):,}개")
    
    if verbose:
        print(f"\n  ✅ 병합 완료: {len(merged):,}개 후보")
        print(f"  유저 수: {merged['user_id'].nunique():,}명")
        print(f"  아이템 수: {merged['item_id'].nunique():,}개")
    
    # ========================================
    # Step 3: 결측치 처리 및 정규화 (벡터화)
    # ========================================
    if verbose:
        print("\n[3/4] 점수 정규화 중 (유저별 MinMax)...")
    
    # 각 Lambda별 점수 컬럼 추출
    score_cols = [f'ease_score_lambda{int(lv)}' for lv in lambda_values]
    rank_cols = [f'ease_rank_lambda{int(lv)}' for lv in lambda_values]
    
    # 결측치 처리: 유저별 최소값의 50%로 채움 (벡터화)
    for score_col in tqdm(score_cols, desc="  결측치 처리", disable=not verbose):
        user_min = merged.groupby('user_id')[score_col].transform(
            lambda x: x.min() * 0.5 if x.notna().any() else 0.0
        )
        merged[score_col] = merged[score_col].fillna(user_min).astype('float32')
    
    # Rank 결측치는 최대값+1로 채움
    for rank_col in rank_cols:
        merged[rank_col] = merged[rank_col].fillna(999).astype('int16')
    
    # 유저별 MinMax 정규화 (벡터화)
    for score_col in tqdm(score_cols, desc="  MinMax 정규화", disable=not verbose):
        user_max = merged.groupby('user_id')[score_col].transform('max')
        user_min = merged.groupby('user_id')[score_col].transform('min')
        
        norm_col = score_col.replace('ease_score', 'ease_score_norm')
        merged[norm_col] = (
            (merged[score_col] - user_min) / (user_max - user_min + 1e-8)
        ).fillna(0.0).astype('float32')
    
    # ========================================
    # Step 4: Lambda 기반 가중 평균 (벡터화)
    # ========================================
    if verbose:
        print("\n[4/4] Combined Score 계산 중...")
    
    # Lambda 값에 비례한 가중치 계산
    # λ가 클수록 보수적(인기작) → 더 높은 가중치
    # λ가 작을수록 공격적(롱테일) → 더 낮은 가중치
    total_lambda = sum(lambda_values)
    weights = [lv / total_lambda for lv in lambda_values]
    
    if verbose:
        print(f"  Lambda 가중치:")
        for lv, w in zip(lambda_values, weights):
            print(f"    λ={lv:>4} → 가중치={w:.3f}")
    
    # Combined Score = 가중 평균
    norm_cols = [f'ease_score_norm_lambda{int(lv)}' for lv in lambda_values]
    
    merged['combined_score'] = sum(
        merged[col] * w for col, w in zip(norm_cols, weights)
    ).astype('float32')
    
    # ========================================
    # Step 5: Top-K 선택 및 정리 (메모리 최적화)
    # ========================================
    if verbose:
        print(f"\n  정렬 및 Top-{k_final} 선택 중...")
    
    # 정렬 후 Top-K (벡터화)
    merged = (
        merged
        .sort_values(['user_id', 'combined_score'], ascending=[True, False])
        .groupby('user_id', as_index=False)
        .head(k_final)
        .reset_index(drop=True)
    )
    
    if verbose:
        print(f"  ✅ Top-{k_final} 선택 완료: {len(merged):,}개")
    
    # ========================================
    # Step 6: 최종 피처 준비 (독립된 3개 점수 유지!)
    # ========================================
    # 🔥 핵심: 각 Lambda별 점수를 독립된 피처로 CatBoost에 전달
    # → CatBoost가 "언제 어떤 모델을 믿을지" 스스로 학습
    
    # Lambda별 정규화된 점수 유지 (3개 독립 피처)
    ease_score_cols = []
    for lambda_val in lambda_values:
        col_name = f'ease_score_norm_lambda{int(lambda_val)}'
        ease_score_cols.append(col_name)
        if verbose:
            non_zero = (merged[col_name] > 0).sum()
            print(f"    {col_name}: {non_zero:,}개 non-zero 값")
    
    # 대표 원본 점수 (피처 엔지니어링용, 중간 lambda 값)
    middle_idx = len(lambda_values) // 2
    middle_lambda = lambda_values[middle_idx]
    merged['ease_score'] = merged[f'ease_score_lambda{int(middle_lambda)}']
    
    # 불필요한 컬럼 제거 (메모리 절약)
    # ⚠️ ease_score_norm_lambda* 3개는 반드시 유지!
    cols_to_keep = ['user_id', 'item_id', 'ease_score'] + ease_score_cols
    merged = merged[cols_to_keep].copy()
    
    if verbose:
        print("\n" + "="*70)
        print("✅ Multi-EASE 병합 완료!")
        print("="*70)
        print(f"  최종 후보: {len(merged):,}개")
        print(f"  유저당 평균: {len(merged) / merged['user_id'].nunique():.1f}개")
        print(f"  독립 피처 수: {len(ease_score_cols)}개 (각 Lambda별)")
        print(f"  피처 목록: {ease_score_cols}")
        print(f"  메모리 사용: {merged.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return merged


def save_multi_ease_candidates(
    merged_df: pd.DataFrame,
    output_path: Path,
    verbose: bool = True
) -> None:
    """Multi-EASE 병합 후보 저장 (Parquet 압축)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False, compression='snappy')
    
    if verbose:
        file_size = output_path.stat().st_size / 1024**2
        print(f"\n💾 후보 저장 완료: {output_path}")
        print(f"   파일 크기: {file_size:.1f} MB")


def load_multi_ease_candidates(
    input_path: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Multi-EASE 병합 후보 로드."""
    if not input_path.exists():
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {input_path}")
    
    df = pd.read_parquet(input_path)
    
    if verbose:
        print(f"\n📂 후보 로드 완료: {input_path}")
        print(f"   후보 수: {len(df):,}개")
        print(f"   유저 수: {df['user_id'].nunique():,}명")
    
    return df


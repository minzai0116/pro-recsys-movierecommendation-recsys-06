"""
Hybrid Merger: EASE + VAE Cross-Scoring

EASE와 VAE 후보를 병합하고, 서로의 점수를 Cross-Scoring합니다.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import gc


def merge_hybrid_candidates(
    ease_path: str = 'multi_ease_candidates/merged_candidates.parquet',
    vae_path: str = 'vae_candidates.parquet',
    output_path: str = 'hybrid_candidates.parquet',
    k_final: int = 200,
    verbose: bool = True
) -> pd.DataFrame:
    """
    EASE와 VAE 후보를 Cross-Scoring으로 병합.
    
    Args:
        ease_path: EASE 후보 파일 경로
        vae_path: VAE 후보 파일 경로
        output_path: 출력 파일 경로
        k_final: 유저당 최종 후보 수
        verbose: 상세 출력 여부
    
    Returns:
        병합된 후보 DataFrame
    """
    if verbose:
        print("\n🔄 Hybrid Merger: EASE + VAE")
    
    # 1. EASE 후보 로드
    if not Path(ease_path).exists():
        raise FileNotFoundError(f"❌ EASE 후보 파일을 찾을 수 없습니다: {ease_path}")
    
    ease_df = pd.read_parquet(ease_path)
    
    if verbose:
        print(f"   EASE: {len(ease_df):,}개 ({ease_df['user_id'].nunique():,}명)")
    
    # 2. VAE 후보 로드
    if not Path(vae_path).exists():
        raise FileNotFoundError(f"❌ VAE 후보 파일을 찾을 수 없습니다: {vae_path}\n   먼저 'python train_vae.py'를 실행하세요.")
    
    vae_df = pd.read_parquet(vae_path)
    
    if verbose:
        print(f"   VAE:  {len(vae_df):,}개 ({vae_df['user_id'].nunique():,}명)")
    
    # 3. Outer Join (Cross-Scoring)
    if verbose:
        print(f"   Cross-Scoring...")
    
    merged_df = pd.merge(
        ease_df,
        vae_df,
        on=['user_id', 'item_id'],
        how='outer'
    )
    
    if verbose:
        print(f"  ✅ 병합 완료: {len(merged_df):,}개 행")
    
    # 4. 결측치 처리 (Cross-Scoring의 핵심)
    # 4. 결측치 처리 (Cross-Scoring)
    ease_score_cols = [col for col in merged_df.columns if 'ease_score' in col]
    
    if 'vae_score' in merged_df.columns:
        user_vae_min = merged_df.groupby('user_id')['vae_score'].transform('min')
        merged_df['vae_score'] = merged_df['vae_score'].fillna(user_vae_min).fillna(0.0).astype('float32')
    
    for col in ease_score_cols:
        user_ease_min = merged_df.groupby('user_id')[col].transform('min')
        merged_df[col] = merged_df[col].fillna(user_ease_min).fillna(0.0).astype('float32')
    
    # 5. 교차 정보 피처 생성
    ease_main_col = 'ease_score_norm_lambda500' if 'ease_score_norm_lambda500' in merged_df.columns else ease_score_cols[0]
    
    merged_df['is_in_ease'] = (merged_df[ease_main_col] > 0).astype('int8')
    merged_df['is_in_vae'] = (merged_df['vae_score'] > 0).astype('int8')
    merged_df['is_hybrid'] = ((merged_df['is_in_ease'] == 1) & (merged_df['is_in_vae'] == 1)).astype('int8')
    merged_df['ease_vae_score_diff'] = np.abs(merged_df[ease_main_col] - merged_df['vae_score']).astype('float32')
    merged_df['combined_score'] = (merged_df[ease_main_col] + merged_df['vae_score']) / 2.0
    
    # 6. Top-K 필터링 (유저당)
    merged_df = (
        merged_df
        .sort_values(['user_id', 'combined_score'], ascending=[True, False])
        .groupby('user_id')
        .head(k_final)
        .reset_index(drop=True)
    )
    
    # 7. 저장
    merged_df.to_parquet(output_path, index=False)
    
    if verbose:
        print(f"\n✅ Hybrid 완료: {len(merged_df):,}개 후보 ({merged_df['user_id'].nunique():,}명)")
        print(f"   저장: {output_path}")
    
    # 메모리 정리
    gc.collect()
    
    return merged_df


if __name__ == "__main__":
    merge_hybrid_candidates()


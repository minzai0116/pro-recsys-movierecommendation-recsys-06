"""
Multi-EASE 후보 생성 스크립트

실행 순서:
1. Multi-EASE 후보 생성 (여러 lambda 값)
2. merged_candidates.parquet 저장

출력:
- multi_ease_candidates/ease_lambda{XXX}_candidates.parquet (각 lambda별)
- multi_ease_candidates/merged_candidates.parquet (병합 결과)
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import gc
import time

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from src.data.data_utils import (
    load_data,
    user_sequence_split,
    create_ground_truth,
    create_user_item_matrix
)
from src.utils.cross_score_generation import user_group_oof_cross_score
from src.mergers.multi_ease_merger import (
    merge_multi_ease_candidates,
    save_multi_ease_candidates,
    load_multi_ease_candidates
)

# ========================================
# 🔥 FINAL SUBMISSION 모드 설정
# ========================================
# True:  전체 train_df로 후보 생성 (Test 제출용)
# False: train_split (90%)로 후보 생성 (로컬 검증용)
FINAL_SUBMISSION_CANDIDATES = False  # 로컬 검증 모드
# ========================================


def main():
    """Multi-EASE 후보 생성 및 Hybrid 병합."""
    start_time = time.time()
    print("\n🚀 Multi-EASE 후보 생성 시작")
    
    # 1. 설정 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # WandB 초기화
    use_wandb = config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config['wandb'].get('project', 'recsys'),
            name=f"ease_{config['wandb'].get('run_name', 'train')}",
            config={'multi_ease': config.get('multi_ease', {})},
            tags=['ease', 'candidate_generation']
        )
    
    # 2. 데이터 로드
    train_df = load_data(config['data']['train_path'])
    
    # 3. Train/Validation 분할
    if FINAL_SUBMISSION_CANDIDATES:
        print(f"📊 모드: FINAL SUBMISSION (전체 데이터)")
        train_split = train_df
        val_split = pd.DataFrame()
        val_ground_truth = {}
        print(f"   Train: {len(train_split):,}개")
    else:
        print(f"📊 모드: 로컬 검증 (Train/Val Split)")
        train_split, val_split = user_sequence_split(
            train_df,
            val_items_per_user=config['data'].get('val_items_per_user', 10),
            seed=config.get('seed', 42)
        )
        val_ground_truth = create_ground_truth(val_split)
        print(f"   Train: {len(train_split):,}개 / Val: {len(val_split):,}개")
    
    # 4. Multi-EASE 설정
    multi_ease_config = config.get('multi_ease', {})
    multi_ease_enabled = multi_ease_config.get('enabled', False)
    
    if not multi_ease_enabled:
        raise ValueError("❌ Multi-EASE가 비활성화되어 있습니다. config.yaml에서 multi_ease.enabled=true로 설정하세요.")
    
    lambda_values = multi_ease_config.get('lambda_values', [100, 500, 2000])
    k_per_model = multi_ease_config.get('k_per_model', 100)
    k_final = multi_ease_config.get('k_final', 200)
    candidates_dir = Path(multi_ease_config.get('candidates_dir', 'multi_ease_candidates'))
    merged_path = candidates_dir / "merged_candidates.parquet"
    
    print(f"\n[1/1] Multi-EASE 후보 생성...")
    print(f"   Lambda: {lambda_values}, K/model: {k_per_model}, K/final: {k_final}")
    
    # 병합된 후보가 이미 존재하는지 확인
    if merged_path.exists():
        print(f"  📂 캐시 발견: {merged_path}")
        final_candidates = load_multi_ease_candidates(merged_path, verbose=False)
        print(f"  ✅ 로드: {len(final_candidates):,}개")
    else:
        print(f"  🔄 생성 시작...")
        
        # 각 Lambda 값에 대해 EASE 후보 생성
        candidate_dfs = []
        
        for i, lambda_val in enumerate(lambda_values):
            lambda_start = time.time()
            
            # 캐시 경로
            cache_path = candidates_dir / f"ease_lambda{int(lambda_val)}_candidates.parquet"
            
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                print(f"  [{i+1}/{len(lambda_values)}] λ={lambda_val}: 캐시 ({len(df):,}개)")
            else:
                print(f"  [{i+1}/{len(lambda_values)}] λ={lambda_val}: 학습 중...", end='', flush=True)
                
                # EASE 후보 생성
                df = user_group_oof_cross_score(
                    train_df=train_split,
                    val_ground_truth=val_ground_truth,
                    n_groups=config.get('cross_score', {}).get('n_groups', 2),
                    k=k_per_model,
                    lambda_reg=lambda_val,
                    output_dir=str(candidates_dir / f"lambda{int(lambda_val)}"),
                    verbose=False,  # 로그 최소화
                    use_wandb=False,
                    seed=config.get('seed', 42)
                )
                
                # 캐시 저장
                candidates_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False, compression='snappy')
                
                elapsed = time.time() - lambda_start
                print(f" 완료 ({len(df):,}개, {elapsed:.1f}s)")
                
                if use_wandb:
                    wandb.log({
                        f'ease_lambda{lambda_val}_candidates': len(df),
                        f'ease_lambda{lambda_val}_time': elapsed
                    })
            
            candidate_dfs.append(df)
            gc.collect()
        
        # 메모리 정리 (병합 전)
        gc.collect()
        
        # 후보 병합
        print(f"\n  🔄 병합 중...")
        merge_start = time.time()
        final_candidates = merge_multi_ease_candidates(
            candidate_dfs=candidate_dfs,
            lambda_values=lambda_values,
            k_final=k_final,
            verbose=False
        )
        merge_time = time.time() - merge_start
        
        # 병합 결과 저장
        save_multi_ease_candidates(final_candidates, merged_path, verbose=False)
        print(f"  ✅ 병합: {len(final_candidates):,}개 ({merge_time:.1f}s)")
        
        # 메모리 정리 (병합 후)
        del candidate_dfs
        gc.collect()
        
        # 독립 피처 확인
        multi_ease_cols = [col for col in final_candidates.columns if col.startswith('ease_score_norm_lambda')]
        print(f"  📊 피처: {multi_ease_cols}")
        
        # Validation Recall 계산 (로컬 검증 모드인 경우)
        if not FINAL_SUBMISSION_CANDIDATES and len(val_ground_truth) > 0:
            print(f"\n  📊 EASE Recall 계산 중...")
            from src.utils.metrics import recall_at_k
            
            # Top-K 추출 (각 lambda별)
            ease_recall = {}
            for lambda_val in lambda_values:
                col_name = f'ease_score_norm_lambda{int(lambda_val)}'
                if col_name in final_candidates.columns:
                    # Top-10 추출
                    top10_preds = {}
                    for user_id in final_candidates['user_id'].unique():
                        user_cands = final_candidates[final_candidates['user_id'] == user_id].nlargest(10, col_name)
                        top10_preds[user_id] = user_cands['item_id'].tolist()
                    
                    recall = recall_at_k(top10_preds, val_ground_truth, k=10)
                    ease_recall[lambda_val] = recall
                    print(f"     λ={lambda_val}: Recall@10 = {recall:.4f}")
            
            if use_wandb:
                for lambda_val, recall in ease_recall.items():
                    wandb.log({f'ease_lambda{lambda_val}_recall@10': recall})
        
        if use_wandb:
            wandb.log({
                'total_candidates': len(final_candidates),
                'unique_users': final_candidates['user_id'].nunique(),
                'merge_time': merge_time
            })
    
    # 5. 완료
    total_time = time.time() - start_time
    print(f"\n✅ Multi-EASE 완료! ({total_time:.1f}s)")
    print(f"   📂 {merged_path}")
    print(f"   📊 {len(final_candidates):,}개 ({final_candidates['user_id'].nunique():,}명)")
    print(f"\n💡 다음: python merge_hybrid.py")
    
    if use_wandb:
        wandb.log({'total_time': total_time})
        wandb.finish()


if __name__ == "__main__":
    main()


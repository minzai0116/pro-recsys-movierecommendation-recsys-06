"""
CatBoost Ranker 학습 스크립트

실행 순서:
1. hybrid_candidates.parquet 로드 (없으면 merged_candidates.parquet)
2. 피처 엔지니어링
3. CatBoost 학습
4. 모델 저장

입력:
- hybrid_candidates.parquet (또는 multi_ease_candidates/merged_candidates.parquet)

출력:
- models/catboost_ranker.cbm
"""
import pandas as pd
import yaml
from pathlib import Path
import time
import torch

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
    create_ground_truth
)
from src.features.feature_engineering import create_ranker_dataset
from src.models.catboost_ranker import train_catboost_ranker, predict_ranker
from src.utils.metrics import recall_at_k


def main():
    """CatBoost Ranker 학습."""
    start_time = time.time()
    print("\n🚀 CatBoost Ranker 학습 시작")
    
    # 1. 설정 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # GPU 설정 확인
    gpu_config = config.get('gpu', {})
    use_gpu = gpu_config.get('enabled', True) and torch.cuda.is_available()
    if use_gpu:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # WandB 초기화
    use_wandb = config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config['wandb'].get('project', 'recsys'),
            entity=config['wandb'].get('entity', None),
            name=f"ranker_{config['wandb'].get('run_name', 'train')}",
            config={
                'ranker': config.get('ranker', {}),
                'gpu': gpu_config,
                'feature_engineering': config.get('feature_engineering', {})
            },
            tags=['catboost', 'ranker', 'training']
        )
    
    # 2. 데이터 로드
    train_df = load_data(config['data']['train_path'])
    
    # 3. Train/Validation 분할
    train_split, val_split = user_sequence_split(
        train_df,
        val_items_per_user=config['data'].get('val_items_per_user', 10),
        seed=config.get('seed', 42)
    )
    val_ground_truth = create_ground_truth(val_split)
    
    print(f"📊 데이터 분할")
    print(f"   Train: {len(train_split):,}개 / Val: {len(val_split):,}개")
    
    # 4. 후보 로드
    print(f"\n[1/4] 후보 로드...")
    
    hybrid_path = Path("hybrid_candidates.parquet")
    ease_merged_path = Path("multi_ease_candidates/merged_candidates.parquet")
    
    if hybrid_path.exists():
        final_candidates = pd.read_parquet(hybrid_path)
        mode = "Hybrid (EASE+VAE)"
        hybrid_features = [col for col in final_candidates.columns if 'vae' in col or 'is_' in col]
        print(f"  🔥 {mode}: {len(final_candidates):,}개 ({final_candidates['user_id'].nunique():,}명)")
        if hybrid_features:
            print(f"  📊 VAE 피처: {len(hybrid_features)}개")
    elif ease_merged_path.exists():
        final_candidates = pd.read_parquet(ease_merged_path)
        mode = "Multi-EASE"
        print(f"  📂 {mode}: {len(final_candidates):,}개 ({final_candidates['user_id'].nunique():,}명)")
    else:
        raise FileNotFoundError("❌ 후보 파일 없음. train_vae.py && train_ease.py 먼저 실행하세요.")
    
    # 5. 피처 엔지니어링
    print(f"\n[2/4] 피처 엔지니어링...")
    print(f"  후보 수: {len(final_candidates):,}개")
    feat_start = time.time()
    ranker_dataset = create_ranker_dataset(
        scores_df=final_candidates,
        val_ground_truth=val_ground_truth,
        train_df=train_split,
        metadata_path=config['data'].get('metadata_path', None),
        rare_threshold=config.get('feature_engineering', {}).get('rare_threshold', 3),
        verbose=True  # 진행 상황 표시
    )
    feat_time = time.time() - feat_start
    
    print(f"\n  ✅ 피처 엔지니어링 완료!")
    print(f"     데이터셋: {len(ranker_dataset):,}개 행")
    print(f"     피처: {len(ranker_dataset.columns)}개")
    print(f"     소요 시간: {feat_time:.1f}s")
    
    # 피처 확인
    multi_ease_features = [col for col in ranker_dataset.columns if col.startswith('ease_score_norm_lambda')]
    if multi_ease_features:
        print(f"  📊 EASE 피처: {multi_ease_features}")
    
    if use_wandb:
        wandb.log({
            'num_features': len(ranker_dataset.columns),
            'num_samples': len(ranker_dataset),
            'feature_engineering_time': feat_time
        })
    
    # 6. Train/Val 분할
    # ⚠️ 중요: ranker_dataset은 이미 train_split 기반으로 생성되었으므로
    # 추가 시간 기반 분할을 하면 안 됩니다. (중복 분할 = 치팅)
    # CatBoost는 전체 ranker_dataset을 Train으로 사용하고,
    # Validation은 val_split 유저에 대한 예측으로 평가합니다.
    
    ranker_train = ranker_dataset.copy()
    ranker_val = pd.DataFrame()  # 빈 DataFrame (CatBoost 내부 검증 없음)
    
    print(f"  📊 데이터셋 구성:")
    print(f"     Train: {len(ranker_train):,}개 행 (train_split 기반)")
    print(f"     Val: val_split 유저 예측으로 평가")
    
    # 7. CatBoost 학습
    print(f"\n[3/4] CatBoost 학습...")
    ranker_config = config.get('ranker', {})
    
    train_start = time.time()
    model = train_catboost_ranker(
        train_df=ranker_train,
        val_df=None,  # ⚠️ 중요: 내부 검증 없음, val_split 유저 예측으로 평가
        val_ground_truth=val_ground_truth,
        iterations=ranker_config.get('iterations', 2000),
        early_stopping_rounds=None,  # ⚠️ 내부 검증 없으므로 early stopping 비활성화
        learning_rate=ranker_config.get('learning_rate', 0.01),
        depth=ranker_config.get('depth', 6),
        l2_leaf_reg=ranker_config.get('l2_leaf_reg', 2.0),
        loss_function=ranker_config.get('loss_function', 'YetiRankPairwise'),
        verbose=100,  # 100 iter마다 로그
        use_wandb=use_wandb
    )
    train_time = time.time() - train_start
    print(f"  ✅ 학습 완료 ({train_time:.1f}s)")
    
    # 8. Validation 성능 평가
    print(f"\n[4/4] Validation 평가...")
    
    eval_start = time.time()
    
    # Validation 유저의 후보 준비
    val_user_ids = val_split['user'].unique()
    val_candidates = final_candidates[final_candidates['user_id'].isin(val_user_ids)].copy()
    
    # 피처 엔지니어링 (Validation 후보에 대해)
    val_ranker_dataset = create_ranker_dataset(
        scores_df=val_candidates,
        val_ground_truth=val_ground_truth,
        train_df=train_split,
        metadata_path=config['data'].get('metadata_path', None),
        rare_threshold=config.get('feature_engineering', {}).get('rare_threshold', 3),
        verbose=False
    )
    
    # 예측
    predictions = predict_ranker(
        model=model,
        test_df=val_ranker_dataset,
        k=10,
        ground_truth=val_ground_truth,
        verbose=False,
        use_wandb=use_wandb
    )
    eval_time = time.time() - eval_start
    
    val_recall = recall_at_k(predictions, val_ground_truth, k=10)
    total_time = time.time() - start_time
    
    print(f"\n📊 Validation Recall@10: {val_recall:.4f}")
    print(f"⏱️  총 소요 시간: {total_time:.1f}s")
    print(f"   - 피처: {feat_time:.1f}s")
    print(f"   - 학습: {train_time:.1f}s")
    print(f"   - 평가: {eval_time:.1f}s")
    
    # 모델 저장
    model_save_path = 'models/catboost_ranker.cbm'
    Path('models').mkdir(parents=True, exist_ok=True)
    model.save_model(model_save_path)
    print(f"\n💾 모델 저장 완료: {model_save_path}")
    
    if use_wandb:
        wandb.log({
            'final_val_recall@10': val_recall,
            'train_time': train_time,
            'eval_time': eval_time,
            'total_time': total_time
        })
        wandb.finish()
    
    print(f"\n✅ 완료! 모델: {model_save_path}")


if __name__ == "__main__":
    main()


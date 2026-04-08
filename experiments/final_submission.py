"""
최종 제출용 자동화 스크립트

실행 순서:
1. Ensemble 폴더 준비
2. VAE 후보 생성 (100% 데이터)
3. EASE 후보 생성 (100% 데이터, n_groups=1)
4. Hybrid 병합
5. CatBoost 모델 로드 (train_ranker.py에서 학습한 모델)
6. Test 예측

핵심 전략:
- VAE/EASE: 100% 데이터로 후보 생성
- CatBoost: 기존 모델 사용 (시간 기반 OOF로 학습되어 일반화 성능 최적화)

출력 구조:
  ensemble/
  ├── models/
  │   └── multi_vae.pt
  ├── multi_ease_candidates/
  │   └── merged_candidates.parquet
  ├── vae_candidates.parquet
  ├── hybrid_candidates.parquet
  └── submission.csv

💡 기존 로컬 검증 파일(models/, multi_ease_candidates/ 등)은 그대로 유지됩니다.

사용법:
  python final_submission.py
"""
import subprocess
import sys
from pathlib import Path
import shutil
import pandas as pd
import yaml
import torch
import numpy as np
from tqdm import tqdm
import gc

# 기존 스크립트들의 함수 임포트
from src.data.data_utils import (
    load_data,
    create_user_item_matrix,
    create_ground_truth
)
from src.utils.cross_score_generation import user_group_oof_cross_score
from src.mergers.multi_ease_merger import merge_multi_ease_candidates, save_multi_ease_candidates
from src.mergers.hybrid_merger import merge_hybrid_candidates
from src.models.multi_vae import MultiVAE, train_vae_epoch, predict_vae
from src.features.feature_engineering import create_ranker_dataset
from src.models.catboost_ranker import train_catboost_ranker, predict_ranker
from catboost import CatBoostRanker
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def prepare_ensemble_dir():
    """Ensemble 폴더 준비."""
    print("\n📁 Ensemble 폴더 준비 중...")
    
    ensemble_dir = Path('ensemble')
    ensemble_dir.mkdir(exist_ok=True)
    
    # 하위 폴더 생성
    (ensemble_dir / 'models').mkdir(exist_ok=True)
    (ensemble_dir / 'multi_ease_candidates').mkdir(exist_ok=True)
    
    print("   ✅ 폴더 구조:")
    print("      ensemble/")
    print("      ├── models/")
    print("      ├── multi_ease_candidates/")
    print("      ├── vae_candidates.parquet")
    print("      ├── hybrid_candidates.parquet")
    print("      └── submission.csv")
    print("   ✅ 준비 완료!")


def train_vae_full(config):
    """VAE 학습 (100% 데이터)."""
    print("\n" + "="*60)
    print("[1/5] VAE 후보 생성 (100% 데이터)")
    print("="*60)
    
    vae_config = config.get('multi_vae', {})
    latent_dim = vae_config.get('latent_dim', 200)
    encoder_dims = vae_config.get('encoder_dims', [600])
    dropout = vae_config.get('dropout', 0.5)
    learning_rate = vae_config.get('learning_rate', 0.001)
    batch_size = vae_config.get('batch_size', 500)
    epochs = vae_config.get('epochs', 100)
    anneal_cap = vae_config.get('anneal_cap', 0.2)
    anneal_steps = vae_config.get('anneal_steps', 20000)
    k = vae_config.get('k', 100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드 (100%)
    train_df = load_data(config['data']['train_path'])
    print(f"   데이터: {len(train_df):,}개 (100%)")
    
    # User-Item 행렬 생성
    train_matrix, user_id_to_idx, item_id_to_idx = create_user_item_matrix(train_df)
    idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    num_users, num_items = train_matrix.shape
    
    # 데이터셋 준비
    train_data = train_matrix.toarray().astype(np.float32)
    train_tensor = torch.FloatTensor(train_data)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    del train_data
    gc.collect()
    
    # VAE 모델 학습
    p_dims = [latent_dim] + encoder_dims + [num_items]
    model = MultiVAE(p_dims, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    print(f"   학습: Latent={latent_dim}, Epochs={epochs}")
    
    update_count = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        if update_count < anneal_steps:
            anneal = min(anneal_cap, update_count / anneal_steps)
        else:
            anneal = anneal_cap
        
        avg_loss, avg_recon, avg_kl = train_vae_epoch(model, optimizer, train_loader, device, anneal)
        update_count += len(train_loader)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"     Epoch {epoch+1:3d}: Loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            Path('ensemble/models').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'ensemble/models/multi_vae.pt')
    
    print(f"   ✅ VAE 학습 완료! Best Loss: {best_loss:.4f}")
    
    # 후보 생성
    print(f"   후보 생성 중 (Top-{k})...")
    model.eval()
    all_candidates = []
    
    with torch.no_grad():
        for user_idx in tqdm(range(num_users), desc="     VAE 예측", disable=True):
            if user_idx % 5000 == 0 and user_idx > 0:
                print(f"      진행: {user_idx:,}/{num_users:,} 유저", end='\r', flush=True)
            
            user_data = train_matrix[user_idx]
            exclude_items = set(user_data.nonzero()[1])
            top_k_items, top_k_scores = predict_vae(model, user_data, device, k=k, exclude_items=exclude_items)
            
            user_id = idx_to_user_id[user_idx]
            for item_idx, score in zip(top_k_items, top_k_scores):
                item_id = idx_to_item_id[item_idx]
                all_candidates.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'vae_score': score
                })
    
    print(f"      완료: {num_users:,}/{num_users:,} 유저")
    
    candidates_df = pd.DataFrame(all_candidates)
    candidates_df['vae_score_norm'] = candidates_df.groupby('user_id')['vae_score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    ).astype('float32')
    candidates_df = candidates_df[['user_id', 'item_id', 'vae_score_norm']].rename(columns={'vae_score_norm': 'vae_score'})
    
    output_path = 'ensemble/vae_candidates.parquet'
    candidates_df.to_parquet(output_path, index=False)
    print(f"   ✅ VAE 완료: {len(candidates_df):,}개 후보")


def generate_ease_full(config):
    """EASE 후보 생성 (100% 데이터, Full Fit)."""
    print("\n" + "="*60)
    print("[2/5] EASE 후보 생성 (100% 데이터, Full Fit)")
    print("="*60)
    
    from src.models.ease import EASE
    
    multi_ease_config = config.get('multi_ease', {})
    lambda_values = multi_ease_config.get('lambda_values', [100, 500, 2000])
    k_per_model = multi_ease_config.get('k_per_model', 100)
    k_final = multi_ease_config.get('k_final', 200)
    candidates_dir = Path('ensemble/multi_ease_candidates')
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드 (100%)
    train_df = load_data(config['data']['train_path'])
    print(f"   데이터: {len(train_df):,}개 (100%)")
    print(f"   Lambda: {lambda_values}")
    
    # User-Item 행렬 생성
    print("\n   User-Item 행렬 생성 중...")
    train_matrix, user_id_to_idx, item_id_to_idx = create_user_item_matrix(train_df)
    idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    print(f"   ✅ 행렬 크기: {train_matrix.shape[0]:,} 유저 × {train_matrix.shape[1]:,} 아이템")
    
    candidate_dfs = []
    
    for i, lambda_val in enumerate(lambda_values):
        print(f"\n   [{i+1}/{len(lambda_values)}] λ={lambda_val}...")
        
        cache_path = candidates_dir / f"ease_lambda{int(lambda_val)}_candidates.parquet"
        
        # EASE 학습 (전체 데이터)
        print(f"      EASE 학습 중...", end='', flush=True)
        ease_model = EASE(lambda_reg=lambda_val)
        ease_model.fit(train_matrix, item_id_to_idx, verbose=False)
        print(" 완료")
        
        # 점수 생성 (배치 처리)
        print(f"      점수 생성 중...")
        df = ease_model.predict_batch_with_scores(
            user_item_matrix=train_matrix,
            user_id_to_idx=user_id_to_idx,
            k=k_per_model,
            verbose=False
        )
        
        # merge_multi_ease_candidates가 ease_score, ease_rank 컬럼을 기대함
        # 컬럼명 변경 없이 그대로 사용
        
        df.to_parquet(cache_path, index=False, compression='snappy')
        print(f"      ✅ 완료: {len(df):,}개 후보")
        
        candidate_dfs.append(df)
        
        del ease_model
        gc.collect()
    
    # 병합
    print(f"\n   병합 중...")
    final_candidates = merge_multi_ease_candidates(
        candidate_dfs=candidate_dfs,
        lambda_values=lambda_values,
        k_final=k_final,
        verbose=True
    )
    
    merged_path = candidates_dir / "merged_candidates.parquet"
    save_multi_ease_candidates(final_candidates, merged_path, verbose=True)
    
    del candidate_dfs
    gc.collect()
    
    print(f"   ✅ EASE 완료: {len(final_candidates):,}개 후보 (Full Fit)")


def merge_hybrid_full():
    """Hybrid 병합 (EASE + VAE)."""
    print("\n" + "="*60)
    print("[3/5] Hybrid 병합 (EASE + VAE)")
    print("="*60)
    
    ease_path = Path('ensemble/multi_ease_candidates/merged_candidates.parquet')
    vae_path = Path('ensemble/vae_candidates.parquet')
    output_path = Path('ensemble/hybrid_candidates.parquet')
    
    if not ease_path.exists():
        raise FileNotFoundError(f"❌ EASE 후보를 찾을 수 없습니다: {ease_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"❌ VAE 후보를 찾을 수 없습니다: {vae_path}")
    
    print(f"   EASE 로드 중...")
    ease_df = pd.read_parquet(ease_path)
    print(f"   VAE 로드 중...")
    vae_df = pd.read_parquet(vae_path)
    
    print(f"   병합 중...")
    # Outer join
    hybrid_df = ease_df.merge(vae_df, on=['user_id', 'item_id'], how='outer')
    
    # Cross-information 피처 (결측값 처리 전)
    hybrid_df['is_in_ease'] = (~hybrid_df['combined_score'].isna()).astype('int8') if 'combined_score' in hybrid_df.columns else 0
    hybrid_df['is_in_vae'] = (~hybrid_df['vae_score'].isna()).astype('int8')
    hybrid_df['is_hybrid'] = (hybrid_df['is_in_ease'] & hybrid_df['is_in_vae']).astype('int8')
    
    # 결측값 처리
    if 'combined_score' in hybrid_df.columns:
        user_min_ease = hybrid_df.groupby('user_id')['combined_score'].transform('min')
        hybrid_df['combined_score'] = hybrid_df['combined_score'].fillna(user_min_ease)
    
    user_min_vae = hybrid_df.groupby('user_id')['vae_score'].transform('min')
    hybrid_df['vae_score'] = hybrid_df['vae_score'].fillna(user_min_vae)
    
    # Cross-score difference
    if 'combined_score' in hybrid_df.columns:
        hybrid_df['ease_vae_score_diff'] = (hybrid_df['combined_score'] - hybrid_df['vae_score']).abs().astype('float32')
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid_df.to_parquet(output_path, index=False)
    
    print(f"   ✅ Hybrid 완료: {len(hybrid_df):,}개 후보")


def load_catboost_model(config):
    """CatBoost 모델 로드 (재학습 안 함)."""
    print("\n" + "="*60)
    print("[4/5] CatBoost 모델 로드 (재학습 안 함)")
    print("="*60)
    
    model_path = 'models/catboost_ranker.cbm'
    if not Path(model_path).exists():
        raise FileNotFoundError(f"❌ CatBoost 모델을 찾을 수 없습니다: {model_path}")
    
    print(f"   ✅ 모델 로드: {model_path}")
    print("   💡 train_ranker.py에서 학습한 모델을 사용합니다.")
    print("   💡 이 모델은 시간 기반 OOF로 학습되어 일반화 성능이 최적화되었습니다.")


def predict_full(config):
    """Test 예측."""
    print("\n" + "="*60)
    print("[5/5] Test 예측")
    print("="*60)
    
    # 1. 데이터 로드
    train_df = load_data(config['data']['train_path'])
    
    # Test 유저 추출
    sample_submission_path = config['data'].get('sample_submission_path', '../RecSys_RRS_Framework/data/eval/sample_submission.csv')
    if Path(sample_submission_path).exists():
        sample_submission = pd.read_csv(sample_submission_path)
        test_user_ids = sample_submission['user'].unique()
        print(f"   Test 유저: {len(test_user_ids):,}명")
    else:
        raise FileNotFoundError(f"❌ sample_submission.csv를 찾을 수 없습니다: {sample_submission_path}")
    
    # 2. 후보 로드
    hybrid_path = Path("ensemble/hybrid_candidates.parquet")
    if not hybrid_path.exists():
        raise FileNotFoundError(f"❌ hybrid_candidates.parquet를 찾을 수 없습니다: {hybrid_path}")
    
    final_candidates = pd.read_parquet(hybrid_path)
    
    # Test 유저만 필터링
    final_candidates = final_candidates[final_candidates['user_id'].isin(test_user_ids)].reset_index(drop=True)
    print(f"   후보: {len(final_candidates):,}개 (Test 유저만)")
    
    # 3. 피처 엔지니어링
    print("   피처 엔지니어링 중...")
    ranker_dataset = create_ranker_dataset(
        scores_df=final_candidates,
        val_ground_truth={},  # Test용
        train_df=train_df,
        metadata_path=config['data'].get('metadata_path', None),
        rare_threshold=config.get('feature_engineering', {}).get('rare_threshold', 3),
        verbose=False
    )
    
    if 'label' in ranker_dataset.columns:
        ranker_dataset = ranker_dataset.drop(columns=['label'])
    
    print(f"   ✅ 피처: {len(ranker_dataset):,}개 행")
    
    # 4. CatBoost 예측
    print("   CatBoost 예측 중...")
    model_path = 'models/catboost_ranker.cbm'
    model = CatBoostRanker()
    model.load_model(model_path)
    
    predictions = predict_ranker(
        model=model,
        test_df=ranker_dataset,
        k=10,
        ground_truth=None,
        verbose=False,
        use_wandb=False
    )
    
    # 5. 제출 파일 생성
    print("   submission.csv 생성 중...")
    
    # 형식 확인
    user_counts = sample_submission['user'].value_counts()
    submission_list = []
    
    if user_counts.iloc[0] > 1:
        # 각 유저당 여러 줄
        for user_id in sorted(predictions.keys()):
            items = predictions[user_id]
            for item in items:
                submission_list.append({'user': user_id, 'item': item})
    else:
        # 각 유저당 1줄
        for user_id in sorted(predictions.keys()):
            items = predictions[user_id]
            items_str = ' '.join(map(str, items))
            submission_list.append({'user': user_id, 'item': items_str})
    
    submission_df = pd.DataFrame(submission_list)
    submission_path = 'ensemble/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"   ✅ 제출 파일 완료: {submission_path}")
    print(f"   ✅ 유저: {submission_df['user'].nunique():,}명")
    print(f"   ✅ 총 줄 수: {len(submission_df):,}줄")


def main():
    """최종 제출 자동화 메인 함수."""
    print("\n" + "="*60)
    print("🚀 최종 제출 자동화 시작 (100% 데이터)")
    print("="*60)
    print("\n💡 전략:")
    print("   - VAE: 100% 데이터로 학습")
    print("   - EASE: 100% 데이터로 후보 생성 (n_groups=1)")
    print("   - CatBoost: 기존 모델 사용 (train_ranker.py에서 학습)")
    print("   - 모든 파일은 ensemble/ 폴더에 저장")
    print("   - 기존 로컬 검증 파일은 유지")
    print("   - 예상 소요 시간: 40-50분\n")
    
    # 사용자 확인
    response = input("계속하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        return
    
    # 설정 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # 1. Ensemble 폴더 준비
        prepare_ensemble_dir()
        
        # 2. VAE 후보 생성 (100% 데이터)
        train_vae_full(config)
        
        # 3. EASE 후보 생성 (100% 데이터)
        generate_ease_full(config)
        
        # 4. Hybrid 병합
        merge_hybrid_full()
        
        # 5. CatBoost 모델 로드
        load_catboost_model(config)
        
        # 6. Test 예측
        predict_full(config)
        
        print("\n" + "="*60)
        print("✅ 최종 제출 완료!")
        print("="*60)
        print("   📂 출력: ensemble/submission.csv")
        print("   💡 이제 제출하세요!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


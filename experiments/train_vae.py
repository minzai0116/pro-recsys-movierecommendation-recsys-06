"""
Multi-VAE 학습 및 후보 생성 스크립트

실행 순서:
1. 데이터 로드
2. User-Item 행렬 생성
3. Multi-VAE 학습
4. Top-100 후보 추출
5. vae_candidates.parquet 저장
"""
import pandas as pd
import numpy as np
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda.amp')
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import gc

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from src.data.data_utils import load_data, create_user_item_matrix, user_sequence_split
from src.models.multi_vae import MultiVAE, train_vae_epoch, predict_vae

# ========================================
# 🔥 FINAL SUBMISSION 모드 설정
# ========================================
# True:  전체 train_df로 VAE 학습 및 후보 생성 (Test 제출용)
# False: train_split (90%)로 학습 (로컬 검증용)
FINAL_SUBMISSION_CANDIDATES = False  # 로컬 검증 모드
# ========================================


def main():
    """Multi-VAE 학습 및 후보 생성."""
    print("\n🚀 Multi-VAE 학습 시작")
    
    # 1. 설정 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # VAE 설정 (없으면 기본값)
    vae_config = config.get('multi_vae', {
        'latent_dim': 200,
        'encoder_dims': [600],
        'dropout': 0.5,
        'learning_rate': 0.001,
        'batch_size': 500,
        'epochs': 100,
        'anneal_cap': 0.2,
        'anneal_steps': 20000,
        'k': 100
    })
    
    latent_dim = vae_config.get('latent_dim', 200)
    encoder_dims = vae_config.get('encoder_dims', [600])
    dropout = vae_config.get('dropout', 0.5)
    learning_rate = vae_config.get('learning_rate', 0.001)
    batch_size = vae_config.get('batch_size', 1024)
    num_workers = vae_config.get('num_workers', 4)
    pin_memory = vae_config.get('pin_memory', True)
    epochs = vae_config.get('epochs', 100)
    anneal_cap = vae_config.get('anneal_cap', 0.2)
    anneal_steps = vae_config.get('anneal_steps', 20000)
    k = vae_config.get('k', 100)
    log_interval = vae_config.get('log_interval', 20)
    
    # GPU 설정
    gpu_config = config.get('gpu', {})
    use_gpu = gpu_config.get('enabled', True) and torch.cuda.is_available()
    mixed_precision = gpu_config.get('mixed_precision', True) and use_gpu
    memory_limit = gpu_config.get('memory_limit_gb', 25)
    
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    if use_gpu:
        torch.cuda.set_per_process_memory_fraction(memory_limit / torch.cuda.get_device_properties(0).total_memory * 1024**3)
        if gpu_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
        print(f"   GPU: {torch.cuda.get_device_name(0)}, 메모리: {memory_limit}GB")
    
    # WandB 초기화
    use_wandb = config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config['wandb'].get('project', 'recsys'),
            name=f"vae_{config['wandb'].get('run_name', 'train')}",
            config={'vae': vae_config, 'gpu': gpu_config},
            tags=['vae', 'candidate_generation']
        )
    
    # 2. 데이터 로드 및 Train/Validation 분할
    train_df = load_data(config['data']['train_path'])
    
    # FINAL_SUBMISSION_CANDIDATES에 따라 분할 여부 결정
    if FINAL_SUBMISSION_CANDIDATES:
        print(f"📊 모드: FINAL SUBMISSION (전체 데이터)")
        train_split = train_df
        val_split = pd.DataFrame()
    else:
        print(f"📊 모드: 로컬 검증 (Train/Val Split)")
        train_split, val_split = user_sequence_split(
            train_df,
            val_items_per_user=config.get('val_items_per_user', 10),
            seed=config.get('seed', 42)
        )
    
    print(f"   Train: {len(train_split):,}개")
    
    # 3. User-Item 행렬 생성
    train_matrix, user_id_to_idx, item_id_to_idx = create_user_item_matrix(train_split)
    idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    
    num_users, num_items = train_matrix.shape
    print(f"   Matrix: {num_users:,} users × {num_items:,} items")
    
    # 4. 데이터셋 준비
    train_data = train_matrix.toarray().astype(np.float32)
    train_tensor = torch.FloatTensor(train_data)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # 메모리 정리
    del train_data
    gc.collect()
    if use_gpu:
        torch.cuda.empty_cache()
    
    # 5. Multi-VAE 모델 학습
    p_dims = [latent_dim] + encoder_dims + [num_items]
    model = MultiVAE(p_dims, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scaler = GradScaler() if mixed_precision else None
    
    print(f"\n🔥 VAE 학습: Latent={latent_dim}, Epochs={epochs}, Device={device}")
    if mixed_precision:
        print(f"   Mixed Precision: Enabled (FP16)")
    
    Path('models').mkdir(exist_ok=True)
    update_count = 0
    best_loss = float('inf')
    best_epoch = 0
    log_epochs = [1, 20, 40, 60, 80, 100]  # 로그 출력 epoch
    
    for epoch in range(epochs):
        # KL Annealing
        if update_count < anneal_steps:
            anneal = min(anneal_cap, update_count / anneal_steps)
        else:
            anneal = anneal_cap
        
        # Train
        avg_loss, avg_recon, avg_kl = train_vae_epoch(
            model, optimizer, train_loader, device, anneal, scaler
        )
        update_count += len(train_loader)
        
        # 로그 출력 (로그 라인 제한)
        if (epoch + 1) in log_epochs or (epoch + 1) % log_interval == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, KL={avg_kl:.4f})")
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'recon_loss': avg_recon,
                    'kl_loss': avg_kl,
                    'anneal': anneal
                })
        
        # Best model 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'models/multi_vae.pt')
    
    print(f"\n  📊 VAE 학습 완료!")
    print(f"     Best Loss: {best_loss:.4f} (Epoch {best_epoch})")
    print(f"     Final Loss: {avg_loss:.4f}")
    print(f"     - Recon Loss: {avg_recon:.4f}")
    print(f"     - KL Loss: {avg_kl:.4f}")
    if use_wandb:
        wandb.log({'best_loss': best_loss})
    
    # 6. 후보 추출
    print(f"\n🎯 VAE 후보 생성 중 (Top-{k})...")
    
    model.eval()
    
    all_candidates = []
    
    # 유저별 예측 (배치 단위)
    with torch.no_grad():
        pbar = tqdm(range(num_users), desc="  VAE 예측", ncols=80, disable=True)
        for user_idx in pbar:
            if user_idx % 5000 == 0 and user_idx > 0:  # 5000명마다 로그
                print(f"  VAE 예측: {user_idx}/{num_users} ({user_idx/num_users*100:.1f}%)", end='\r')
            
            user_data = train_matrix[user_idx]
            exclude_items = set(user_data.nonzero()[1])
            
            # 예측
            top_k_items, top_k_scores = predict_vae(model, user_data, device, k=k, exclude_items=exclude_items)
            
            # 결과 저장
            user_id = idx_to_user_id[user_idx]
            for item_idx, score in zip(top_k_items, top_k_scores):
                item_id = idx_to_item_id[item_idx]
                all_candidates.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'vae_score': score
                })
        
        if use_gpu:
            torch.cuda.empty_cache()
    
    print(f"\n  ✅ VAE 예측 완료!")
    
    # DataFrame 변환
    candidates_df = pd.DataFrame(all_candidates)
    
    # MinMax Scaling (유저별)
    print(f"\n  정규화 중 (User-wise MinMax Scaling)...")
    candidates_df['vae_score_norm'] = candidates_df.groupby('user_id')['vae_score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    ).astype('float32')
    
    # 최종 점수
    candidates_df = candidates_df[['user_id', 'item_id', 'vae_score_norm']].rename(columns={'vae_score_norm': 'vae_score'})
    
    # 7. 저장
    output_path = 'vae_candidates.parquet'
    candidates_df.to_parquet(output_path, index=False)
    
    print(f"\n✅ VAE 완료: {len(candidates_df):,}개 후보 ({candidates_df['user_id'].nunique():,}명)")
    print(f"   저장: {output_path}")
    
    if use_wandb:
        wandb.log({
            'total_candidates': len(candidates_df),
            'unique_users': candidates_df['user_id'].nunique(),
            'candidates_per_user': len(candidates_df) / candidates_df['user_id'].nunique()
        })
        wandb.finish()


if __name__ == "__main__":
    main()


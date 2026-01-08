"""하이퍼파라미터 최적화 (베이지안 최적화)."""
import optuna
from typing import Dict, Any, Optional
import yaml
import numpy as np
from scipy import sparse
import wandb

from src.data_utils import (
    load_data,
    create_user_item_matrix,
    user_sequence_split,
    create_ground_truth
)
from src.ease import EASE
from src.metrics import recall_at_k


def objective(trial, train_matrix, val_ground_truth, user_id_to_idx, item_id_to_idx, 
              precomputed_G: np.ndarray, precomputed_seen: Dict[int, set],
              lambda_min=100.0, lambda_max=500.0, use_log_scale=False, log_wandb=False):
    """
    Optuna objective 함수 (최적화된 버전).
    
    Args:
        trial: Optuna trial 객체
        train_matrix: 학습용 user-item 행렬
        val_ground_truth: Validation ground truth
        user_id_to_idx: 사용자 ID -> 인덱스 매핑
        item_id_to_idx: 아이템 ID -> 인덱스 매핑
        precomputed_G: 미리 계산된 Gram 행렬 (X^T X) - 중복 계산 방지
        precomputed_seen: 미리 계산된 사용자별 본 아이템 딕셔너리
        lambda_min: Lambda 최소값
        lambda_max: Lambda 최대값
        use_log_scale: 로그 스케일 사용 여부
        log_wandb: WandB 로깅 여부
    
    Returns:
        Recall@10 점수 (최대화)
    """
    # Lambda 범위: lambda_min ~ lambda_max
    if use_log_scale:
        lambda_reg = trial.suggest_float('lambda_reg', lambda_min, lambda_max, log=True)
    else:
        lambda_reg = trial.suggest_float('lambda_reg', lambda_min, lambda_max, step=10.0)
    
    # 모델 학습 (precomputed_G 재사용으로 X^T X 계산 생략)
    model = EASE(lambda_reg=lambda_reg)
    model.fit(train_matrix, item_id_to_idx, precomputed_G=precomputed_G, verbose=False)
    
    # Validation 예측
    val_user_ids = [uid for uid in val_ground_truth.keys() if uid in user_id_to_idx]
    val_user_id_to_idx = {uid: user_id_to_idx[uid] for uid in val_user_ids if uid in user_id_to_idx}
    
    predictions = {}
    
    if len(val_user_id_to_idx) > 0:
        # 벡터화된 배치 예측 사용
        all_predictions = model.predict_batch_vectorized(
            train_matrix,
            val_user_id_to_idx,
            k=10,
            verbose=False
        )
        
        # 본 아이템 제외 처리 (precomputed_seen 재사용)
        for user_id in val_user_ids:
            if user_id not in all_predictions:
                continue
            seen_items = precomputed_seen.get(user_id, set())
            filtered = [item for item in all_predictions[user_id] if item not in seen_items]
            predictions[user_id] = filtered[:10]
    
    # Recall 계산 (전체 validation 사용자)
    recall = recall_at_k(predictions, val_ground_truth, k=10)
    
    # WandB 로깅 (실시간 모니터링)
    if log_wandb and wandb.run is not None:
        wandb.log({
            "optimization/trial_recall@10": recall,
            "optimization/trial_lambda": lambda_reg,
            "optimization/trial_number": trial.number
        })
    
    return recall


def optimize_lambda(cfg: Dict[str, Any], 
                   n_trials: int = 20,
                   timeout: int = None) -> float:
    """
    Lambda 정규화 파라미터 최적화.
    
    Args:
        cfg: 설정 딕셔너리
        n_trials: 시도 횟수
        timeout: 최대 시간 (초)
    
    Returns:
        최적 Lambda 값
    """
    print("=" * 60)
    print("Lambda 하이퍼파라미터 최적화 시작")
    print("=" * 60)
    print(f"Lambda 범위: 100 ~ 500")
    print(f"시도 횟수: {n_trials}")
    if timeout:
        print(f"최대 시간: {timeout}초")
    
    # 데이터 로드
    data_path = cfg['data']['train_path']
    print(f"\n[1/3] 데이터 로드: {data_path}")
    df = load_data(data_path)
    
    # Train/Validation 분할 (메인과 동일한 방식: user_sequence_split)
    val_items_per_user = cfg.get('val_items_per_user', 10)
    seed = cfg.get('seed', 42)
    print(f"\n[2/3] Train/Validation 분할 (User-sequence split)")
    print(f"  → 사용자당 Validation 아이템 수: {val_items_per_user}")
    print(f"  → 마지막 1개 + 중간 dropout {val_items_per_user - 1}개")
    train_df, val_df = user_sequence_split(df, val_items_per_user=val_items_per_user, seed=seed)
    
    # 행렬 생성
    train_matrix, user_id_to_idx, item_id_to_idx = create_user_item_matrix(train_df)
    val_ground_truth = create_ground_truth(val_df)
    
    print(f"\n[3/4] 사전 계산 (최적화 핵심)")
    print(f"  Train 행렬: {train_matrix.shape}")
    print(f"  Validation 사용자: {len(val_ground_truth):,}")
    
    # [최적화 핵심 1] Gram 행렬 미리 계산 (X^T X는 lambda와 무관하므로 한 번만 계산)
    print("  → Gram 행렬 (X^T X) 사전 계산 중...")
    X = train_matrix.astype(np.float32)
    G_base = (X.T @ X).toarray()
    print(f"    ✅ 완료! (크기: {G_base.shape}, 메모리: {G_base.nbytes / 1024 / 1024:.2f} MB)")
    
    # [최적화 핵심 2] 사용자별 본 아이템 사전 계산 (lambda와 무관하므로 한 번만 계산)
    # Critical Fix: O(N^2) -> O(N) 최적화 (groupby 사용)
    print("  → 사용자별 본 아이템 사전 캐싱 중 (Groupby 최적화)...")
    val_user_ids = [uid for uid in val_ground_truth.keys() if uid in user_id_to_idx]
    # 한 번의 groupby로 모든 유저의 본 아이템 계산 (약 1초 내 완료)
    all_seen_items = train_df.groupby('user')['item'].apply(set).to_dict()
    # 필요한 유저만 필터링
    seen_items_dict = {uid: all_seen_items[uid] for uid in val_user_ids if uid in all_seen_items}
    print(f"    ✅ 완료! (총 {len(seen_items_dict):,}명)")
    
    # Lambda 범위 가져오기
    opt_cfg = cfg.get('optimization', {})
    lambda_min = opt_cfg.get('lambda_min', 100.0)
    lambda_max = opt_cfg.get('lambda_max', 500.0)
    use_log_scale = opt_cfg.get('use_log_scale', False)
    
    print(f"\n[4/4] 최적화 시작...")
    print(f"  Lambda 탐색 범위: {lambda_min} ~ {lambda_max}")
    print(f"  탐색 스케일: {'로그 스케일' if use_log_scale else '선형 스케일'}")
    print(f"  시도 횟수: {n_trials}")
    print(f"  ⚡ 최적화: Gram 행렬 재사용으로 약 20배 빠른 탐색 예상")
    
    # WandB 로깅 설정
    log_wandb = cfg.get('wandb', {}).get('enabled', False)
    
    # Optuna study 생성
    study = optuna.create_study(
        direction='maximize',  # Recall@10 최대화
        study_name='ease_lambda_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)  # TPE (Tree-structured Parzen Estimator)
    )
    
    # 최적화 실행 (사전 계산된 값들 전달)
    study.optimize(
        lambda trial: objective(
            trial, train_matrix, val_ground_truth, 
            user_id_to_idx, item_id_to_idx,
            precomputed_G=G_base,  # 사전 계산된 Gram 행렬
            precomputed_seen=seen_items_dict,  # 사전 계산된 본 아이템
            lambda_min=lambda_min, 
            lambda_max=lambda_max,
            use_log_scale=use_log_scale,
            log_wandb=log_wandb
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("최적화 완료!")
    print("=" * 60)
    print(f"최적 Lambda: {study.best_params['lambda_reg']:.2f}")
    print(f"최고 Recall@10: {study.best_value:.4f}")
    print(f"\n시도 횟수: {len(study.trials)}")
    print(f"평균 Recall@10: {np.mean([t.value for t in study.trials if t.value is not None]):.4f}")
    
    return study.best_params['lambda_reg']


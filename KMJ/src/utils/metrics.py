"""평가 지표 계산."""
from typing import List, Dict, Set
import numpy as np
from tqdm import tqdm


def recall_at_k(predictions: Dict[int, List[int]], 
                ground_truth: Dict[int, Set[int]], 
                k: int = 10) -> float:
    """
    일반 Recall@K 계산.
    
    일반 Recall@K: 분모는 |Ground Truth|
    (Normalized Recall@K와 달리 min(K, |Ground Truth|)를 사용하지 않음)
    
    Args:
        predictions: {user_id: [item1, item2, ...]} - 예측된 Top-K 아이템
        ground_truth: {user_id: {item1, item2, ...}} - 실제 정답 아이템 집합
        k: Top-K
    
    Returns:
        Recall@K 점수
    """
    total_recall = 0.0
    num_users = 0
    
    for user_id, pred_items in predictions.items():
        if user_id not in ground_truth:
            continue
        
        gt_items = ground_truth[user_id]
        if len(gt_items) == 0:
            continue
        
        # Top-K 예측
        top_k_pred = set(pred_items[:k])
        
        # 교집합 계산
        intersection = top_k_pred & gt_items
        
        # 일반 Recall@K: |Ground Truth|를 분모로 사용
        denominator = len(gt_items)
        if denominator > 0:
            recall = len(intersection) / denominator
            total_recall += recall
            num_users += 1
    
    return total_recall / num_users if num_users > 0 else 0.0


def calculate_recall_during_training(model, 
                                    user_item_matrix, 
                                    val_ground_truth: Dict[int, Set[int]],
                                    k: int = 10,
                                    batch_size: int = 1000,
                                    verbose: bool = True) -> float:
    """
    학습 중간에 Recall@K 계산 (향후 사용 가능, 현재는 미사용).
    
    주의: 현재 main.py에서는 사용하지 않지만, 향후 step-wise validation 등에 활용 가능.
    
    Args:
        model: EASE 모델 (predict_batch 메서드 보유)
        user_item_matrix: scipy.sparse.csr_matrix (전체 user-item 행렬)
        val_ground_truth: {user_id: {item1, item2, ...}}
        k: Top-K
        batch_size: 배치 크기 (메모리 절약)
        verbose: 진행 상황 출력
    
    Returns:
        Recall@K 점수
    """
    predictions = {}
    user_ids = list(val_ground_truth.keys())
    
    if verbose:
        iterator = tqdm(range(0, len(user_ids), batch_size), desc="Calculating Recall@K")
    else:
        iterator = range(0, len(user_ids), batch_size)
    
    for i in iterator:
        batch_users = user_ids[i:i+batch_size]
        
        # 배치 예측
        batch_predictions = model.predict_batch(user_item_matrix, batch_users, k=k)
        
        for user_id, pred_items in zip(batch_users, batch_predictions):
            predictions[user_id] = pred_items
    
    # Recall 계산
    recall = recall_at_k(predictions, val_ground_truth, k=k)
    
    return recall


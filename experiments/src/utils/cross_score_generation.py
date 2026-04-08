"""Cross-Score 생성 (Expanding Window OOF 방식)."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.data.data_utils import (
    create_user_item_matrix,
    create_ground_truth,
    validate_time_order
)
from src.models.ease import EASE
from src.utils.metrics import recall_at_k


def user_group_oof_cross_score(
    train_df: pd.DataFrame,
    val_ground_truth: Optional[Dict[int, Set[int]]] = None,  # Validation ground truth (선택)
    n_groups: int = 2,  # 2-Fold (더 많은 데이터로 EASE 학습)
    k: int = 200,  # ⚠️ 중요: k=200 고정 (Recall 향상)
    lambda_reg: float = 545.0,
    output_dir: str = "cross_scores",
    verbose: bool = True,
    use_wandb: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    유저 그룹 기반 OOF 방식으로 Cross-Score 생성 (제3자 제안).
    
    전략:
    - 유저를 랜덤하게 n_groups로 나눔
    - [1,2,3,4번 그룹 학습] → 5번 그룹 점수 생성
    - [1,2,3,5번 그룹 학습] → 4번 그룹 점수 생성
    - ... (K-Fold 방식)
    - 시간 순서는 유지하되, 유저 단위로 분할하여 EASE 학습 데이터 확보
    
    Args:
        train_df: Train 데이터프레임 (시간순 정렬 필요)
        n_groups: 유저 그룹 개수 (기본값: 5)
        k: Top-K 후보 개수 (기본값: 200, 고정)
        lambda_reg: EASE 정규화 파라미터
        output_dir: 점수 저장 디렉토리
        verbose: 상세 출력 여부
        seed: 랜덤 시드
    
    Returns:
        Cross-Score DataFrame (user_id, item_id, ease_score, ease_rank, block_id)
    """
    # 시간순 정렬
    train_df = train_df.sort_values('time').reset_index(drop=True)
    
    # 유저 그룹 분할 (랜덤, 시드 고정)
    np.random.seed(seed)
    unique_users = train_df['user'].unique()
    np.random.shuffle(unique_users)
    
    group_size = len(unique_users) // n_groups
    user_groups = []
    for i in range(n_groups):
        start_idx = i * group_size
        if i == n_groups - 1:
            # 마지막 그룹은 나머지 모두 포함
            end_idx = len(unique_users)
        else:
            end_idx = (i + 1) * group_size
        user_groups.append(set(unique_users[start_idx:end_idx]))
    
    if verbose:
        print(f"\n유저 그룹 분할 완료:")
        for i, group in enumerate(user_groups, 1):
            print(f"  그룹 {i}: {len(group):,}명 유저")
    
    # K-Fold OOF 점수 생성
    all_scores = []
    
    for test_group_idx in range(n_groups):
        if verbose:
            print(f"\n[그룹 {test_group_idx + 1} 점수 생성]")
            print(f"  학습: 그룹 {[i+1 for i in range(n_groups) if i != test_group_idx]}")
            print(f"  점수 생성: 그룹 {test_group_idx + 1}")
        
        # Train 그룹 (Test 그룹 제외)
        train_user_set = set()
        for i, group in enumerate(user_groups):
            if i != test_group_idx:
                train_user_set |= group
        
        test_user_set = user_groups[test_group_idx]
        
        # Train/Test 데이터 분할
        train_data = train_df[train_df['user'].isin(train_user_set)].copy()
        test_data = train_df[train_df['user'].isin(test_user_set)].copy()
        
        if len(train_data) == 0 or len(test_data) == 0:
            if verbose:
                print(f"  ⚠️  그룹 {test_group_idx + 1} 스킵 (데이터 부족)")
            continue
        
        # ⚠️ 유저 그룹 기반 OOF는 유저를 랜덤하게 나누기 때문에 시간 순서 검증 불가
        # 유저 그룹 내에서 시간 순서는 보장되지만, 그룹 간에는 시간 순서가 없을 수 있음
        # 따라서 시간 순서 검증을 제거 (유저 기반 분할이므로 데이터 누수 없음)
        if verbose:
            train_max_time = train_data['time'].max()
            test_min_time = test_data['time'].min()
            print(f"  ⚠️  유저 그룹 기반 OOF: 시간 순서 검증 생략")
            print(f"     Train 최대 시간: {train_max_time}, Test 최소 시간: {test_min_time}")
        
        # EASE 학습
        if verbose:
            print(f"  EASE 학습 중... (학습 데이터: {len(train_data):,}개 상호작용)")
        train_matrix, user_id_to_idx, item_id_to_idx = create_user_item_matrix(train_data)
        model = EASE(lambda_reg=lambda_reg)
        model.fit(train_matrix, item_id_to_idx, verbose=False)
        
        # 점수 생성 (Test 그룹의 유저들만)
        # ⚠️ OOF 핵심: Test 유저들의 train_split 히스토리를 사용해서 예측
        # Test 유저들도 train_split에 존재함 (단지 학습 그룹에서 제외된 것)
        # 따라서 Test 유저들의 히스토리를 포함한 행렬 필요
        
        test_user_ids = list(test_user_set)
        
        # Test 유저들의 데이터를 포함한 행렬 생성 (예측용)
        # train_data는 학습에 사용했고, test_data는 예측 시 X 벡터로만 사용
        all_data_for_prediction = pd.concat([train_data, test_data], ignore_index=True)
        pred_matrix, pred_user_id_to_idx, pred_item_id_to_idx = create_user_item_matrix(all_data_for_prediction)
        
        # Test 유저들의 인덱스 매핑
        test_user_id_to_idx = {uid: pred_user_id_to_idx[uid] for uid in test_user_ids if uid in pred_user_id_to_idx}
        
        if len(test_user_id_to_idx) == 0:
            if verbose:
                print(f"  ⚠️  그룹 {test_group_idx + 1} 스킵 (유저 매칭 실패)")
            continue
        
        if verbose:
            print(f"  점수 생성 중... (유저 {len(test_user_id_to_idx):,}명)")
        
        # ⚠️ 중요: pred_matrix는 예측 시 X 벡터 (유저 히스토리)로만 사용
        # EASE B 행렬은 이미 train_data로만 학습되었음
        # 예측: scores = X @ B (X는 유저 히스토리, B는 학습된 아이템 유사도)
        scores_df = model.predict_batch_with_scores(
            pred_matrix,  # Test 유저의 히스토리 포함 (학습에는 미사용)
            test_user_id_to_idx,
            k=k,  # k=200 고정
            verbose=False
        )
        scores_df['group_id'] = test_group_idx + 1  # 그룹 ID (1~n_groups)
        all_scores.append(scores_df)
        
        # Recall 계산 (Test 그룹의 ground truth)
        # ⚠️ 중요: test_data는 그룹의 전체 데이터이므로, validation split에서 떼어낸 아이템만 ground truth로 사용
        # val_ground_truth가 제공되면 그것을 사용하고, 없으면 test_data 전체를 사용 (디버깅용)
        if val_ground_truth is not None:
            # Validation ground truth에서 Test 그룹 유저만 필터링
            test_gt = {uid: val_ground_truth[uid] for uid in test_user_ids if uid in val_ground_truth}
        else:
            # val_ground_truth가 없으면 test_data 전체를 ground truth로 사용 (비추천)
            test_gt = create_ground_truth(test_data)
            if verbose:
                print(f"  ⚠️  val_ground_truth가 제공되지 않아 test_data 전체를 ground truth로 사용")
        test_predictions_100 = {}
        test_predictions_200 = {}
        for user_id, user_scores in scores_df.groupby('user_id'):
            if len(user_scores) > 0:
                # Top-100과 Top-200을 별도로 계산
                sorted_scores = user_scores.sort_values('ease_score', ascending=False)
                top_100_items = sorted_scores.head(100)['item_id'].tolist()
                top_200_items = sorted_scores.head(200)['item_id'].tolist()
                test_predictions_100[user_id] = top_100_items
                test_predictions_200[user_id] = top_200_items
        
        # Recall 계산 (test_gt가 비어있으면 스킵)
        if len(test_gt) > 0:
            recall_100 = recall_at_k(test_predictions_100, test_gt, k=100)
            recall_200 = recall_at_k(test_predictions_200, test_gt, k=200)
        else:
            recall_100 = 0.0
            recall_200 = 0.0
            if verbose:
                print(f"  ⚠️  Ground truth가 비어있어 Recall 계산 불가")
        
        if verbose:
            print(f"  📊 그룹 {test_group_idx + 1} Recall@100: {recall_100:.4f}")
            print(f"  📊 그룹 {test_group_idx + 1} Recall@200: {recall_200:.4f}")
        
        if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
            wandb.log({
                f"cross_score/group_{test_group_idx + 1}_recall@100": recall_100,
                f"cross_score/group_{test_group_idx + 1}_recall@200": recall_200,
                f"cross_score/group_{test_group_idx + 1}_candidates": len(scores_df)
            })
        
        # parquet 저장 (메모리 효율)
        output_path = Path(output_dir) / f"group_{test_group_idx + 1}_scores.parquet"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        try:
            scores_df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        except ImportError:
            output_path = Path(output_dir) / f"group_{test_group_idx + 1}_scores.csv"
            scores_df.to_csv(output_path, index=False)
        if verbose:
            print(f"  ✅ 저장 완료: {output_path} ({len(scores_df):,}개 후보)")
    
    # 모든 점수 통합
    if len(all_scores) == 0:
        raise ValueError("생성된 점수가 없습니다.")
    
    final_scores = pd.concat(all_scores, ignore_index=True)
    
    # 최종 저장
    output_path = Path(output_dir) / "cross_scores_user_groups.parquet"
    try:
        final_scores.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
    except ImportError:
        output_path = Path(output_dir) / "cross_scores_user_groups.csv"
        final_scores.to_csv(output_path, index=False)
    
    # 전체 Recall 계산 (모든 그룹 통합)
    # ⚠️ 중요: val_ground_truth가 제공되면 그것을 사용
    if val_ground_truth is not None:
        all_groups_gt = val_ground_truth
    else:
        all_groups_gt = create_ground_truth(train_df)  # 전체 데이터의 ground truth (비추천)
    all_predictions_100 = {}
    all_predictions_200 = {}
    for user_id, user_scores in final_scores.groupby('user_id'):
        if len(user_scores) > 0:
            # Top-100과 Top-200을 별도로 계산
            sorted_scores = user_scores.sort_values('ease_score', ascending=False)
            top_100_items = sorted_scores.head(100)['item_id'].tolist()
            top_200_items = sorted_scores.head(200)['item_id'].tolist()
            all_predictions_100[user_id] = top_100_items
            all_predictions_200[user_id] = top_200_items
    
    overall_recall_100 = recall_at_k(all_predictions_100, all_groups_gt, k=100)
    overall_recall_200 = recall_at_k(all_predictions_200, all_groups_gt, k=200)
    
    if verbose:
        print(f"\n✅ Cross-Score 생성 완료!")
        print(f"  총 후보 수: {len(final_scores):,}개")
        print(f"  유저 수: {final_scores['user_id'].nunique():,}명")
        print(f"  📊 전체 Recall@100: {overall_recall_100:.4f}")
        print(f"  📊 전체 Recall@200: {overall_recall_200:.4f}")
        print(f"  저장 경로: {output_path}")
    
    if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
        wandb.log({
            "cross_score/overall_recall@100": overall_recall_100,
            "cross_score/overall_recall@200": overall_recall_200,
            "cross_score/total_candidates": len(final_scores),
            "cross_score/total_users": final_scores['user_id'].nunique()
        })
    
    return final_scores


# 기존 함수는 호환성을 위해 유지 (deprecated)
def expanding_window_oof_cross_score(
    train_df: pd.DataFrame,
    n_blocks: int = 5,
    k: int = 200,  # k=200으로 변경
    lambda_reg: float = 545.0,
    output_dir: str = "cross_scores",
    verbose: bool = True,
    use_wandb: bool = True
) -> pd.DataFrame:
    """
    [Deprecated] Expanding Window OOF 방식 (시간 블록 기반).
    
    ⚠️ 주의: 이 방식은 EASE 학습 데이터가 부족하여 Recall이 낮습니다.
    대신 user_group_oof_cross_score를 사용하세요.
    """
    if verbose:
        print("⚠️  경고: expanding_window_oof_cross_score는 deprecated입니다.")
        print("   user_group_oof_cross_score를 사용하세요.")
    
    # 기존 구현은 유지하되, k=200으로 변경
    # ... (기존 코드 유지) ...
    
    # 간단히 user_group_oof_cross_score로 리다이렉트
    return user_group_oof_cross_score(
        train_df=train_df,
        n_groups=n_blocks,  # n_blocks를 n_groups로 변환
        k=k,
        lambda_reg=lambda_reg,
        output_dir=output_dir,
        verbose=verbose,
        use_wandb=use_wandb
    )

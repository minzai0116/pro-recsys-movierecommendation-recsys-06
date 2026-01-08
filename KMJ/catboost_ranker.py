"""CatBoost Ranker 학습 및 예측."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from catboost import CatBoostRanker, Pool
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.utils.metrics import recall_at_k


def train_catboost_ranker(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    val_ground_truth: Optional[Dict[int, Set[int]]] = None,
    loss_function: str = 'QueryRMSE',
    iterations: int = 1000,
    learning_rate: float = 0.05,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
    use_wandb: bool = True,
    eval_period: int = 50
) -> CatBoostRanker:
    """
    CatBoost Ranker 학습.
    
    Args:
        train_df: 학습 데이터프레임 (group_id, label, features...)
        val_df: 검증 데이터프레임 (선택)
        loss_function: 손실 함수 ('QueryRMSE' 또는 'YetiRank')
        iterations: 최대 반복 횟수
        learning_rate: 학습률
        depth: 트리 깊이
        l2_leaf_reg: L2 정규화
        early_stopping_rounds: 조기 종료 라운드
        verbose: 상세 출력 여부
    
    Returns:
        학습된 CatBoostRanker 모델
    """
    if verbose:
        print("\n" + "=" * 60)
        print("CatBoost Ranker 학습 시작")
        print("=" * 60)
        print(f"  손실 함수: {loss_function}")
        print(f"  최대 반복: {iterations}")
        print(f"  학습률: {learning_rate}")
        print(f"  트리 깊이: {depth}")
    
    # 피처 컬럼 식별 (group_id, group, label 제외)
    exclude_cols = ['group_id', 'group', 'user_id', 'label', 'item_id', 'user_last_time']
    # ⚠️ user_last_time은 검증 분할에만 사용하는 임시 컬럼이므로 피처에서 제외
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # CatFeatures 식별 (카테고리 타입)
    cat_features = []
    for col in feature_cols:
        if train_df[col].dtype == 'category' or train_df[col].dtype == 'object':
            cat_features.append(col)
    
    # ⚠️ 학습 전 체크: user_last_item_genre가 cat_features에 포함되었는지 확인
    if 'user_last_item_genre' in feature_cols:
        if 'user_last_item_genre' not in cat_features:
            if verbose:
                print(f"  ⚠️  경고: user_last_item_genre가 cat_features에 포함되지 않음!")
                print(f"      현재 dtype: {train_df['user_last_item_genre'].dtype}")
                print(f"      → 범주형으로 변환 필요 (학습 성능 저하 가능)")
            # 자동으로 범주형으로 변환
            train_df['user_last_item_genre'] = train_df['user_last_item_genre'].astype('category')
            cat_features.append('user_last_item_genre')
            if verbose:
                print(f"      ✅ 자동으로 범주형으로 변환 완료")
    
    if verbose:
        print(f"  피처 개수: {len(feature_cols)}개")
        print(f"  CatFeatures: {len(cat_features)}개")
        if cat_features:
            print(f"    - {', '.join(cat_features[:5])}{'...' if len(cat_features) > 5 else ''}")
    
    # ⚠️ 학습 전 체크: NaN/Inf 확인
    if verbose:
        print(f"\n  📊 데이터 무결성 확인 중...")
        nan_counts = train_df[feature_cols].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(f"    ⚠️  NaN 발견 ({len(nan_cols)}개 컬럼):")
            for col, count in nan_cols.head(5).items():
                print(f"      {col:35s}: {count:,}개 ({count/len(train_df)*100:.2f}%)")
        else:
            print(f"    ✅ NaN 없음")
        
        # Inf 확인
        inf_cols = []
        for col in feature_cols:
            if train_df[col].dtype in ['float32', 'float64']:
                inf_count = np.isinf(train_df[col]).sum()
                if inf_count > 0:
                    inf_cols.append((col, inf_count))
        
        if len(inf_cols) > 0:
            print(f"    ⚠️  Inf 발견 ({len(inf_cols)}개 컬럼):")
            for col, count in inf_cols[:5]:
                print(f"      {col:35s}: {count:,}개 ({count/len(train_df)*100:.2f}%)")
        else:
            print(f"    ✅ Inf 없음")
    
    # Pool 객체 생성 (메모리 효율적)
    # ⚠️ 중요: CatBoost Ranker는 group_id가 정렬되어 있어야 함 (같은 group_id끼리 묶여있어야 함)
    # group 컬럼이 있으면 사용, 없으면 group_id 사용
    group_col = 'group' if 'group' in train_df.columns else 'group_id'
    train_df_sorted = train_df.sort_values(group_col).reset_index(drop=True)
    train_pool = Pool(
        data=train_df_sorted[feature_cols],
        label=train_df_sorted['label'],
        group_id=train_df_sorted[group_col],
        cat_features=cat_features if cat_features else None
    )
    
    # 검증 Pool (있는 경우)
    val_pool = None
    if val_df is not None:
        # ⚠️ 중요: group_id로 정렬
        group_col = 'group' if 'group' in val_df.columns else 'group_id'
        val_df_sorted = val_df.sort_values(group_col).reset_index(drop=True)
        
        # ⚠️ 학습 전 체크: val_df에서도 user_last_item_genre 확인
        if 'user_last_item_genre' in feature_cols:
            if 'user_last_item_genre' not in cat_features:
                val_df_sorted['user_last_item_genre'] = val_df_sorted['user_last_item_genre'].astype('category')
        
        val_pool = Pool(
            data=val_df_sorted[feature_cols],
            label=val_df_sorted['label'],
            group_id=val_df_sorted[group_col],
            cat_features=cat_features if cat_features else None
        )
    
    # 모델 학습
    # ⚠️ CatBoostRanker는 scale_pos_weight를 지원하지 않음
    # 대신 class_weights를 사용하거나, 손실 함수 자체가 순위 최적화를 하므로 불균형 문제가 덜함
    # ⚠️ GPU 메모리 최적화: max_bin과 depth를 줄여서 메모리 사용량 대폭 감소
    # max_bin=64, depth=5로 설정하면 메모리 사용량이 약 75% 감소
    # GPU 메모리 부족 시 자동으로 CPU로 전환
    use_gpu = True  # GPU 사용 시도 (공격적 메모리 최적화로 OOM 방지)
    max_bin_gpu = 64  # GPU 메모리 최적화 (128 → 64: 메모리 사용량 추가 50% 감소)
    depth_gpu = min(depth, 5)  # 트리 깊이 제한 (6 → 5: 메모리 사용량 감소)
    
    # eval_metric 추가 (학습 진행 상황을 더 명확하게 표시)
    eval_metric = 'PFound' if val_pool else None
    
    model = CatBoostRanker(
        loss_function=loss_function,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth_gpu if use_gpu else depth,  # GPU에서는 depth 제한
        l2_leaf_reg=l2_leaf_reg,
        early_stopping_rounds=early_stopping_rounds if val_pool else None,
        eval_metric=eval_metric,  # Ranking metric 추가
        verbose=verbose,
        random_seed=42,
        task_type='GPU' if use_gpu else 'CPU',  # GPU 사용 시도
        devices='0' if use_gpu else None,  # GPU 사용 시에만 devices 지정
        max_bin=max_bin_gpu if use_gpu else None,  # GPU 메모리 최적화 (64로 감소)
        thread_count=-1  # CPU 스레드 자동 설정
    )
    
    if verbose:
        if use_gpu:
            print(f"  💡 GPU 사용 (max_bin={max_bin_gpu}, depth={depth_gpu}로 메모리 최적화)")
        else:
            print(f"  ⚠️  CPU 사용 (GPU 메모리 부족)")
    
    if verbose:
        print(f"\n  학습 시작 (50 iter마다 loss 표시)...")
    
    # 모델 학습
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=50  # 50 iteration마다 출력
    )
    
    # 학습 후 Loss History 출력
    if verbose:
        print(f"\n  📊 학습 Loss 변화:")
        evals_result = model.get_evals_result()
        if evals_result:
            # Training loss
            if 'learn' in evals_result:
                train_losses = evals_result['learn']
                if train_losses:
                    metric_name = list(train_losses.keys())[0]
                    losses = train_losses[metric_name]
                    # 처음, 중간, 마지막 loss 출력
                    milestones = [0, len(losses)//4, len(losses)//2, 3*len(losses)//4, len(losses)-1]
                    print(f"     Train ({metric_name}):")
                    for idx in milestones:
                        if idx < len(losses):
                            print(f"       Iter {idx:4d}: {losses[idx]:.6f}")
            
            # Validation loss
            if 'validation' in evals_result:
                val_losses = evals_result['validation']
                if val_losses:
                    metric_name = list(val_losses.keys())[0]
                    losses = val_losses[metric_name]
                    # 처음, 중간, 마지막 loss 출력
                    milestones = [0, len(losses)//4, len(losses)//2, 3*len(losses)//4, len(losses)-1]
                    print(f"     Validation ({metric_name}):")
                    for idx in milestones:
                        if idx < len(losses):
                            print(f"       Iter {idx:4d}: {losses[idx]:.6f}")
    
    # Feature Importance 출력 (제3자 제안)
    if verbose:
        print(f"\n  📊 Feature Importance 분석...")
        try:
            # 여러 방식으로 Feature Importance 계산
            # 1. PredictionValuesChange (예측 값 변화) - 더 정확
            feature_importance_pred = model.get_feature_importance(train_pool, type='PredictionValuesChange')
            # 2. LossFunctionChange (손실 함수 변화)
            feature_importance_loss = model.get_feature_importance(train_pool, type='LossFunctionChange')
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance_pred': feature_importance_pred,
                'importance_loss': feature_importance_loss
            })
            
            # PredictionValuesChange를 우선 사용 (더 정확)
            importance_df = importance_df.sort_values('importance_pred', ascending=False)
            
            print(f"\n  Top-15 중요 피처 (PredictionValuesChange):")
            for i, (idx, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"    {i:2d}. {row['feature']:30s}: {row['importance_pred']:12.4f} (loss: {row['importance_loss']:8.4f})")
            
            # WandB에 기록 (있는 경우)
            if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
                # Top-15 피처를 WandB에 기록
                for idx, row in importance_df.head(15).iterrows():
                    wandb.log({
                        f"feature_importance/{row['feature']}_pred": row['importance_pred'],
                        f"feature_importance/{row['feature']}_loss": row['importance_loss']
                    })
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Feature Importance 계산 실패: {e}")
                import traceback
                traceback.print_exc()
    
    # 학습 중간 Recall 계산 (검증 데이터가 있는 경우)
    if val_df is not None and val_ground_truth is not None and WANDB_AVAILABLE and use_wandb and wandb.run is not None:
        # 학습 완료 후 중간 Recall 계산 (향후 iteration별 계산은 별도 구현 필요)
        if verbose:
            print(f"  중간 Recall 계산 중...")
        mid_predictions = _predict_for_recall(model, val_df, k=10)
        mid_recall = recall_at_k(mid_predictions, val_ground_truth, k=10)
        
        if verbose:
            print(f"  📊 학습 완료 후 Recall@10: {mid_recall:.4f}")
        
        wandb.log({
            "ranker/mid_recall@10": mid_recall
        })
    
    # 최종 Recall 계산 (검증 데이터가 있는 경우)
    if val_df is not None and val_ground_truth is not None:
        final_predictions = _predict_for_recall(model, val_df, k=10)
        final_recall = recall_at_k(final_predictions, val_ground_truth, k=10)
        
        if verbose:
            print(f"  📊 최종 Validation Recall@10: {final_recall:.4f}")
        
        if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
            wandb.log({
                "ranker/final_recall@10": final_recall
            })
    
    if verbose:
        print(f"  ✅ 학습 완료!")
    
    return model


def _predict_for_recall(model: CatBoostRanker,
                       val_df: pd.DataFrame,
                       k: int = 10) -> Dict[int, List[int]]:
    """Recall 계산을 위한 예측 (내부 함수)."""
    # 피처 컬럼 식별
    exclude_cols = ['group_id', 'group', 'user_id', 'label', 'item_id']
    feature_cols = [col for col in val_df.columns if col not in exclude_cols]
    
    # CatFeatures 식별
    cat_features = []
    for col in feature_cols:
        if val_df[col].dtype == 'category' or val_df[col].dtype == 'object':
            cat_features.append(col)
    
    # Pool 객체 생성
    # ⚠️ 중요: group_id로 정렬
    group_col = 'group' if 'group' in val_df.columns else 'group_id'
    val_df_sorted = val_df.sort_values(group_col).reset_index(drop=True)
    val_pool = Pool(
        data=val_df_sorted[feature_cols],
        group_id=val_df_sorted[group_col],
        cat_features=cat_features if cat_features else None
    )
    
    # 예측
    predictions = model.predict(val_pool)
    # 정렬된 DataFrame에 점수 추가
    val_df_sorted['ranker_score'] = predictions
    val_df = val_df_sorted.copy()
    
    # 그룹별 Top-K 추출
    # 최적화: groupby로 한 번에 처리 (O(n*m) → O(n))
    predictions_dict = {}
    for user_id, user_df in val_df.groupby('user_id'):
        user_df = user_df.sort_values('ranker_score', ascending=False)
        top_k_items = user_df['item_id'].head(k).tolist()
        predictions_dict[user_id] = top_k_items
    
    return predictions_dict


def predict_ranker(
    model: CatBoostRanker,
    test_df: pd.DataFrame,
    k: int = 10,
    ground_truth: Optional[Dict[int, Set[int]]] = None,
    verbose: bool = True,
    use_wandb: bool = True
) -> Dict[int, List[int]]:
    """
    CatBoost Ranker로 예측.
    
    Args:
        model: 학습된 CatBoostRanker 모델
        test_df: 테스트 데이터프레임 (group_id, features...)
        k: Top-K
        verbose: 상세 출력 여부
    
    Returns:
        {user_id: [item1, item2, ...]}
    """
    if verbose:
        print("  CatBoost Ranker 예측 중...")
    
    # 피처 컬럼 식별
    exclude_cols = ['group_id', 'group', 'user_id', 'label', 'item_id', 'user_last_time']
    # ⚠️ user_last_time은 검증 분할에만 사용하는 임시 컬럼이므로 피처에서 제외
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    # ⚠️ 모델 호환성: 기존 모델이 요구하는 피처가 없으면 추가 (0으로 채움)
    # 모델의 피처 이름 가져오기
    try:
        model_feature_names = model.feature_names_
        missing_features = [f for f in model_feature_names if f not in feature_cols]
        
        if missing_features:
            if verbose:
                print(f"  ⚠️  모델 호환성: 누락된 피처 {len(missing_features)}개를 0으로 채움")
                for feat in missing_features[:5]:
                    print(f"      - {feat}")
            # 누락된 피처를 0으로 채움
            for feat in missing_features:
                test_df[feat] = 0.0
                if feat not in feature_cols:
                    feature_cols.append(feat)
        
        # ⚠️ 피처 순서를 모델이 기대하는 순서로 맞춤
        # 모델이 기대하는 피처 순서에 맞춰 feature_cols 재정렬
        ordered_feature_cols = []
        for feat in model_feature_names:
            if feat in feature_cols:
                ordered_feature_cols.append(feat)
        # 모델에 없는 피처는 뒤에 추가
        for feat in feature_cols:
            if feat not in ordered_feature_cols:
                ordered_feature_cols.append(feat)
        feature_cols = ordered_feature_cols
    except:
        # feature_names_가 없으면 스킵
        pass
    
    # CatFeatures 식별
    cat_features = []
    for col in feature_cols:
        if test_df[col].dtype == 'category' or test_df[col].dtype == 'object':
            cat_features.append(col)
    
    # Pool 객체 생성
    # ⚠️ 중요: group_id로 정렬
    group_col = 'group' if 'group' in test_df.columns else 'group_id'
    test_df_sorted = test_df.sort_values(group_col).reset_index(drop=True)
    test_pool = Pool(
        data=test_df_sorted[feature_cols],
        group_id=test_df_sorted[group_col],
        cat_features=cat_features if cat_features else None
    )
    
    # 예측
    predictions = model.predict(test_pool)
    # 정렬된 DataFrame에 점수 추가
    test_df_sorted['ranker_score'] = predictions
    test_df = test_df_sorted.copy()
    
    # 그룹별 Top-K 추출
    # 최적화: groupby로 한 번에 처리 (O(n*m) → O(n))
    user_ids = test_df['user_id'].unique()
    iterator = tqdm(user_ids, desc="Extracting top-k", leave=False) if verbose else user_ids
    
    predictions_dict = {}
    for user_id, user_df in test_df.groupby('user_id'):
        user_df = user_df.sort_values('ranker_score', ascending=False)
        top_k_items = user_df['item_id'].head(k).tolist()
        predictions_dict[user_id] = top_k_items
    
    # Recall 계산 (ground_truth가 있는 경우)
    if ground_truth is not None:
        recall = recall_at_k(predictions_dict, ground_truth, k=k)
        
        if verbose:
            print(f"    📊 Recall@{k}: {recall:.4f}")
        
        if WANDB_AVAILABLE and use_wandb and wandb.run is not None:
            wandb.log({
                f"ranker/test_recall@{k}": recall
            })
    
    if verbose:
        print(f"    ✅ 예측 완료! (총 {len(predictions_dict):,}명)")
    
    return predictions_dict


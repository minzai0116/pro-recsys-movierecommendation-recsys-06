"""데이터 로드 및 전처리."""
import pandas as pd
import numpy as np
from scipy import sparse
from typing import Dict, Set, Tuple
from collections import defaultdict


def load_data(data_path: str) -> pd.DataFrame:
    """데이터 로드."""
    df = pd.read_csv(data_path)
    return df


def create_user_item_matrix(df: pd.DataFrame, 
                           user_col: str = "user",
                           item_col: str = "item") -> Tuple[sparse.csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    User-Item 행렬 생성 (희소 행렬).
    
    Returns:
        - user_item_matrix: scipy.sparse.csr_matrix (shape: [n_users, n_items])
        - user_id_to_idx: {user_id: matrix_index}
        - item_id_to_idx: {item_id: matrix_index}
    """
    # 고유 ID 추출
    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())
    
    # ID -> Index 매핑
    user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    # 행렬 생성 (1 = 상호작용 존재)
    rows = [user_id_to_idx[uid] for uid in df[user_col]]
    cols = [item_id_to_idx[iid] for iid in df[item_col]]
    data = np.ones(len(df))
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    return user_item_matrix, user_id_to_idx, item_id_to_idx


def time_based_split(df: pd.DataFrame, 
                     val_ratio: float = 0.2,
                     time_col: str = "time") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    시간 기반 train/validation 분할 (하이퍼파라미터 최적화용).
    
    주의: 메인 학습에서는 user_sequence_split을 사용합니다.
    이 함수는 hyperparameter_tuning.py에서만 사용됩니다.
    
    Args:
        df: 전체 데이터프레임
        val_ratio: validation 비율
        time_col: 시간 컬럼명
    
    Returns:
        train_df, val_df
    """
    df_sorted = df.sort_values(time_col)
    
    # 시간 기준으로 분할
    split_idx = int(len(df_sorted) * (1 - val_ratio))
    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, val_df


def user_sequence_split(df: pd.DataFrame,
                       user_col: str = "user",
                       item_col: str = "item",
                       time_col: str = "time",
                       val_items_per_user: int = 10,
                       seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    사용자별 시퀀스에서 중간 dropout + 마지막 1개를 Validation으로 분리.
    실제 대회와 유사하게: 마지막 1개는 반드시 포함, 나머지는 중간에서 랜덤 선택.
    모든 사용자를 Train에 포함시키고, 각 사용자의 시퀀스에서 일부만 분리.
    
    Args:
        df: 전체 데이터프레임
        user_col: 사용자 컬럼명
        item_col: 아이템 컬럼명
        time_col: 시간 컬럼명
        val_items_per_user: 사용자당 Validation 아이템 수 (마지막 1개 + 중간 9개 = 10개)
        seed: 랜덤 시드
    
    Returns:
        train_df, val_df
    """
    # 시드 고정
    np.random.seed(seed)
    
    # 사용자별로 시간 순 정렬
    df_sorted = df.sort_values([user_col, time_col])
    
    train_rows = []
    val_rows = []
    
    # 사용자별로 처리
    for user_id, user_df in df_sorted.groupby(user_col):
        user_items = user_df.to_dict('records')
        
        # 사용자당 최소 아이템 수 확인
        if len(user_items) <= val_items_per_user:
            # 아이템이 너무 적으면 모두 Train에 포함
            train_rows.extend(user_items)
        else:
            # 마지막 1개는 반드시 Validation에 포함
            last_item = user_items[-1]
            
            # 나머지 시퀀스 (마지막 제외)
            remaining_items = user_items[:-1]
            
            # 중간에서 랜덤하게 (val_items_per_user - 1)개 선택
            num_dropout = min(val_items_per_user - 1, len(remaining_items))
            
            if num_dropout > 0:
                # 중간 아이템에서 랜덤 선택 (인덱스 기반)
                dropout_indices = set(np.random.choice(
                    len(remaining_items), 
                    size=num_dropout, 
                    replace=False
                ))
                dropout_items = [remaining_items[i] for i in dropout_indices]
            else:
                dropout_items = []
            
            # Validation 아이템: 중간 dropout + 마지막 1개
            val_items = dropout_items + [last_item]
            
            # Train 아이템: Validation에 포함되지 않은 나머지
            train_items = [item for i, item in enumerate(remaining_items) if i not in dropout_indices]
            
            train_rows.extend(train_items)
            val_rows.extend(val_items)
    
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    
    return train_df, val_df


def create_ground_truth(df: pd.DataFrame,
                       user_col: str = "user",
                       item_col: str = "item") -> Dict[int, Set[int]]:
    """
    Ground truth 생성 (validation용).
    
    Returns:
        {user_id: {item1, item2, ...}}
    """
    ground_truth = defaultdict(set)
    
    for _, row in df.iterrows():
        user_id = int(row[user_col])
        item_id = int(row[item_col])
        ground_truth[user_id].add(item_id)
    
    return dict(ground_truth)




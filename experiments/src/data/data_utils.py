"""데이터 로드 및 전처리."""
import pandas as pd
import numpy as np
from scipy import sparse
from typing import Dict, Set, Tuple, Optional
from collections import defaultdict
import os


def load_data(data_path: str) -> pd.DataFrame:
    """데이터 로드."""
    df = pd.read_csv(data_path)
    return df


def apply_time_decay(df: pd.DataFrame,
                    time_col: str = "time",
                    gamma: float = 0.01,
                    min_weight: float = 1e-5,
                    reference_time: Optional[float] = None) -> pd.DataFrame:
    """
    시간 가중치(Time Decay) 적용.
    
    지수 감쇠 공식: W = exp(-gamma * delta_t)
    - delta_t: 일 단위 시간 차이 (데이터셋 내 max_timestamp 기준)
    - gamma: 감쇠 속도 (일 단위 기준, 보통 0.001~0.1)
    
    주의: 기준 시간은 반드시 데이터셋 내 max_timestamp를 사용합니다.
    현재 시간을 사용하면 데이터 누수(Data Leakage)가 발생할 수 있습니다.
    
    Args:
        df: ['user', 'item', 'time'] 컬럼이 포함된 데이터프레임
        time_col: 시간 컬럼명
        gamma: 시간 감쇠 계수 (일 단위)
            - 0.01: 100일 전 데이터가 약 0.36의 가중치 (보수적)
            - 0.05: 100일 전 데이터가 약 0.006의 가중치 (공격적)
        min_weight: 최소 가중치 (이보다 작으면 0으로 처리, 희소성 유지)
        reference_time: 기준 시간 (None이면 데이터셋 내 max_timestamp 사용)
    
    Returns:
        'weight' 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    
    # 1. 기준 시간 설정 (데이터셋 내 최신 interaction 시간)
    if reference_time is None:
        reference_time = df[time_col].max()
    
    # 2. 시간 차이 계산 (초 단위 -> 일 단위로 변환)
    # 일 단위(86400초)로 변환하여 gamma 값이 관리하기 편하게 함
    # 타입 안전성을 위해 명시적으로 float로 변환
    df['diff_days'] = ((reference_time - df[time_col]) / 86400.0).astype(np.float64)
    
    # 3. 지수 감쇠(Exponential Decay) 적용
    # W = exp(-gamma * delta_t)
    # 기존 weight 컬럼이 있으면 삭제 (타입 충돌 방지)
    if 'weight' in df.columns:
        df = df.drop(columns=['weight'])
    
    # weight 컬럼을 명시적으로 float64로 생성
    weight_values = np.exp(-gamma * df['diff_days'].values).astype(np.float64)
    df['weight'] = weight_values
    
    # 4. 너무 낮은 가중치는 0으로 처리 (희소성 유지)
    # min_weight를 명시적으로 float로 변환 (타입 안전성)
    min_weight_float = float(min_weight)
    mask = weight_values < min_weight_float
    df.loc[mask, 'weight'] = 0.0
    
    # 5. 가중치가 0인 행 제거 (선택적, 메모리 효율)
    # 주의: 이 부분은 선택사항이며, 필요에 따라 주석 처리 가능
    # df = df[df['weight'] > 0].copy()
    
    return df


def create_user_item_matrix(df: pd.DataFrame, 
                           user_col: str = "user",
                           item_col: str = "item",
                           weight_col: Optional[str] = None) -> Tuple[sparse.csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    User-Item 행렬 생성 (희소 행렬).
    
    Args:
        df: 데이터프레임
        user_col: 사용자 컬럼명
        item_col: 아이템 컬럼명
        weight_col: 가중치 컬럼명 (None이면 1.0 사용, 시간 가중치 적용 시 'weight' 사용)
    
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
    
    # 행렬 생성
    rows = [user_id_to_idx[uid] for uid in df[user_col]]
    cols = [item_id_to_idx[iid] for iid in df[item_col]]
    
    # 가중치 사용 (시간 가중치 또는 1.0)
    if weight_col and weight_col in df.columns:
        data = df[weight_col].values
    else:
        data = np.ones(len(df))
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    return user_item_matrix, user_id_to_idx, item_id_to_idx


def parse_tfidf_string(tfidf_string: str) -> Dict[str, float]:
    """
    TF-IDF 문자열을 딕셔너리로 파싱.
    
    Args:
        tfidf_string: "Genre1:score1|Genre2:score2|..." 형식의 문자열
    
    Returns:
        {genre: score} 딕셔너리
    """
    if pd.isna(tfidf_string) or tfidf_string == 'Unknown' or tfidf_string == '':
        return {}
    
    result = {}
    parts = str(tfidf_string).split('|')
    for part in parts:
        if ':' not in part:
            continue
        try:
            genre, score_str = part.split(':', 1)
            score = float(score_str.strip())
            result[genre.strip()] = score
        except (ValueError, AttributeError):
            continue
    
    return result


def create_augmented_user_item_matrix(df: pd.DataFrame,
                                      metadata_path: str,
                                      user_id_to_idx: Dict[int, int],
                                      item_id_to_idx: Dict[int, int],
                                      meta_weight: float = 0.1,
                                      weight_col: Optional[str] = None,
                                      include_director: bool = True,
                                      include_writer: bool = True) -> Tuple[sparse.csr_matrix, Dict[str, int], Dict[int, Tuple[str, str]]]:
    """
    아이템 뒤에 장르(TF-IDF), 감독, 작가를 붙여 확장 행렬 생성.
    User x [Item + Genre + Director + Writer]
    
    Args:
        df: User-Item 상호작용 데이터프레임
        metadata_path: 메타데이터 CSV 파일 경로 (preprocessed_metadata_TF.csv)
        user_id_to_idx: User ID -> Index 매핑
        item_id_to_idx: Item ID -> Index 매핑
        meta_weight: 메타데이터 가중치 (CF 정보와 Content 정보의 균형 조절)
        weight_col: 가중치 컬럼명 (None이면 1.0 사용)
        include_director: 감독 메타데이터 포함 여부
        include_writer: 작가 메타데이터 포함 여부
    
    Returns:
        - augmented_matrix: scipy.sparse.csr_matrix (shape: [n_users, n_items + n_meta])
        - meta_to_idx: {meta_name: matrix_index} 매핑 (장르, 감독, 작가 모두 포함)
        - index_to_type: {matrix_index: (meta_type, original_id)} 매핑
          - meta_type: 'genre', 'director', 'writer' 중 하나
          - original_id: 원본 메타데이터 ID (장르명, 감독명, 작가명)
    """
    # 1. 메타데이터 로드
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
    
    meta_df = pd.read_csv(metadata_path)
    
    # 2. 고유 메타데이터 추출 및 인덱싱
    n_items = len(item_id_to_idx)
    current_idx = n_items
    
    # 2-1. 장르 추출 및 인덱싱
    all_genres = set()
    for genres_str in meta_df['genres']:
        if pd.notna(genres_str) and genres_str != 'Unknown' and genres_str != '':
            parsed = parse_tfidf_string(genres_str)
            all_genres.update(parsed.keys())
    
    sorted_genres = sorted(all_genres)
    genre_to_idx = {genre: current_idx + idx for idx, genre in enumerate(sorted_genres)}
    current_idx += len(sorted_genres)
    
    # 2-2. 감독 추출 및 인덱싱 (Unknown 제외)
    director_to_idx = {}
    if include_director:
        all_directors = set()
        for director in meta_df['director']:
            if pd.notna(director) and director != 'Unknown' and director != '':
                all_directors.add(str(director))
        
        sorted_directors = sorted(all_directors)
        director_to_idx = {director: current_idx + idx for idx, director in enumerate(sorted_directors)}
        current_idx += len(sorted_directors)
    
    # 2-3. 작가 추출 및 인덱싱 (Unknown 제외)
    writer_to_idx = {}
    if include_writer:
        all_writers = set()
        for writer in meta_df['writer']:
            if pd.notna(writer) and writer != 'Unknown' and writer != '':
                all_writers.add(str(writer))
        
        sorted_writers = sorted(all_writers)
        writer_to_idx = {writer: current_idx + idx for idx, writer in enumerate(sorted_writers)}
        current_idx += len(sorted_writers)
    
    # 메타데이터 인덱스 통합
    meta_to_idx = {}
    meta_to_idx.update(genre_to_idx)
    meta_to_idx.update(director_to_idx)
    meta_to_idx.update(writer_to_idx)
    
    # index_to_type 매핑 생성 (제3자 제안: 메타데이터 타입 명시적 관리)
    # 형식: {matrix_index: (meta_type, original_id)}
    index_to_type = {}
    for genre, idx in genre_to_idx.items():
        index_to_type[idx] = ('genre', genre)
    for director, idx in director_to_idx.items():
        index_to_type[idx] = ('director', director)
    for writer, idx in writer_to_idx.items():
        index_to_type[idx] = ('writer', writer)
    
    # 로그 출력
    meta_parts = [f"{n_items} items", f"{len(sorted_genres)} genres"]
    if include_director:
        meta_parts.append(f"{len(director_to_idx)} directors")
    if include_writer:
        meta_parts.append(f"{len(writer_to_idx)} writers")
    meta_str = " + ".join(meta_parts)
    print(f"  → 확장 행렬: {len(user_id_to_idx)} users × ({meta_str})")
    
    # 3. 아이템별 메타데이터 캐싱
    item_genre_map = {}
    item_director_map = {}
    item_writer_map = {}
    
    for _, row in meta_df.iterrows():
        item_id = int(row['item'])
        if item_id in item_id_to_idx:
            # 장르 (TF-IDF)
            genres_str = row['genres']
            item_genre_map[item_id] = parse_tfidf_string(genres_str)
            
            # 감독 (Binary)
            if include_director:
                director = row['director']
                if pd.notna(director) and director != 'Unknown' and director != '':
                    item_director_map[item_id] = str(director)
            
            # 작가 (Binary)
            if include_writer:
                writer = row['writer']
                if pd.notna(writer) and writer != 'Unknown' and writer != '':
                    item_writer_map[item_id] = str(writer)
    
    # 4. 기본 User-Item 행렬 데이터 준비
    rows = [user_id_to_idx[uid] for uid in df['user']]
    cols = [item_id_to_idx[iid] for iid in df['item']]
    
    # 가중치 사용 (시간 가중치 또는 1.0)
    if weight_col and weight_col in df.columns:
        data = df[weight_col].values
    else:
        data = np.ones(len(df))
    
    # 5. User-Metadata 데이터 누적 (유저별로 본 영화의 메타데이터 점수를 합산)
    # 최적화: iterrows() 대신 벡터화된 연산 사용 (O(n) → O(n) but 훨씬 빠름)
    user_meta_scores = defaultdict(lambda: defaultdict(float))  # 장르, 감독, 작가 모두 포함
    user_item_counts = defaultdict(int)  # 유저별 본 영화 수 (정규화용)
    
    # 벡터화된 연산: numpy 배열로 변환하여 빠르게 처리
    user_ids = df['user'].values
    item_ids = df['item'].values
    
    for i in range(len(df)):
        user_id = user_ids[i]
        item_id = item_ids[i]
        
        # 해당 아이템의 가중치 (시간 가중치 또는 1.0)
        item_weight = data[i] if weight_col and weight_col in df.columns else 1.0
        
        # 유저별 본 영화 수 카운트 (정규화용)
        user_item_counts[user_id] += 1
        
        # 장르 점수 누적 (TF-IDF) - meta_weight 곱하기 전에 누적
        if item_id in item_genre_map:
            for genre, tfidf_score in item_genre_map[item_id].items():
                user_meta_scores[user_id][genre] += item_weight * tfidf_score
        
        # 감독 점수 누적 (Binary) - meta_weight 곱하기 전에 누적
        if include_director and item_id in item_director_map:
            director = item_director_map[item_id]
            user_meta_scores[user_id][director] += item_weight * 1.0
        
        # 작가 점수 누적 (Binary) - meta_weight 곱하기 전에 누적
        if include_writer and item_id in item_writer_map:
            writer = item_writer_map[item_id]
            user_meta_scores[user_id][writer] += item_weight * 1.0
    
    # ⚠️ 비판적 포인트: 메타데이터 신호 Capping (제3자 분석 반영)
    # 제3자 분석: 정규화 제거 시 메타데이터 신호가 폭발적으로 커짐
    # - 유저가 특정 감독의 영화를 10편 봤다면: 10 * 0.5 = 5.0
    # - 아이템 신호: 1.0 → X^T X에서 1.0^2 = 1.0
    # - 메타데이터 신호: 5.0 → X^T X에서 5.0^2 = 25.0 (25배!)
    # - 결과: 모델이 메타데이터에만 집중, 아이템 추천 능력 상실 (Recall 0.0444)
    # 해결: 메타데이터 신호를 최대 1.0으로 Capping 후 meta_weight 적용
    # "그 감독의 영화를 얼마나 많이 봤나"보다 "그 감독의 영화를 본 적이 있는가"가 중요
    # 💡 제3자 제안: Capping을 meta_weight 곱하기 전에 적용하여 정확한 비율 유지
    for user_id in user_meta_scores:
        for meta_key in list(user_meta_scores[user_id].keys()):
            # 1. 먼저 누적된 점수에 Capping 적용 (최대 1.0)
            capped_score = min(user_meta_scores[user_id][meta_key], 1.0)
            # 2. 그 후 meta_weight를 곱해 최종 행렬에 주입
            # 이렇게 하면 메타데이터 칸의 최대값이 정확히 meta_weight가 되어 아이템(1.0)과 균형 유지
            user_meta_scores[user_id][meta_key] = capped_score * meta_weight
    
    # 6. User-Metadata 데이터를 행렬 형식으로 변환
    n_users = len(user_id_to_idx)
    n_cols = current_idx  # n_items + n_genres + n_directors + n_writers
    
    # User-Item 데이터와 User-Metadata 데이터를 합침
    # numpy array는 append가 없으므로 리스트로 변환
    all_rows = rows.copy().tolist() if isinstance(rows, np.ndarray) else list(rows)
    all_cols = cols.copy().tolist() if isinstance(cols, np.ndarray) else list(cols)
    all_data = data.copy().tolist() if isinstance(data, np.ndarray) else list(data)
    
    # 메타데이터 타입별 카운트 (디버깅용)
    genre_count = 0
    director_count = 0
    writer_count = 0
    
    for user_id, meta_scores in user_meta_scores.items():
        if user_id not in user_id_to_idx:
            continue
        user_idx = user_id_to_idx[user_id]
        for meta_key, score in meta_scores.items():
            if meta_key in meta_to_idx:
                all_rows.append(user_idx)
                all_cols.append(meta_to_idx[meta_key])
                all_data.append(score)
                
                # 메타데이터 타입별 카운트 (디버깅용)
                if meta_key in genre_to_idx:
                    genre_count += 1
                elif meta_key in director_to_idx:
                    director_count += 1
                elif meta_key in writer_to_idx:
                    writer_count += 1
    
    # 디버깅: 메타데이터 실제 주입 확인
    print(f"  → 메타데이터 실제 주입 확인:")
    print(f"     장르 엔트리: {genre_count:,}개")
    print(f"     감독 엔트리: {director_count:,}개")
    print(f"     작가 엔트리: {writer_count:,}개")
    
    # 7. 최종 확장 행렬 생성
    augmented_matrix = sparse.csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n_users, n_cols)
    )
    
    return augmented_matrix, meta_to_idx, index_to_type


def validate_metadata_integrity(df: pd.DataFrame, 
                                 index_to_type: Dict[int, Tuple[str, str]],
                                 verbose: bool = True) -> Dict[str, int]:
    """
    메타데이터 무결성 검증 (제3자 제안).
    
    Args:
        df: 피처 엔지니어링된 데이터프레임
        index_to_type: {matrix_index: (meta_type, original_id)} 매핑
        verbose: 상세 출력 여부
    
    Returns:
        각 meta_type별 nunique() 결과 딕셔너리
    """
    results = {}
    
    # index_to_type에서 meta_type별로 그룹화
    type_to_indices = {}
    for idx, (meta_type, _) in index_to_type.items():
        if meta_type not in type_to_indices:
            type_to_indices[meta_type] = []
        type_to_indices[meta_type].append(idx)
    
    if verbose:
        print("  → 메타데이터 무결성 검증:")
    
    for meta_type, indices in type_to_indices.items():
        count = len(indices)
        results[meta_type] = count
        
        if verbose:
            print(f"     {meta_type}: {count:,}개")
    
    return results


def validate_time_order(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        time_col: str = "time",
                        verbose: bool = True) -> bool:
    """
    시간 순서 검증 (제3자 제안: assert 문 포함).
    
    Args:
        train_df: 학습 데이터
        test_df: 테스트 데이터 (점수 생성 대상)
        time_col: 시간 컬럼명
        verbose: 상세 출력 여부
    
    Returns:
        검증 통과 여부 (True: 통과, False: 실패)
    
    Raises:
        AssertionError: 시간 순서 위반 시
    """
    train_max_time = train_df[time_col].max()
    test_min_time = test_df[time_col].min()
    
    # 제3자 제안: assert 문으로 엄격히 검증
    # ⚠️ 주의: 블록 분할 시 경계값이 같을 수 있으므로 <= 로 검증
    assert train_max_time <= test_min_time, \
        f"시간 순서 위반: Train 최대 시간 ({train_max_time}) > Test 최소 시간 ({test_min_time})"
    
    if verbose:
        print(f"  ✅ 시간 순서 검증 통과:")
        print(f"     Train 최대 시간: {train_max_time}")
        print(f"     Test 최소 시간: {test_min_time}")
        print(f"     차이: {test_min_time - train_max_time}초")
    
    return True


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
                dropout_indices = set()  # 빈 집합으로 초기화
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




"""피처 엔지니어링 (2-Stage CatBoost Ranker용)."""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm


def add_ease_features(scores_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    EASE 관련 피처 추가 (Multi-EASE 대응).
    
    피처 목록:
    - ease_score_norm_lambda100/500/2000: 각 Lambda별 정규화된 점수 (독립 피처!)
    - ease_score_ratio_log: log1p(ease_score / user_mean_ease_score) - 기존 호환성
    
    Args:
        scores_df: EASE 점수가 포함된 DataFrame
                  (columns: user_id, item_id, ease_score, 
                   ease_score_norm_lambda100, ease_score_norm_lambda500, ease_score_norm_lambda2000)
        verbose: 진행 상황 출력
    
    Returns:
        피처가 추가된 DataFrame
    """
    df = scores_df.copy()
    
    if verbose:
        print("  EASE 피처 계산 중...")
    
    # 🔥 Multi-EASE: Lambda별 독립 피처 확인
    multi_ease_cols = [col for col in df.columns if col.startswith('ease_score_norm_lambda')]
    
    if len(multi_ease_cols) > 0:
        # Multi-EASE 모드: 3개 독립 피처 사용
        if verbose:
            print(f"    🔥 Multi-EASE 모드 감지!")
            print(f"       독립 피처: {multi_ease_cols}")
        
        # 각 Lambda별 점수가 이미 정규화되어 있으므로 그대로 사용
        # CatBoost가 "어느 모델을 믿을지" 스스로 학습
        
        # 기존 호환성을 위한 ease_score_ratio_log (중간 lambda 사용)
        if 'ease_score' in df.columns:
            user_mean_ease = df.groupby('user_id')['ease_score'].transform('mean')
            ease_score_ratio = (df['ease_score'] / (user_mean_ease + 1e-8)).fillna(1.0).astype('float32')
            df['ease_score_ratio_log'] = np.log1p(ease_score_ratio).astype('float32')
            
            if verbose:
                print(f"    ✅ ease_score_ratio_log 추가 (기존 호환)")
    else:
        # 단일 EASE 모드 (기존 방식)
        if verbose:
            print(f"    ⚠️  단일 EASE 모드 (legacy)")
        
        user_mean_scores = df.groupby('user_id')['ease_score'].transform('mean')
        ease_score_ratio = (df['ease_score'] / user_mean_scores).fillna(1.0).astype('float32')
        df['ease_score_ratio_log'] = np.log1p(ease_score_ratio).astype('float32')
        
        if verbose:
            print(f"    ✅ ease_score_ratio_log 추가")
    
    # ease_rank와 ease_score 원본은 삭제 (메타데이터 신호를 위해)
    if 'ease_rank' in df.columns:
        df = df.drop(columns=['ease_rank'])
    if 'ease_score' in df.columns:
        df = df.drop(columns=['ease_score'])
    
    if verbose:
        print(f"    ✅ EASE 피처 추가 완료 (ease_score_ratio_log)")
    
    return df


def add_sasrec_features_DISABLED(scores_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    SASRec 관련 피처 추가.
    
    피처:
    1. sasrec_score_ratio_log (유저 평균 대비 비율 log, float32)
    2. ease_sasrec_score_product (EASE * SASRec 상호작용, float32)
    
    Args:
        scores_df: DataFrame (user_id, item_id, ..., sasrec_score, sasrec_rank)
        verbose: 상세 출력 여부
    
    Returns:
        DataFrame with additional SASRec features
    """
    df = scores_df.copy()
    
    # sasrec_score가 없으면 스킵
    if 'sasrec_score' not in df.columns:
        if verbose:
            print("  SASRec 피처 없음 (스킵)")
        return df
    
    if verbose:
        print("  SASRec 피처 계산 중...")
    
    # sasrec_score_ratio 계산 (유저 평균 대비 비율)
    # sasrec_score=0인 경우 (SASRec 후보가 아닌 경우) 평균 계산에서 제외
    sasrec_score_nonzero = df[df['sasrec_score'] > 0].copy()
    
    if len(sasrec_score_nonzero) > 0:
        user_mean_scores = sasrec_score_nonzero.groupby('user_id')['sasrec_score'].transform('mean')
        sasrec_score_ratio = (
            sasrec_score_nonzero['sasrec_score'] / user_mean_scores
        ).fillna(1.0)
        
        # log 변환
        sasrec_score_ratio_log = np.log1p(sasrec_score_ratio).astype('float32')
        
        # 원본 DataFrame에 병합
        df.loc[sasrec_score_nonzero.index, 'sasrec_score_ratio_log'] = sasrec_score_ratio_log
        df['sasrec_score_ratio_log'] = df['sasrec_score_ratio_log'].fillna(0.0).astype('float32')
    else:
        df['sasrec_score_ratio_log'] = 0.0
    
    # ========================================
    # 상호작용 피처 1: EASE * SASRec 곱
    # ========================================
    # 🧪 [The Final Test] EASE 포함 피처 제거
    # if 'ease_score_ratio_log' in df.columns:
    #     df['ease_sasrec_score_product'] = (
    #         df['ease_score_ratio_log'] * df['sasrec_score_ratio_log']
    #     ).astype('float32')
    #     
    #     if verbose:
    #         print(f"    ✅ ease_sasrec_score_product 추가 (EASE * SASRec 상호작용)")
    
    # ========================================
    # P1: SASRec Rank 피처 (안정적인 신호)
    # ========================================
    # Logit 대신 Rank 자체를 피처로 사용
    if 'sasrec_rank' in df.columns:
        # rank가 0인 경우 (SASRec 후보가 아님) → 201로 처리
        df['sasrec_rank_feature'] = df['sasrec_rank'].replace(0, 201).astype('int16')
        
        # 정규화된 rank (0~1 사이)
        df['sasrec_rank_norm'] = (201 - df['sasrec_rank_feature']) / 200.0
        df['sasrec_rank_norm'] = df['sasrec_rank_norm'].astype('float32')
        
        if verbose:
            print(f"    ✅ sasrec_rank_feature, sasrec_rank_norm 추가 (안정적 신호)")
    
    # ========================================
    # 상호작용 피처 2: SASRec vs EASE 의견 차이
    # ========================================
    # 두 모델의 rank 차이 → 큰 차이 = 주목할 만한 아이템
    if 'sasrec_rank' in df.columns and 'ease_rank' in df.columns:
        # rank가 0인 경우 (해당 모델이 추천하지 않음) 처리
        sasrec_rank_safe = df['sasrec_rank'].replace(0, 201)  # 순위 밖
        ease_rank_safe = df['ease_rank'].replace(0, 201)
        
        df['sasrec_ease_disagreement'] = (
            abs(sasrec_rank_safe - ease_rank_safe) / 200.0
        ).astype('float32')
        
        if verbose:
            print(f"    ✅ sasrec_ease_disagreement 추가 (모델 의견 차이)")
    
    # sasrec_score와 sasrec_rank 원본은 삭제 (정규화된 버전만 유지)
    if 'sasrec_score' in df.columns:
        df = df.drop(columns=['sasrec_score'])
    if 'sasrec_rank' in df.columns:
        df = df.drop(columns=['sasrec_rank'])
    
    # ease_rank도 삭제 (ease_score_ratio_log만 유지)
    if 'ease_rank' in df.columns:
        df = df.drop(columns=['ease_rank'])
    
    if verbose:
        print(f"    ✅ SASRec 피처 추가 완료")
        print(f"       - sasrec_score_ratio_log (정규화)")
        print(f"       - ease_sasrec_score_product (EASE * SASRec)")
        print(f"       - sasrec_ease_disagreement (의견 차이)")
    
    return df


def add_item_features(scores_df: pd.DataFrame,
                     train_df: pd.DataFrame,
                     metadata_path: Optional[str] = None,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Item 피처 추가 (P1).
    
    피처:
    1. item_popularity (전체 기간 인기도, int32)
    2. item_popularity_7d (최근 7일 인기도, int32) - 트렌드 반영
    3. item_popularity_30d (최근 30일 인기도, int32) - 트렌드 반영
    4. item_popularity_recent_ratio (최근 인기도 비율, float32) - 트렌드 강도
    5. item_decade (연대, category) - metadata_path 필요
    6. item_director (카테고리, category) - metadata_path 필요
    7. item_writer (카테고리, category) - metadata_path 필요
    8. item_release_year (연도, int16) - metadata_path 필요
    
    Args:
        scores_df: Cross-Score DataFrame
        train_df: Train 데이터프레임 (통계 계산용)
        metadata_path: 메타데이터 CSV 파일 경로 (선택)
        verbose: 상세 출력 여부
    
    Returns:
        DataFrame with additional Item features
    """
    df = scores_df.copy()
    
    if verbose:
        print("  Item 피처 계산 중...")
    
    # 1. item_popularity (전체 기간 인기도) - log1p 처리하여 영향력 감소
    item_popularity = train_df['item'].value_counts().to_dict()
    item_popularity_values = df['item_id'].map(item_popularity).fillna(0).astype('int32')
    
    # ⚠️ item_popularity 최종 가지치기: 로그 버전만 유지 (중복 제거)
    # item_popularity와 item_popularity_log는 사실상 같은 정보 (도합 12.4%)
    df['item_popularity_log'] = np.log1p(item_popularity_values).astype('float32')
    # item_popularity는 삭제 (최근 인기도 비율 계산용으로만 임시 사용)
    
    # 최근 인기도 비율 계산용 임시 컬럼 (계산 후 삭제)
    df['_item_popularity_temp'] = item_popularity_values
    
    # 1-1. 최근 인기도 (트렌드 반영) - 시간 윈도우 기반
    # ⚠️ 데이터 누수 방지: train_df는 train_split이므로 validation 데이터는 포함되지 않음
    if 'time' in train_df.columns:
        # 최신 시간 기준 (train_split의 최신 시간 사용, validation 데이터 미포함)
        max_time = train_df['time'].max()
        
        # 최근 7일 인기도
        recent_7d_threshold = max_time - (7 * 86400)  # 7일 = 7 * 86400초
        recent_7d_df = train_df[train_df['time'] >= recent_7d_threshold]
        item_popularity_7d = recent_7d_df['item'].value_counts().to_dict()
        df['item_popularity_7d'] = df['item_id'].map(item_popularity_7d).fillna(0).astype('int32')
        
        # 최근 30일 인기도
        recent_30d_threshold = max_time - (30 * 86400)  # 30일
        recent_30d_df = train_df[train_df['time'] >= recent_30d_threshold]
        item_popularity_30d = recent_30d_df['item'].value_counts().to_dict()
        df['item_popularity_30d'] = df['item_id'].map(item_popularity_30d).fillna(0).astype('int32')
        
        # 최근 인기도 비율 (전체 대비 최근 조회수 비율)
        df['item_popularity_recent_ratio'] = (
            df['item_popularity_30d'] / (df['_item_popularity_temp'] + 1)
        ).fillna(0.0).astype('float32')
        
        # 임시 컬럼 삭제
        df = df.drop(columns=['_item_popularity_temp'])
        
        if verbose:
            print(f"    ✅ 최근 인기도 피처 추가 (7일, 30일, 비율)")
            print(f"    ✅ item_popularity 삭제 (로그 버전만 유지)")
    else:
        # time 컬럼이 없으면 기본값으로 채움
        df['item_popularity_7d'] = 0
        df['item_popularity_30d'] = 0
        df['item_popularity_recent_ratio'] = 0.0
        # 임시 컬럼 삭제
        if '_item_popularity_temp' in df.columns:
            df = df.drop(columns=['_item_popularity_temp'])
        if verbose:
            print(f"    ⚠️  time 컬럼 없음: 최근 인기도 피처 스킵")
    
    # 2-5. 메타데이터 피처 (metadata_path가 있는 경우)
    if metadata_path:
        try:
            meta_df = pd.read_csv(metadata_path)
            # 최적화: iterrows 대신 벡터화된 연산 사용 (O(n) → O(n) but 훨씬 빠름)
            meta_dict = {}
            meta_df['item'] = meta_df['item'].astype(int)
            for item_id in meta_df['item'].unique():
                item_row = meta_df[meta_df['item'] == item_id].iloc[0]
                # year 파싱 (안전하게 처리)
                year_val = item_row.get('year', 0)
                year_int = 0
                if pd.notna(year_val):
                    try:
                        # 문자열인 경우 숫자만 추출 (예: "1970s" -> 1970)
                        if isinstance(year_val, str):
                            # 숫자만 추출
                            numbers = re.findall(r'\d+', str(year_val))
                            if numbers:
                                year_int = int(numbers[0])  # 첫 번째 숫자 사용
                            else:
                                year_int = 0
                        else:
                            year_int = int(year_val)
                    except (ValueError, TypeError):
                        year_int = 0
                
                meta_dict[item_id] = {
                    'genres': item_row.get('genres', ''),
                    'director': item_row.get('director', 'Unknown'),
                    'writer': item_row.get('writer', 'Unknown'),
                    'year': year_int
                }
            
            # ⚠️ item_genre_tfidf 삭제: 평균은 정보 손실이 크고 user_item_genre_similarity와 중복
            # user_item_genre_similarity가 더 정교한 신호를 제공하므로 제거
            
            # item_director, item_writer
            df['item_director'] = df['item_id'].map(
                lambda x: meta_dict.get(x, {}).get('director', 'Unknown')
            ).astype('category')
            
            df['item_writer'] = df['item_id'].map(
                lambda x: meta_dict.get(x, {}).get('writer', 'Unknown')
            ).astype('category')
            
            # item_release_year
            df['item_release_year'] = df['item_id'].map(
                lambda x: meta_dict.get(x, {}).get('year', 0)
            ).astype('int16')
            
            # 추가: 연도 관련 피처 (메타데이터 강화)
            # 연대별 분류 (1970s, 1980s, 1990s, 2000s, 2010s, 2020s)
            def get_decade(year):
                if year == 0:
                    return 'Unknown'
                elif year < 1970:
                    return 'pre_1970'
                elif year < 1980:
                    return '1970s'
                elif year < 1990:
                    return '1980s'
                elif year < 2000:
                    return '1990s'
                elif year < 2010:
                    return '2000s'
                elif year < 2020:
                    return '2010s'
                else:
                    return '2020s'
            
            df['item_decade'] = df['item_release_year'].apply(get_decade).astype('category')
            
            # ⚠️ item_recency 삭제: item_decade와 중복되고 고정 연도 기준은 시간에 따라 약화
            # item_decade가 카테고리로 더 유용함
            
            # 감독/작가 인기도 (Train에서 본 횟수)
            director_popularity = train_df.merge(
                meta_df[['item', 'director']], on='item', how='left'
            )['director'].value_counts().to_dict()
            
            # ⚠️ 피처 다이어트: director/writer popularity 제거 (중요도 < 0.5%)
            # writer_popularity = train_df.merge(
            #     meta_df[['item', 'writer']], on='item', how='left'
            # )['writer'].value_counts().to_dict()
            # 
            # df['item_director_popularity'] = df['item_director'].map(
            #     lambda x: director_popularity.get(x, 0)
            # ).astype('int32')
            # 
            # df['item_writer_popularity'] = df['item_writer'].map(
            #     lambda x: writer_popularity.get(x, 0)
            # ).astype('int32')
            
            if verbose:
                print(f"    ✅ 메타데이터 피처 추가 완료 (director/writer popularity 제거)")
        except Exception as e:
            if verbose:
                print(f"    ⚠️  메타데이터 로드 실패: {e}")
    
    if verbose:
        print(f"    ✅ Item 피처 추가 완료")
    
    return df


def add_user_features(scores_df: pd.DataFrame,
                     train_df: pd.DataFrame,
                     metadata_path: Optional[str] = None,
                     verbose: bool = True) -> pd.DataFrame:
    """
    User 피처 추가 (P1).
    
    피처:
    1. user_total_watched (Train에서 본 영화 수, int32)
    2. user_activity_level (활동성 레벨, int8: 0=낮음, 1=보통, 2=높음)
    3. user_preferred_genres (선호 장르 Top-3, string) - TODO: 메타데이터 필요
    
    Args:
        scores_df: Cross-Score DataFrame
        train_df: Train 데이터프레임 (통계 계산용)
        verbose: 상세 출력 여부
    
    Returns:
        DataFrame with additional User features
    """
    df = scores_df.copy()
    
    if verbose:
        print("  User 피처 계산 중...")
    
    # 1. user_total_watched (Train 기준)
    user_watched_counts = train_df.groupby('user')['item'].nunique().to_dict()
    df['user_total_watched'] = df['user_id'].map(user_watched_counts).fillna(0).astype('int32')
    
    # 2. user_activity_level (활동성 레벨)
    # quantile 기반: 하위 33% = 낮음(0), 중간 33% = 보통(1), 상위 33% = 높음(2)
    watched_values = df['user_total_watched'].values
    q33 = np.percentile(watched_values, 33)
    q67 = np.percentile(watched_values, 67)
    
    df['user_activity_level'] = pd.cut(
        df['user_total_watched'],
        bins=[-np.inf, q33, q67, np.inf],
        labels=[0, 1, 2]
    ).astype('int8')
    
    # 3. user_preferred_genres (Train에서 본 영화의 장르 Top-3)
    if metadata_path:
        try:
            meta_df = pd.read_csv(metadata_path)
            # 유저별 본 영화의 장르 수집
            user_genre_counts = defaultdict(lambda: defaultdict(float))
            
            # Train 데이터와 메타데이터 병합
            train_with_meta = train_df.merge(
                meta_df[['item', 'genres']], 
                on='item', 
                how='left'
            )
            
            # 유저별 장르 TF-IDF 점수 합산
            for _, row in train_with_meta.iterrows():
                genres_str = row.get('genres', '')
                if pd.notna(genres_str) and isinstance(genres_str, str):
                    user_id = row['user']
                    for genre_pair in genres_str.split('|'):
                        if ':' in genre_pair:
                            genre, score = genre_pair.split(':')
                            user_genre_counts[user_id][genre] += float(score)
            
            # 유저별 Top-3 장르 추출
            def get_top3_genres(user_id):
                if user_id in user_genre_counts:
                    genres = user_genre_counts[user_id]
                    top3 = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
                    return '|'.join([g[0] for g in top3])
                return 'Unknown'
            
            df['user_preferred_genres'] = df['user_id'].map(get_top3_genres).astype('category')
            
            if verbose:
                print(f"    ✅ user_preferred_genres 계산 완료")
        except Exception as e:
            if verbose:
                print(f"    ⚠️  user_preferred_genres 계산 실패: {e}")
            df['user_preferred_genres'] = 'Unknown'
    else:
        df['user_preferred_genres'] = 'Unknown'
    
    if verbose:
        print(f"    ✅ User 피처 추가 완료")
    
    return df


def add_interaction_features(scores_df: pd.DataFrame,
                            train_df: pd.DataFrame,
                            metadata_path: Optional[str] = None,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Interaction 피처 추가 (P2).
    
    피처:
    1. user_item_interaction_count (Train에서 상호작용 횟수, int16)
    2. user_director_match (Train에서 본 감독과 일치 여부, bool)
    3. user_writer_match (Train에서 본 작가와 일치 여부, bool)
    4. user_item_genre_similarity (유저 Top-3 장르와 아이템 장르 유사도, float32) - 메타데이터 강화
    5. user_last_item_similarity (유저 마지막 본 아이템과 후보 아이템의 장르 유사도, float32) - 순서 피처
    
    Args:
        scores_df: Cross-Score DataFrame
        train_df: Train 데이터프레임 (통계 계산용)
        metadata_path: 메타데이터 CSV 파일 경로 (선택)
        verbose: 상세 출력 여부
    
    Returns:
        DataFrame with additional Interaction features
    """
    df = scores_df.copy()
    
    if verbose:
        print("  Interaction 피처 계산 중...")
    
    # 1. user_item_interaction_count (Train 기준) - 복원
    # 최적화: apply 대신 merge 사용 (O(n) → O(n) but 훨씬 빠름)
    user_item_counts = train_df.groupby(['user', 'item']).size().reset_index(name='count')
    user_item_counts.columns = ['user_id', 'item_id', 'user_item_interaction_count']
    df = df.merge(user_item_counts, on=['user_id', 'item_id'], how='left')
    df['user_item_interaction_count'] = df['user_item_interaction_count'].fillna(0).astype('int16')
    
    if verbose:
        print(f"    ✅ user_item_interaction_count 추가 (복원)")
    
    # 2-3. user_director_match, user_writer_match
    if metadata_path:
        try:
            meta_df = pd.read_csv(metadata_path)
            
            # Train에서 유저별 본 감독/작가 집합 생성
            train_with_meta = train_df.merge(
                meta_df[['item', 'director', 'writer', 'genres']], 
                on='item', 
                how='left'
            )
            
            user_directors = train_with_meta.groupby('user')['director'].apply(set).to_dict()
            user_writers = train_with_meta.groupby('user')['writer'].apply(set).to_dict()
            
            # 유저별 Top-1 장르 추출 (보편적 피처)
            user_top_genre = {}
            for user_id, user_data in train_with_meta.groupby('user'):
                genre_counts = defaultdict(float)
                for genres_str in user_data['genres'].dropna():
                    if isinstance(genres_str, str):
                        for genre_pair in genres_str.split('|'):
                            if ':' in genre_pair:
                                genre = genre_pair.split(':')[0]
                                score = float(genre_pair.split(':')[1])
                                genre_counts[genre] += score
                if genre_counts:
                    top_genre = max(genre_counts.items(), key=lambda x: x[1])[0]
                    user_top_genre[user_id] = top_genre
                else:
                    user_top_genre[user_id] = None
            
            # 후보 아이템의 감독/작가/장르 정보
            item_director_map = meta_df.set_index('item')['director'].to_dict()
            item_writer_map = meta_df.set_index('item')['writer'].to_dict()
            
            # 아이템의 장르 정보 (Top-1 장르 추출)
            item_top_genre = {}
            for idx, row in meta_df.iterrows():
                item_id = row['item']
                genres_str = row.get('genres', '')
                if pd.notna(genres_str) and isinstance(genres_str, str):
                    genre_scores = {}
                    for genre_pair in genres_str.split('|'):
                        if ':' in genre_pair:
                            genre = genre_pair.split(':')[0]
                            score = float(genre_pair.split(':')[1])
                            if genre not in genre_scores or score > genre_scores[genre]:
                                genre_scores[genre] = score
                    if genre_scores:
                        item_top_genre[item_id] = max(genre_scores.items(), key=lambda x: x[1])[0]
                    else:
                        item_top_genre[item_id] = None
                else:
                    item_top_genre[item_id] = None
            
            # 매칭 확인 (벡터화, 임시 컬럼 없이 직접 계산)
            def check_director_match(row):
                user_id = row['user_id']
                item_id = row['item_id']
                item_dir = item_director_map.get(item_id, '')
                user_dir_set = user_directors.get(user_id, set())
                return item_dir in user_dir_set if item_dir else False
            
            def check_writer_match(row):
                user_id = row['user_id']
                item_id = row['item_id']
                item_wr = item_writer_map.get(item_id, '')
                user_wr_set = user_writers.get(user_id, set())
                return item_wr in user_wr_set if item_wr else False
            
            # ⚠️ user_genre_match 삭제: user_item_genre_similarity가 더 정교한 신호 제공
            # 이진 매칭보다 연속값 유사도가 더 유용함
            
            # 매칭 확인 (apply 사용)
            # ⚠️ 피처 다이어트: user_director_match 제거 (중요도 0.47%)
            # df['user_director_match'] = df.apply(check_director_match, axis=1).astype('bool')
            df['user_writer_match'] = df.apply(check_writer_match, axis=1).astype('bool')
            
            # 추가: 유저-아이템 장르 유사도 (TF-IDF 점수 기반)
            # meta_dict 생성 (genres 정보 포함)
            meta_dict = {}
            for idx, row in meta_df.iterrows():
                item_id = row['item']
                meta_dict[item_id] = {
                    'genres': row.get('genres', '')
                }
            
            # 유저의 Top-3 장르와 점수 (TF-IDF 기반)
            user_top_genres_with_scores = {}
            for user_id, user_data in train_with_meta.groupby('user'):
                genre_scores = defaultdict(float)
                for genres_str in user_data['genres'].dropna():
                    if isinstance(genres_str, str):
                        for genre_pair in genres_str.split('|'):
                            if ':' in genre_pair:
                                genre = genre_pair.split(':')[0]
                                score = float(genre_pair.split(':')[1])
                                genre_scores[genre] += score
                if genre_scores:
                    # Top-3 장르와 점수
                    top3 = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    user_top_genres_with_scores[user_id] = dict(top3)
                else:
                    user_top_genres_with_scores[user_id] = {}
            
            def calculate_genre_similarity(row):
                """유저가 좋아하는 장르와 아이템 장르의 유사도"""
                user_id = row['user_id']
                item_id = row['item_id']
                
                # 유저의 Top-3 장르와 점수
                user_genres = user_top_genres_with_scores.get(user_id, {})
                if not user_genres:
                    return 0.0
                
                # 아이템의 장르 정보
                item_genres_str = meta_dict.get(item_id, {}).get('genres', '')
                if pd.isna(item_genres_str) or item_genres_str == '':
                    return 0.0
                
                # 아이템 장르 TF-IDF 점수 파싱
                item_genre_scores = {}
                for genre_pair in str(item_genres_str).split('|'):
                    if ':' in genre_pair:
                        genre, score = genre_pair.split(':')
                        item_genre_scores[genre] = float(score)
                
                # 유저 Top-3 장르와 아이템 장르의 교집합 점수 합산
                similarity = 0.0
                for genre, user_score in user_genres.items():
                    if genre in item_genre_scores:
                        similarity += user_score * item_genre_scores[genre]
                
                return similarity
            
            df['user_item_genre_similarity'] = df.apply(calculate_genre_similarity, axis=1).astype('float32')
            
            # ⚠️ 순서 피처: user_last_item_similarity (유저 마지막 본 아이템과 후보 아이템의 장르 유사도)
            # "방금 로맨스를 본 유저는 다음에도 로맨스를 볼 가능성이 높다"는 순서의 논리
            # ⚠️ 데이터 누수 방지: train_df는 이미 train_split이므로 validation 데이터는 포함되지 않음
            if 'time' in train_df.columns:
                # 유저별 마지막 본 아이템 추출 (시간 순서 기준)
                # train_df는 train_split이므로 validation 데이터는 포함되지 않음 (데이터 누수 없음)
                train_sorted = train_df.sort_values(['user', 'time'])
                user_last_item = train_sorted.groupby('user').tail(1)[['user', 'item']].set_index('user')['item'].to_dict()
                
                # 마지막 본 아이템의 장르 정보
                last_item_genres = {}
                for user_id, last_item_id in user_last_item.items():
                    if last_item_id in meta_dict:
                        genres_str = meta_dict[last_item_id].get('genres', '')
                        if pd.notna(genres_str) and isinstance(genres_str, str):
                            genre_scores = {}
                            for genre_pair in str(genres_str).split('|'):
                                if ':' in genre_pair:
                                    genre, score = genre_pair.split(':')
                                    genre_scores[genre] = float(score)
                            last_item_genres[user_id] = genre_scores
                        else:
                            last_item_genres[user_id] = {}
                    else:
                        last_item_genres[user_id] = {}
                
                def calculate_last_item_similarity(row):
                    """유저가 마지막에 본 아이템과 후보 아이템의 장르 유사도"""
                    user_id = row['user_id']
                    item_id = row['item_id']
                    
                    # 유저의 마지막 본 아이템 장르
                    last_genres = last_item_genres.get(user_id, {})
                    if not last_genres:
                        return 0.0
                    
                    # 후보 아이템의 장르 정보
                    item_genres_str = meta_dict.get(item_id, {}).get('genres', '')
                    if pd.isna(item_genres_str) or item_genres_str == '':
                        return 0.0
                    
                    # 후보 아이템 장르 TF-IDF 점수 파싱
                    item_genre_scores = {}
                    for genre_pair in str(item_genres_str).split('|'):
                        if ':' in genre_pair:
                            genre, score = genre_pair.split(':')
                            item_genre_scores[genre] = float(score)
                    
                    # 코사인 유사도 계산
                    intersection = set(last_genres.keys()) & set(item_genre_scores.keys())
                    numerator = sum([last_genres[g] * item_genre_scores[g] for g in intersection])
                    
                    sum_sq_last = sum([v**2 for v in last_genres.values()])
                    sum_sq_item = sum([v**2 for v in item_genre_scores.values()])
                    
                    denominator = np.sqrt(sum_sq_last) * np.sqrt(sum_sq_item)
                    
                    if denominator == 0:
                        return 0.0
                    return numerator / denominator
                
                df['user_last_item_similarity'] = df.apply(calculate_last_item_similarity, axis=1).astype('float32')
                
                # ⚠️ 순서 정보 재정의: user_last_item_id 대신 user_last_item_genre 사용
                # ID는 카디널리티가 너무 높아 패턴을 찾지 못함 → 장르로 대체
                # 마지막 본 아이템의 장르 정보 추출
                user_last_item_genre = {}
                for user_id, last_item_id in user_last_item.items():
                    if last_item_id in meta_dict:
                        genres_str = meta_dict[last_item_id].get('genres', '')
                        if pd.notna(genres_str) and isinstance(genres_str, str):
                            # Top-1 장르 추출
                            genre_scores = {}
                            for genre_pair in str(genres_str).split('|'):
                                if ':' in genre_pair:
                                    genre = genre_pair.split(':')[0]
                                    score = float(genre_pair.split(':')[1])
                                    if genre not in genre_scores or score > genre_scores[genre]:
                                        genre_scores[genre] = score
                            if genre_scores:
                                top_genre = max(genre_scores.items(), key=lambda x: x[1])[0]
                                user_last_item_genre[user_id] = top_genre
                            else:
                                user_last_item_genre[user_id] = 'Unknown'
                        else:
                            user_last_item_genre[user_id] = 'Unknown'
                    else:
                        user_last_item_genre[user_id] = 'Unknown'
                
                # user_last_item_genre를 범주형 피처로 추가
                df['user_last_item_genre'] = df['user_id'].map(user_last_item_genre).fillna('Unknown').astype('category')
                
                # ⚠️ 순서 피처 강화: (유사도 + 마지막 아이템의 인기도) 결합
                # 마지막에 본 게 인기작일수록 가점
                item_popularity_dict = train_df['item'].value_counts().to_dict()
                if len(item_popularity_dict) > 0:
                    # 마지막 아이템의 인기도 (log 스케일)
                    last_item_popularity = {}
                    for user_id, last_item_id in user_last_item.items():
                        if last_item_id in item_popularity_dict:
                            last_item_popularity[user_id] = np.log1p(item_popularity_dict[last_item_id])
                        else:
                            last_item_popularity[user_id] = 0.0
                    df['user_last_item_popularity'] = df['user_id'].map(last_item_popularity).fillna(0.0).astype('float32')
                    
                    # 상호작용: 유사도 * (1 + 마지막 아이템 인기도)
                    df['user_last_item_similarity_boosted'] = (
                        df['user_last_item_similarity'] * (1.0 + df['user_last_item_popularity'])
                    ).astype('float32')
                
                if verbose:
                    print(f"    ✅ user_last_item_similarity 추가 (순서 피처: 마지막 본 아이템과의 유사도)")
                    print(f"    ✅ user_last_item_genre 추가 (범주형: 마지막 본 아이템의 장르)")
                    if 'user_last_item_similarity_boosted' in df.columns:
                        print(f"    ✅ user_last_item_similarity_boosted 추가 (유사도 + 인기도 결합)")
            else:
                df['user_last_item_similarity'] = 0.0
                df['user_last_item_genre'] = 'Unknown'
                df['user_last_item_genre'] = df['user_last_item_genre'].astype('category')
                if verbose:
                    print(f"    ⚠️  time 컬럼 없음: 순서 피처 스킵")
            
            # ⚠️ 피처 다이어트: user_director_match 제거 (중요도 0.47%)
            # user_writer_match는 유지 (1.77%)
            if verbose:
                print(f"    ✅ user_writer_match 계산 완료")
                print(f"    ✅ user_item_genre_similarity 추가 (메타데이터 강화)")
                print(f"    ⚠️  user_director_match 삭제 (중요도 0.47%)")
                print(f"    ⚠️  user_genre_match 삭제 (user_item_genre_similarity로 대체)")
        except Exception as e:
            if verbose:
                print(f"    ⚠️  감독/작가 매칭 계산 실패: {e}")
            df['user_writer_match'] = False
    else:
        df['user_writer_match'] = False
    
    # ========================================
    # 핵심 엔진: contextual_sasrec_score (P0)
    # ========================================
    # SASRec의 시퀀스 점수를 유저의 마지막 맥락 유사도로 가중치 부여
    # → SASRec이 추천한 아이템이 "유저의 최근 취향과도 맞는지" 검증
    # ⚠️ 중요: 이 피처를 먼저 생성하여 CatBoost가 초기 학습부터 인식하도록 함
    if 'sasrec_score_ratio_log' in df.columns and 'user_last_item_similarity' in df.columns:
        df['contextual_sasrec_score'] = (
            df['sasrec_score_ratio_log'] * df['user_last_item_similarity']
        ).astype('float32')
        # 강제 부스팅: 값이 0인 경우 작은 값으로 대체하여 변별력 확보
        df.loc[df['contextual_sasrec_score'] == 0, 'contextual_sasrec_score'] = 1e-6
        if verbose:
            print(f"    ✅ contextual_sasrec_score 추가 (SASRec * 맥락 유사도, 강제 활성화)")
    
    # ==================== 여기부터 추가 (Shot 2: EDA-Based Features) ====================
    if metadata_path:
        if verbose:
            print("\n   [EDA] 메타데이터 기반 명시적 피처 생성 시작...")
        
        try:
            # 메타데이터 로드 (무조건 새로 로드)
            meta_df = pd.read_csv(metadata_path)
            if verbose:
                print(f"      ✅ 메타데이터 로드: {len(meta_df):,}개 아이템")
            
            # Release Year 처리
            if 'release_year' in meta_df.columns:
                meta_df['item_release_year'] = pd.to_numeric(meta_df['release_year'], errors='coerce').fillna(2000).astype('int16')
            elif 'item_release_year' in meta_df.columns:
                meta_df['item_release_year'] = pd.to_numeric(meta_df['item_release_year'], errors='coerce').fillna(2000).astype('int16')
            else:
                meta_df['item_release_year'] = 2000
            
            # Train에 메타데이터 결합
            train_with_meta = train_df.merge(
                meta_df[['item', 'director', 'writer', 'item_release_year']], 
                on='item',
                how='left'
            )
            
            if verbose:
                print(f"      ✅ Train 메타데이터 결합: {len(train_with_meta):,}개 행")
            
            # ========================================
            # 1. Density/Ratio Features (취향 농도)
            # ========================================
            if verbose:
                print("      [1/3] Density Features (Writer/Director Ratio)...")
            
            # ⚠️ 컬럼명 확인: train_df의 user 컬럼이 'user'인지 'user_id'인지
            user_col = 'user' if 'user' in train_df.columns else 'user_id'
            
            # 유저별 총 시청 수
            user_total_count = train_df.groupby(user_col).size().reset_index(name='total_count')
            user_total_count.columns = ['user_id', 'total_count']
            
            # ⚠️ 중요: df에 item_writer, item_director 컬럼이 있는지 확인
            if 'item_writer' not in df.columns or 'item_director' not in df.columns:
                if verbose:
                    print(f"      ⚠️  item_writer 또는 item_director 컬럼이 없음 → Density 피처 스킵")
            else:
                # Writer Ratio
                user_writer_counts = train_with_meta.groupby([user_col, 'writer']).size().reset_index(name='writer_count')
                user_writer_counts.columns = ['user_id', 'item_writer', 'writer_count']
                user_writer_counts = user_writer_counts.merge(user_total_count, on='user_id')
                user_writer_counts['user_writer_ratio'] = (user_writer_counts['writer_count'] / user_writer_counts['total_count']).astype('float32')
                
                df = df.merge(user_writer_counts[['user_id', 'item_writer', 'user_writer_ratio']], 
                             on=['user_id', 'item_writer'], how='left')
                df['user_writer_ratio'] = df['user_writer_ratio'].fillna(0.0).astype('float32')
                
                # Director Ratio
                user_director_counts = train_with_meta.groupby([user_col, 'director']).size().reset_index(name='director_count')
                user_director_counts.columns = ['user_id', 'item_director', 'director_count']
                user_director_counts = user_director_counts.merge(user_total_count, on='user_id')
                user_director_counts['user_director_ratio'] = (user_director_counts['director_count'] / user_director_counts['total_count']).astype('float32')
                
                df = df.merge(user_director_counts[['user_id', 'item_director', 'user_director_ratio']], 
                             on=['user_id', 'item_director'], how='left')
                df['user_director_ratio'] = df['user_director_ratio'].fillna(0.0).astype('float32')
                
                if verbose:
                    print(f"         ✅ Writer Ratio: mean={df['user_writer_ratio'].mean():.4f}, max={df['user_writer_ratio'].max():.4f}")
                    print(f"         ✅ Director Ratio: mean={df['user_director_ratio'].mean():.4f}, max={df['user_director_ratio'].max():.4f}")
            
            # ========================================
            # 2. Temporal Features (시간적 취향)
            # ========================================
            if verbose:
                print("      [2/3] Temporal Features (Release Year Preference)...")
            
            # 유저별 평균 시청 연도
            user_avg_year = train_with_meta.groupby(user_col)['item_release_year'].mean().reset_index()
            user_avg_year.columns = ['user_id', 'user_avg_release_year']
            user_avg_year['user_avg_release_year'] = user_avg_year['user_avg_release_year'].astype('float32')
            
            df = df.merge(user_avg_year, on='user_id', how='left')
            df['user_avg_release_year'] = df['user_avg_release_year'].fillna(2000.0).astype('float32')
            
            # 후보 아이템의 개봉 연도 (이미 존재할 수 있음)
            if 'item_release_year' not in df.columns:
                item_year_df = meta_df[['item', 'item_release_year']].copy()
                item_year_df.columns = ['item_id', 'item_release_year']
                df = df.merge(item_year_df, on='item_id', how='left')
                df['item_release_year'] = df['item_release_year'].fillna(2000).astype('int16')
            
            # 연도 차이 (절대값)
            df['release_year_diff'] = np.abs(df['item_release_year'].astype('float32') - df['user_avg_release_year']).astype('float32')
            
            # 유저의 최신작 선호도 (최근 10개)
            if 'time' in train_df.columns:
                train_sorted = train_with_meta.sort_values([user_col, 'time'])
                user_recent_pref = train_sorted.groupby(user_col).tail(10).groupby(user_col)['item_release_year'].mean().reset_index()
                user_recent_pref.columns = ['user_id', 'user_recency_preference']
                user_recent_pref['user_recency_preference'] = user_recent_pref['user_recency_preference'].astype('float32')
                
                df = df.merge(user_recent_pref, on='user_id', how='left')
                df['user_recency_preference'] = df['user_recency_preference'].fillna(df['user_avg_release_year']).astype('float32')
                
                if verbose:
                    print(f"         ✅ Release Year Diff: mean={df['release_year_diff'].mean():.2f}")
                    print(f"         ✅ Recency Preference: mean={df['user_recency_preference'].mean():.2f}")
            else:
                df['user_recency_preference'] = df['user_avg_release_year']
                if verbose:
                    print(f"         ⚠️  time 컬럼 없음: recency_preference = avg_year")
            
            # ========================================
            # 3. Diversity Features (탐색 성향)
            # ========================================
            if verbose:
                print("      [3/3] Diversity Features (Exploration Tendency)...")
            
            # 유저별 유니크 작가/감독 수
            user_unique_writers = train_with_meta.groupby(user_col)['writer'].nunique().reset_index()
            user_unique_writers.columns = ['user_id', 'user_diversity_writers']
            user_unique_writers['user_diversity_writers'] = user_unique_writers['user_diversity_writers'].astype('int16')
            
            user_unique_directors = train_with_meta.groupby(user_col)['director'].nunique().reset_index()
            user_unique_directors.columns = ['user_id', 'user_diversity_directors']
            user_unique_directors['user_diversity_directors'] = user_unique_directors['user_diversity_directors'].astype('int16')
            
            df = df.merge(user_unique_writers, on='user_id', how='left')
            df = df.merge(user_unique_directors, on='user_id', how='left')
            df['user_diversity_writers'] = df['user_diversity_writers'].fillna(1).astype('int16')
            df['user_diversity_directors'] = df['user_diversity_directors'].fillna(1).astype('int16')
            
            # 종합 다양성 점수
            df['user_diversity_score'] = (df['user_diversity_writers'] + df['user_diversity_directors']).astype('int16')
            
            # Mainstream Ratio: 벡터화된 계산
            item_col = 'item' if 'item' in train_df.columns else 'item_id'
            item_popularity = train_df[item_col].value_counts()
            threshold_80 = item_popularity.quantile(0.80)
            mainstream_items = set(item_popularity[item_popularity >= threshold_80].index)
            
            # 유저별 mainstream 아이템 비율 계산
            train_df_temp = train_df.copy()
            train_df_temp['is_mainstream'] = train_df_temp[item_col].isin(mainstream_items).astype('int8')
            user_mainstream = train_df_temp.groupby(user_col)['is_mainstream'].mean().reset_index()
            user_mainstream.columns = ['user_id', 'user_mainstream_ratio']
            user_mainstream['user_mainstream_ratio'] = user_mainstream['user_mainstream_ratio'].astype('float32')
            
            df = df.merge(user_mainstream, on='user_id', how='left')
            df['user_mainstream_ratio'] = df['user_mainstream_ratio'].fillna(0.5).astype('float32')
            
            if verbose:
                print(f"         ✅ Diversity Score: mean={df['user_diversity_score'].mean():.2f}")
                print(f"         ✅ Mainstream Ratio: mean={df['user_mainstream_ratio'].mean():.4f}")
            
            if verbose:
                print(f"    ✅ [EDA] 모든 명시적 피처 생성 완료!")
                print(f"       → Density: writer_ratio, director_ratio")
                print(f"       → Temporal: year_diff, recency_preference")
                print(f"       → Diversity: diversity_score, mainstream_ratio")
        
        except Exception as e:
            # ⚠️ 무조건 출력 (디버깅을 위해)
            print(f"\n❌❌❌ [EDA] 피처 생성 실패! ❌❌❌")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("❌❌❌ [EDA] 에러 끝 ❌❌❌\n")
    # ==================== EDA Features 끝 ====================
    
    if verbose:
        print(f"    ✅ Interaction 피처 추가 완료")
    
    return df


def process_rare_metadata(df: pd.DataFrame,
                         metadata_cols: List[str],
                         rare_threshold: int = 3,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Rare 메타데이터 처리 (제3자 제안: 하위 10~20%, 3회 미만 등장).
    
    Args:
        df: 피처 엔지니어링된 DataFrame
        metadata_cols: 메타데이터 컬럼 리스트 (예: ['item_director', 'item_writer'])
        rare_threshold: Rare 기준 (기본값: 3회 미만)
        verbose: 상세 출력 여부
    
    Returns:
        DataFrame with Rare 그룹 처리된 메타데이터
    """
    df = df.copy()
    
    if verbose:
        print(f"  Rare 메타데이터 처리 중... (기준: {rare_threshold}회 미만)")
    
    for col in metadata_cols:
        if col not in df.columns:
            continue
        
        # 빈도수 계산
        value_counts = df[col].value_counts()
        
        # Rare 그룹 식별 (rare_threshold 미만 등장)
        rare_values = value_counts[value_counts < rare_threshold].index.tolist()
        
        if len(rare_values) > 0:
            # 'Rare' 그룹으로 묶기
            # ⚠️ CategoricalDtype의 경우 replace 대신 cat 메서드 사용 (FutureWarning 방지, Pandas 권장 방식)
            if df[col].dtype.name == 'category':
                # 카테고리 타입인 경우: cat.add_categories + 직접 할당 사용 (Pandas 권장)
                categories = df[col].cat.categories.tolist()
                # Rare가 카테고리에 없으면 추가
                if 'Rare' not in categories:
                    df[col] = df[col].cat.add_categories(['Rare'])
                # Rare 값들을 'Rare'로 변경 (직접 할당, replace 대신)
                df.loc[df[col].isin(rare_values), col] = 'Rare'
                # 사용하지 않는 카테고리 제거 (메모리 최적화)
                df[col] = df[col].cat.remove_unused_categories()
            else:
                # 일반 타입인 경우: replace 사용
                df[col] = df[col].replace(rare_values, 'Rare')
            
            if verbose:
                print(f"    {col}: {len(rare_values):,}개 → 'Rare' 그룹으로 묶음")
                print(f"      (전체: {value_counts.sum():,}개 중 {len(rare_values):,}개)")
    
    if verbose:
        print(f"    ✅ Rare 그룹 처리 완료")
    
    return df


def create_ranker_dataset(scores_df: pd.DataFrame,
                         val_ground_truth: Dict[int, Set[int]],
                         train_df: pd.DataFrame,
                         metadata_path: Optional[str] = None,
                         rare_threshold: int = 3,
                         verbose: bool = True) -> pd.DataFrame:
    """
    CatBoost Ranker 학습용 데이터셋 생성.
    
    Args:
        scores_df: Cross-Score DataFrame (user_id, item_id, ease_score, ease_rank)
        val_ground_truth: Validation ground truth {user_id: {item1, item2, ...}}
        train_df: Train 데이터프레임 (통계 계산용)
        metadata_path: 메타데이터 CSV 파일 경로 (선택)
        rare_threshold: Rare 메타데이터 기준 (기본값: 3회 미만)
        verbose: 상세 출력 여부
    
    Returns:
        DataFrame: CatBoost Ranker 학습용 데이터셋
    """
    if verbose:
        print("  [1/7] EASE 피처...", end='', flush=True)
    
    # 1. EASE 피처 추가
    df = add_ease_features(scores_df, verbose=False)
    
    if verbose:
        print(f" 완료 ({len(df.columns)}개 컬럼)")
        print("  [2/7] Item 피처...", end='', flush=True)
    
    # 2. Item 피처 추가
    df = add_item_features(df, train_df, metadata_path, verbose=False)
    
    if verbose:
        print(f" 완료 ({len(df.columns)}개 컬럼)")
        print("  [3/7] User 피처...", end='', flush=True)
    
    # 3. User 피처 추가
    df = add_user_features(df, train_df, metadata_path, verbose=False)
    
    if verbose:
        print(f" 완료 ({len(df.columns)}개 컬럼)")
        print("  [4/7] Interaction 피처...", end='', flush=True)
    
    # 4. Interaction 피처 추가
    df = add_interaction_features(df, train_df, metadata_path, verbose=False)
    
    if verbose:
        print(f" 완료 ({len(df.columns)}개 컬럼)")
    
    # 4-1. 상호작용 피처 추가 (모든 피처가 추가된 후)
    if verbose:
        print("  [5/7] 상호작용 피처...", end='', flush=True)
    
    # ⚠️ 상호작용 피처: item_popularity_recent_ratio * ease_score_ratio_log
    # 트렌드와 EASE 점수의 조합으로 메타데이터 신호 강화
    if 'item_popularity_recent_ratio' in df.columns and 'ease_score_ratio_log' in df.columns:
        df['trend_ease_interaction'] = (
            df['item_popularity_recent_ratio'] * df['ease_score_ratio_log']
        ).astype('float32')
    
    if verbose:
        print(f" 완료 ({len(df.columns)}개 컬럼)")
        print("  [6/7] 레이블링...", end='', flush=True)
    
    # 5. 레이블링 (Validation 기준)
    
    # 최적화: apply 대신 벡터화된 연산 사용 (O(n) → O(n) but 훨씬 빠름)
    # ground_truth를 DataFrame으로 변환하여 merge
    gt_list = []
    for user_id, items in val_ground_truth.items():
        for item_id in items:
            gt_list.append({'user_id': user_id, 'item_id': item_id})
    
    if gt_list:
        gt_df = pd.DataFrame(gt_list)
        gt_df['label'] = 1
        df = df.merge(gt_df, on=['user_id', 'item_id'], how='left')
        df['label'] = df['label'].fillna(0).astype('int8')
    else:
        df['label'] = 0
        df['label'] = df['label'].astype('int8')
    
    if verbose:
        print(f" 완료")
        print("  [7/7] 데이터 타입 최적화...", end='', flush=True)
    
    # 6. Rare 메타데이터 처리
    metadata_cols = ['item_director', 'item_writer']
    df = process_rare_metadata(df, metadata_cols, rare_threshold, verbose=False)
    
    # 7. 데이터 타입 최적화
    
    # 이미 최적화된 타입은 유지, 추가 최적화
    # float64 → float32
    for col in df.select_dtypes(include=['float64']).columns:
        if col not in ['ease_score', 'ease_percentile', 'ease_score_gap', 
                       'ease_score_gap_ratio', 'ease_score_ratio']:
            df[col] = df[col].astype('float32')
    
    # int64 → int32 또는 int16
    for col in df.select_dtypes(include=['int64']).columns:
        max_val = df[col].max()
        if max_val < 32767:
            df[col] = df[col].astype('int16')
        else:
            df[col] = df[col].astype('int32')
    
    if verbose:
        print(f" 완료 (메모리: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB)")
    
    # 8. group_id 추가 (CatBoost Ranker 필수)
    
    # ⚠️ 중요: scores_df에 이미 group_id가 있으면 유지, 없으면 user_id를 group_id로 사용
    if 'group_id' not in df.columns:
        # user_id를 group_id로 사용 (CatBoost Ranker는 group_id로 그룹화)
        df['group_id'] = df['user_id'].astype('int32')
    else:
        # 이미 group_id가 있으면 유지 (유저 그룹 기반 OOF에서 생성된 group_id)
        df['group_id'] = df['group_id'].astype('int32')  # CatBoost용 group_id는 별도로 저장
        # group_id는 유저 그룹 ID (1, 2, ...), group은 CatBoost용 (user_id)
        df['group'] = df['user_id'].astype('int32')
    
    # 9. 학습 데이터셋 구성 (제3자 제안: 정답 포함 유저 우선)
    # ⚠️ Test 유저의 경우: val_ground_truth가 비어있으므로 모든 후보를 그대로 반환
    if len(val_ground_truth) == 0:
        # Test 유저: 모든 후보를 그대로 반환
        return df
    
    # 학습/Validation 유저: 정답 포함 유저 우선 샘플링
    
    # 정답 포함 유저 그룹
    users_with_label = df.groupby('user_id')['label'].sum()
    users_with_positive = users_with_label[users_with_label > 0].index.tolist()
    users_without_positive = users_with_label[users_with_label == 0].index.tolist()
    
    # 정답 포함 유저 우선, 정답 없는 유저는 샘플링하여 추가
    df_positive = df[df['user_id'].isin(users_with_positive)].copy()
    df_negative = df[df['user_id'].isin(users_without_positive)].copy()
    
    # 정답 없는 유저는 일부만 샘플링 (음성 샘플로 활용)
    if len(df_negative) > 0:
        # 정답 포함 유저 수의 50%만 샘플링
        n_negative_users = min(len(users_without_positive), len(users_with_positive) // 2) if len(users_with_positive) > 0 else len(users_without_positive)
        if n_negative_users > 0:
            sampled_negative_users = np.random.choice(
                users_without_positive, 
                size=n_negative_users, 
                replace=False
            )
            df_negative = df_negative[df_negative['user_id'].isin(sampled_negative_users)]
        else:
            df_negative = pd.DataFrame()  # 빈 DataFrame
    
    final_df = pd.concat([df_positive, df_negative], ignore_index=True)
    
    return final_df


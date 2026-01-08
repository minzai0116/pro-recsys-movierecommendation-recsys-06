"""EASE (Embarrassingly Shallow Autoencoders) 모델."""
import numpy as np
from scipy import sparse
from typing import List, Dict, Optional, Set
import time
from tqdm import tqdm


class EASE:
    """
    EASE 모델 구현 (정확한 수식 적용).
    
    논문: "Embarrassingly Shallow Autoencoders for Sparse Data"
    https://arxiv.org/abs/1905.03375
    
    정확한 공식:
        P = (X^T X + λI)^(-1)
        B_ij = -P_ij / P_jj  (i != j)
        B_ii = 0
    """
    
    def __init__(self, lambda_reg: float = 100.0):
        """
        Args:
            lambda_reg: 정규화 파라미터 (높을수록 더 단순한 모델)
        """
        self.lambda_reg = lambda_reg
        self.B = None  # Item-Item similarity matrix
        self.item_id_to_idx = None
        self.idx_to_item_id = None
        self.n_items = None
    
    def fit(self, 
            user_item_matrix: sparse.csr_matrix,
            item_id_to_idx: Dict[int, int],
            precomputed_G: Optional[np.ndarray] = None,
            verbose: bool = True) -> None:
        """
        EASE 모델 학습 (정확한 수식 적용).
        
        Args:
            user_item_matrix: scipy.sparse.csr_matrix (shape: [n_users, n_items])
            item_id_to_idx: {item_id: matrix_index}
            precomputed_G: 미리 계산된 Gram 행렬 (X^T X) - 최적화용, None이면 새로 계산
            verbose: 진행 상황 출력
        """
        if verbose:
            print("=" * 60)
            print("EASE 모델 학습 시작 (정확한 수식 적용)")
            print("=" * 60)
            print(f"User-Item 행렬 크기: {user_item_matrix.shape}")
            print(f"정규화 파라미터 (λ): {self.lambda_reg}")
        
        start_time = time.time()
        
        self.item_id_to_idx = item_id_to_idx
        self.idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
        self.n_items = user_item_matrix.shape[1]
        
        # 1. X^T X 계산 (Item-Item Gram matrix)
        # 최적화: precomputed_G가 제공되면 재사용 (하이퍼파라미터 최적화 시 시간 절약)
        if precomputed_G is not None:
            if verbose:
                print("\n[1/3] 미리 계산된 X^T X 사용 (최적화 모드)")
            G = precomputed_G.copy()  # 원본 보존을 위해 copy
        else:
            if verbose:
                print("\n[1/3] X^T X 계산 중...")
            X = user_item_matrix.astype(np.float32)  # 메모리 절약
            G = (X.T @ X).toarray()  # [n_items, n_items] - dense로 변환
        
        if verbose:
            print(f"  X^T X 크기: {G.shape}")
            print(f"  희소도: {(1 - np.count_nonzero(G) / (G.shape[0] * G.shape[1])) * 100:.2f}%")
        
        # 2. P = (X^T X + λI)^(-1) 계산
        if verbose:
            print("\n[2/3] P = (X^T X + λI)^(-1) 계산 중...")
        
        # 대각선에 lambda_reg 추가
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.lambda_reg
        
        # 역행렬 계산
        if G.shape[0] < 10000:
            # 작은/중간 행렬: 직접 역행렬 계산
            P = np.linalg.inv(G)
        else:
            # 큰 행렬: float32로 메모리 절약
            P = np.linalg.inv(G.astype(np.float32)).astype(np.float32)
        
        # 3. B = -P / diag(P) 공식 적용 (논문 정석 공식)
        # B_ij = -P_ij / P_jj (i != j), B_ii = 0
        if verbose:
            print("\n[3/3] B = -P / diag(P) 계산 중... (정확한 EASE 공식)")
        
        # diag(P) 계산
        diag_P = np.diag(P)
        
        # B = -P / diag(P) (broadcasting)
        # 주의: P_jj로 나누므로 각 열을 해당 대각선 원소로 나눔
        B = -P / diag_P[np.newaxis, :]  # [n_items, n_items] / [1, n_items] -> broadcasting
        
        # 대각선을 0으로 설정
        B[diag_indices] = 0.0
        
        self.B = B.astype(np.float32)  # 메모리 절약
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ 학습 완료! (소요 시간: {elapsed_time:.2f}초)")
            print(f"  가중치 행렬 B 크기: {self.B.shape}")
            print(f"  메모리 사용량: {self.B.nbytes / 1024 / 1024:.2f} MB")
            print(f"  데이터 타입: {self.B.dtype}")
    
    def predict_batch_vectorized(self,
                                user_item_matrix: sparse.csr_matrix,
                                user_id_to_idx: Dict[int, int],
                                k: int = 10,
                                verbose: bool = True) -> Dict[int, List[int]]:
        """
        전체 유저에 대해 벡터화된 배치 예측 (매우 빠름).
        
        Args:
            user_item_matrix: 전체 user-item 행렬
            user_id_to_idx: {user_id: matrix_index}
            k: Top-K
            verbose: 진행 상황 출력
        
        Returns:
            {user_id: [item1, item2, ...]}
        """
        if verbose:
            print("  벡터화된 배치 예측 중...")
        
        # 1. 전역 행렬 곱: [Users, Items] @ [Items, Items] -> [Users, Items]
        # X가 희소 행렬이므로 효율적으로 작동
        if verbose:
            print("    행렬 곱셈 중...")
        scores = user_item_matrix @ self.B  # [n_users, n_items]
        
        # sparse matrix인 경우 dense로 변환
        if sparse.issparse(scores):
            scores = scores.toarray()
        
        if verbose:
            print(f"    점수 행렬 크기: {scores.shape}")
        
        # 2. 이미 본 아이템 점수 마스킹
        # user_item_matrix가 1인 곳(이미 본 곳)을 -inf로
        if verbose:
            print("    본 아이템 마스킹 중...")
        rows, cols = user_item_matrix.nonzero()
        scores[rows, cols] = -np.inf
        
        # 3. Top-K 추출 (Batch wise)
        # numpy의 argpartition은 정렬보다 빠름
        if verbose:
            print("    Top-K 추출 중...")
        
        n_users = scores.shape[0]
        partitioned_indices = np.argpartition(-scores, k, axis=1)[:, :k]
        
        # 정렬하여 최종 반환
        if verbose:
            print("    결과 변환 중...")
        
        predictions = {}
        user_ids = list(user_id_to_idx.keys())
        user_indices = [user_id_to_idx[uid] for uid in user_ids]
        
        for i, user_idx in enumerate(tqdm(user_indices, desc="Extracting top-k", leave=False) if verbose else user_indices):
            row_indices = partitioned_indices[user_idx]
            # 점수 순으로 정렬
            row_scores = scores[user_idx, row_indices]
            sorted_local_idx = np.argsort(-row_scores)
            sorted_indices = row_indices[sorted_local_idx]
            
            # 인덱스 -> 아이템 ID 변환
            pred_items = [self.idx_to_item_id[idx] for idx in sorted_indices if scores[user_idx, idx] > -np.inf]
            predictions[user_ids[i]] = pred_items[:k]
        
        if verbose:
            print(f"    ✅ 예측 완료! (총 {len(predictions):,}명)")
        
        return predictions
    
    def predict(self, 
                user_item_vector: sparse.csr_matrix,
                k: int = 10,
                exclude_items: Optional[Set[int]] = None) -> List[int]:
        """
        단일 사용자에 대한 예측 (하이퍼파라미터 최적화용).
        
        주의: 메인 예측에서는 predict_batch_vectorized를 사용합니다.
        이 메서드는 hyperparameter_tuning.py에서만 사용됩니다.
        
        Args:
            user_item_vector: scipy.sparse.csr_matrix (shape: [1, n_items])
            k: Top-K
            exclude_items: 제외할 아이템 ID 집합 (이미 본 아이템 등)
        
        Returns:
            Top-K 아이템 ID 리스트
        """
        # 예측 점수: user_vector @ B
        scores = user_item_vector @ self.B  # [1, n_items]
        
        # sparse matrix인 경우 toarray(), 이미 array면 그대로 사용
        if sparse.issparse(scores):
            scores = scores.toarray().flatten()  # [n_items]
        else:
            scores = scores.flatten()  # [n_items]
        
        # 이미 본 아이템 제외
        seen_items = set(user_item_vector.indices)
        if exclude_items:
            seen_items |= exclude_items
        
        # 제외할 아이템 인덱스
        exclude_indices = set()
        for item_id in seen_items:
            if item_id in self.item_id_to_idx:
                exclude_indices.add(self.item_id_to_idx[item_id])
        
        # 점수 마스킹
        scores[list(exclude_indices)] = -np.inf
        
        # Top-K 선택
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # 인덱스 -> 아이템 ID 변환
        top_k_items = [self.idx_to_item_id[idx] for idx in top_k_indices if scores[idx] > -np.inf]
        
        return top_k_items
    
    def predict_batch(self,
                     user_item_matrix: sparse.csr_matrix,
                     user_ids: List[int],
                     user_id_to_idx: Optional[Dict[int, int]] = None,
                     k: int = 10,
                     batch_size: int = 1000,
                     verbose: bool = False) -> List[List[int]]:
        """
        배치 예측 (metrics.py의 calculate_recall_during_training용).
        
        주의: 메인 예측에서는 predict_batch_vectorized를 사용합니다.
        이 메서드는 metrics.py에서만 사용됩니다.
        
        Args:
            user_item_matrix: 전체 user-item 행렬
            user_ids: 예측할 사용자 ID 리스트
            user_id_to_idx: {user_id: matrix_index} (None이면 user_ids가 이미 인덱스)
            k: Top-K
            batch_size: 배치 크기 (사용 안 함, 호환성 유지)
            verbose: 진행 상황 출력
        
        Returns:
            각 사용자별 Top-K 아이템 리스트
        """
        predictions = []
        
        iterator = tqdm(user_ids, desc="Predicting", leave=False) if verbose else user_ids
        
        for user_id in iterator:
            if user_id_to_idx:
                user_idx = user_id_to_idx[user_id]
            else:
                user_idx = user_id
            
            user_vector = user_item_matrix[user_idx:user_idx+1]
            pred_items = self.predict(user_vector, k=k)
            predictions.append(pred_items)
        
        return predictions
    
    def predict_all_users(self,
                         user_item_matrix: sparse.csr_matrix,
                         user_id_to_idx: Dict[int, int],
                         k: int = 10,
                         batch_size: int = 1000,
                         verbose: bool = True) -> Dict[int, List[int]]:
        """
        모든 사용자에 대한 예측 (벡터화된 버전 사용).
        
        Args:
            user_item_matrix: 전체 user-item 행렬
            user_id_to_idx: {user_id: matrix_index}
            k: Top-K
            batch_size: 배치 크기 (사용 안 함, 호환성 유지)
            verbose: 진행 상황 출력
        
        Returns:
            {user_id: [item1, item2, ...]}
        """
        # 벡터화된 버전 사용 (모든 사용자 한 번에 처리)
        return self.predict_batch_vectorized(
            user_item_matrix,
            user_id_to_idx,
            k=k,
            verbose=verbose
        )

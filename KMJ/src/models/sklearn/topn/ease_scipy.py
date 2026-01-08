from __future__ import annotations

"""
EASE (Embarrassingly Shallow Autoencoders) - SciPy/Numpy implementation for Top-N.

This is a direct port of `KMJ/NewFrame/src/ease.py` core math, adapted to Baseline contracts:
- input: user-item implicit feedback matrix (CSR, binary)
- output: item-item weight matrix B (dense float32)

Reference:
  "Embarrassingly Shallow Autoencoders for Sparse Data" (arXiv:1905.03375)
    P = (X^T X + λI)^(-1)
    B_ij = -P_ij / P_jj   (i != j)
    B_ii = 0
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse


@dataclass
class EASECheckpoint:
    """
    Serializable model state for joblib.
    - B: item-item weights [n_items, n_items], float32
    - mappings: id <-> index
    - popular_items: fallback list of item_id sorted by popularity (desc)
    """

    lambda_reg: float
    B: np.ndarray
    user_id_to_idx: Dict[int, int]
    item_id_to_idx: Dict[int, int]
    idx_to_item_id: Dict[int, int]
    popular_items: List[int]


class EASEScipy:
    def __init__(self, lambda_reg: float = 100.0):
        self.lambda_reg = float(lambda_reg)
        self.B: Optional[np.ndarray] = None

    def fit(self, X: sparse.csr_matrix, *, verbose: bool = True) -> np.ndarray:
        """
        Fit EASE weights.

        Args:
            X: user-item implicit matrix (CSR) [n_users, n_items]
        Returns:
            B: dense float32 array [n_items, n_items]
        """
        if verbose:
            print("=" * 60)
            print("EASE(SciPy) fit start (closed-form)")
            print("=" * 60)
            print(f"X shape: {X.shape}, nnz={X.nnz:,}, lambda={self.lambda_reg}")

        X = X.astype(np.float32)
        # Gram matrix: X^T X (dense)
        G = (X.T @ X)
        if sparse.issparse(G):
            G = G.toarray()
        else:
            G = np.asarray(G)

        # regularize diagonal
        diag_idx = np.diag_indices(G.shape[0])
        G[diag_idx] += self.lambda_reg

        # inverse
        P = np.linalg.inv(G).astype(np.float32)

        # B = -P / diag(P), with B_ii=0
        d = np.diag(P).astype(np.float32)
        B = (-P / d[np.newaxis, :]).astype(np.float32)
        B[diag_idx] = 0.0

        self.B = B
        if verbose:
            mb = (B.nbytes / 1024 / 1024)
            print(f"✅ fit done. B shape={B.shape}, mem={mb:.2f}MB")
        return B

    @staticmethod
    def build_implicit_matrix(
        df,
        *,
        user_col: str = "user",
        item_col: str = "item",
    ) -> Tuple[sparse.csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], List[int]]:
        """
        Build binary user-item CSR and mappings.

        Returns:
            X, user_id_to_idx, item_id_to_idx, idx_to_item_id, popular_item_ids
        """
        users = sorted(df[user_col].unique())
        items = sorted(df[item_col].unique())

        user_id_to_idx = {int(u): i for i, u in enumerate(users)}
        item_id_to_idx = {int(it): i for i, it in enumerate(items)}
        idx_to_item_id = {i: int(it) for it, i in item_id_to_idx.items()}

        rows = df[user_col].map(lambda x: user_id_to_idx[int(x)]).to_numpy()
        cols = df[item_col].map(lambda x: item_id_to_idx[int(x)]).to_numpy()
        data = np.ones(len(df), dtype=np.float32)

        X = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        # binaryize (in case of duplicates)
        X.data[:] = 1.0
        X.eliminate_zeros()

        # popularity fallback (by interaction count)
        vc = df[item_col].value_counts()
        popular_items = [int(x) for x in vc.index.to_list()]

        return X, user_id_to_idx, item_id_to_idx, idx_to_item_id, popular_items

    def recommend_batch(
        self,
        X: sparse.csr_matrix,
        *,
        user_indices: Sequence[int],
        idx_to_item_id: Dict[int, int],
        k: int,
        popular_items: List[int],
    ) -> List[List[int]]:
        """
        Recommend Top-K for a batch of users (indices in X).
        - excludes seen items via X nonzeros masking
        - if not enough candidates, fills by popular_items
        """
        if self.B is None:
            raise ValueError("EASEScipy is not fitted (B is None).")

        if len(user_indices) == 0:
            return []

        Xb = X[user_indices]  # CSR [b, n_items]
        scores = Xb @ self.B  # dense ndarray [b, n_items] (scipy returns ndarray here)
        if sparse.issparse(scores):
            scores = scores.toarray()
        scores = np.asarray(scores, dtype=np.float32)

        # mask seen
        r, c = Xb.nonzero()
        scores[r, c] = -np.inf

        # top-k indices (argpartition is fast)
        k_eff = min(int(k), scores.shape[1])
        part = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]

        out: List[List[int]] = []
        for bi in range(scores.shape[0]):
            cand = part[bi]
            cand_scores = scores[bi, cand]
            order = np.argsort(-cand_scores)
            top_idx = cand[order]

            recs: List[int] = []
            for j in top_idx:
                if scores[bi, j] == -np.inf:
                    continue
                recs.append(idx_to_item_id[int(j)])
                if len(recs) >= k:
                    break

            # fill if needed (popular items, excluding already in recs and seen)
            if len(recs) < k:
                seen = set(Xb[bi].indices.tolist())
                seen_items = {idx_to_item_id[int(ii)] for ii in seen}
                rec_set = set(recs)
                for it in popular_items:
                    if it in rec_set or it in seen_items:
                        continue
                    recs.append(it)
                    rec_set.add(it)
                    if len(recs) >= k:
                        break

            out.append(recs[:k])

        return out



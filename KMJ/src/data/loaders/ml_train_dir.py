from __future__ import annotations

"""
MovieLens 계열 데이터 로더.

입력:
- cfg.dataset.data_path: str
- (선택) cfg.dataset.item2attributes_file: str
- (선택) cfg.dataset.load_aux_tables: bool
- (선택) cfg.dataset.sample_submission_path: str

출력:
- dict[str, Any]
  - ratings: pd.DataFrame (columns: user,item,time)
  - item2attributes: dict|None
  - sample_submission: pd.DataFrame|None (columns: user,item)
  - aux_tables: dict[str, pd.DataFrame]
  - paths: dict[str, str|None]

비고:
- 대회가 sample_submission 템플릿을 “채워서 제출”하는 형태인 경우,
  제출 대상 user/순서는 sample_submission이 SSoT가 됩니다.
"""

from typing import Any, Dict, Optional
import json
import os

import pandas as pd


def load_ml_train_dir(cfg: Any) -> Dict[str, Any]:
    """cfg.dataset.data_path 디렉토리에서 학습/제출에 필요한 파일들을 로드합니다."""
    base = cfg.dataset.data_path
    if not base.endswith(os.sep):
        base = base + os.sep

    ratings_path = os.path.join(base, "train_ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    ratings = pd.read_csv(ratings_path)
    required = ["user", "item", "time"]
    missing = [c for c in required if c not in ratings.columns]
    if missing:
        raise ValueError(f"train_ratings.csv missing columns: {missing}")

    # optional: item2attributes (S3Rec pretrain 등에 사용 가능)
    item2attr_path = os.path.join(base, cfg.dataset.get("item2attributes_file", "Ml_item2attributes.json"))
    item2attributes: Optional[dict] = None
    if os.path.exists(item2attr_path):
        with open(item2attr_path, "r", encoding="utf-8") as f:
            item2attributes = json.load(f)

    aux_paths = {
        "titles": os.path.join(base, "titles.tsv"),
        "years": os.path.join(base, "years.tsv"),
        "genres": os.path.join(base, "genres.tsv"),
        "directors": os.path.join(base, "directors.tsv"),
        "writers": os.path.join(base, "writers.tsv"),
    }

    load_aux = bool(cfg.dataset.get("load_aux_tables", False))
    aux_tables: Dict[str, pd.DataFrame] = {}
    if load_aux:
        # tsv들은 크기가 작으니 필요할 때만 로드
        for k, p in aux_paths.items():
            if os.path.exists(p):
                aux_tables[k] = pd.read_csv(p, sep="\t")

    # optional: sample_submission for competition template (users/order/K)
    sample_path = None
    try:
        sample_path = cfg.dataset.get("sample_submission_path", None)
    except Exception:
        sample_path = None

    # default: sibling eval dir (..../train -> ..../eval/sample_submission.csv)
    if not sample_path:
        base_dir = os.path.abspath(os.path.join(base, os.pardir))
        candidate = os.path.join(base_dir, "eval", "sample_submission.csv")
        if os.path.exists(candidate):
            sample_path = candidate

    sample_submission = None
    if sample_path and os.path.exists(sample_path):
        try:
            sample_submission = pd.read_csv(sample_path)
        except Exception:
            sample_submission = None

    return {
        "ratings": ratings,
        "item2attributes": item2attributes,
        "sample_submission": sample_submission,
        "aux_paths": aux_paths,
        "aux_tables": aux_tables,
        "paths": {
            "ratings_path": ratings_path,
            "item2attributes_path": item2attr_path if os.path.exists(item2attr_path) else None,
            "sample_submission_path": sample_path if (sample_path and os.path.exists(sample_path)) else None,
        },
    }

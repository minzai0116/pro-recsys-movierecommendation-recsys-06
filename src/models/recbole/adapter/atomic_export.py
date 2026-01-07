from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import os
import pandas as pd

from .fieldmap import FieldMap


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    data_root: str
    dataset_dir: str
    inter_path: str
    item_path: Optional[str] = None


def export_inter(*, bundle, out_root: str, dataset: str, fm: FieldMap) -> DatasetSpec:
    """
    out_root/
      {dataset}/
        {dataset}.inter
    """

    dataset_dir = os.path.join(out_root, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    inter_path = os.path.join(dataset_dir, f"{dataset}.inter")

    df: pd.DataFrame = bundle.train

    cols = [fm.user_col, fm.item_col]
    if fm.target_col is not None:
        cols.append(fm.target_col)
    if fm.time_col is not None:
        cols.append(fm.time_col)

    rename_map: Dict[str, str] = {
        fm.user_col: fm.export_user,
        fm.item_col: fm.export_item,
    }
    if fm.target_col is not None:
        rename_map[fm.target_col] = fm.export_target
    if fm.time_col is not None:
        rename_map[fm.time_col] = fm.export_time

    out_df = df[cols].rename(columns=rename_map).copy()

    header_parts = [
        f"{fm.export_user}:token",
        f"{fm.export_item}:token",
    ]
    if fm.target_col is not None:
        header_parts.append(f"{fm.export_target}:float")
    if fm.time_col is not None:
        header_parts.append(f"{fm.export_time}:float")

    with open(inter_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header_parts) + "\n")
        out_df.to_csv(f, sep="\t", index=False, header=False)

    return DatasetSpec(
        dataset=dataset,
        data_root=out_root,
        dataset_dir=dataset_dir,
        inter_path=inter_path,
        item_path=None
    )


def export_item_features(spec: DatasetSpec, item_df: pd.DataFrame, fm: FieldMap) -> DatasetSpec:
    """
    기존 DatasetSpec을 받아 .item 파일을 생성하고 경로가 업데이트된 새 Spec을 반환
    """
    if item_df is None or item_df.empty:
        return spec

    item_path = os.path.join(spec.dataset_dir, f"{spec.dataset}.item")
    
    # RecBole용 헤더 생성 로직 (간소화됨: float는 float, 나머지는 token으로 처리)
    # 실제로는 feature 성격에 따라 token_seq 등을 지정해야 할 수 있음
    header_parts = []
    rename_map = {}
    
    # ID 컬럼 처리
    # item_df의 컬럼 중 fm.item_col("item_id")와 매칭되는 것을 찾아야 함
    # 보통 loader에서 원본 컬럼명을 그대로 들고 있으므로 확인 필요
    # 여기서는 item_df에 fm.item_col이 포함되어 있다고 가정
    
    # RecBole Item ID Field
    rename_map[fm.item_col] = fm.export_item
    header_parts.append(f"{fm.export_item}:token")

    out_cols = [fm.item_col]
    
    seq_cols = ["genre", "genres", "tags"] 

    for col in item_df.columns:
        if col == fm.item_col:
            continue
            
        dtype = item_df[col].dtype
        
        # 1. 컬럼명이 seq_cols에 포함되면 token_seq로 지정
        if col in seq_cols:
            suffix = "token_seq"
        # 2. 실수형은 float
        elif pd.api.types.is_float_dtype(dtype):
            suffix = "float"
        # 3. 나머지는 token
        else:
            suffix = "token"
            
        header_parts.append(f"{col}:{suffix}")
        out_cols.append(col)
        # 컬럼명은 그대로 사용
        rename_map[col] = col

    # DataFrame 재구성
    out_df = item_df[out_cols].rename(columns=rename_map)

    # 파일 쓰기
    with open(item_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header_parts) + "\n")
        out_df.to_csv(f, sep="\t", index=False, header=False)

    # 업데이트된 Spec 반환
    return DatasetSpec(
        dataset=spec.dataset,
        data_root=spec.data_root,
        dataset_dir=spec.dataset_dir,
        inter_path=spec.inter_path,
        item_path=item_path
    )
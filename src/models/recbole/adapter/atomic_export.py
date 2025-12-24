from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .fieldmap import FieldMap


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    data_root: str
    dataset_dir: str
    inter_path: str


def export_inter(*, bundle, out_root: str, dataset: str, fm: FieldMap) -> DatasetSpec:
    """
    out_root/
      {dataset}/
        {dataset}.inter
    """
    import os

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
    )

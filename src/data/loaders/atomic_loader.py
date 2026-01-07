from typing import Any, Dict

import os
import pandas as pd

def load_atomic_files(cfg: Any) -> Dict[str, Any]:
    """
    cfg.dataset.data_path에 있는 .inter, .item 파일을 로드합니다.
    """
    base = cfg.dataset.data_path
    dataset_name = cfg.recbole.dataset
    
    # 1. Load Interaction (.inter)
    inter_path = os.path.join(base, f"{dataset_name}.inter")
    if not os.path.exists(inter_path):
        raise FileNotFoundError(f"Inter file not found: {inter_path}")
    
    # RecBole 포맷은 tab separated
    inter_df = pd.read_csv(inter_path, sep='\t')
    
    # 2. Load Item Features (.item) - 선택사항
    item_path = os.path.join(base, f"{dataset_name}.item")
    item_df = None
    if os.path.exists(item_path):
        item_df = pd.read_csv(item_path, sep='\t')

    return {
        "ratings": inter_df,       # pipeline에서 'ratings'란 키를 기대함
        "item_df": item_df,        # 추가된 item feature
        "sample_submission": None, # 필요하다면 별도 로드
        "paths": {
            "inter": inter_path,
            "item": item_path
        }
    }
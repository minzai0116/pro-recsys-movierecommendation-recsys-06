from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FieldMap:
    # 원본 DataFrame 컬럼명
    user_col: str
    item_col: str
    target_col: Optional[str]
    time_col: Optional[str]

    # export 파일에 기록할 표준 컬럼명 (RecBole config도 이 이름을 참조)
    export_user: str = "user_id"
    export_item: str = "item_id"
    export_target: str = "rating"
    export_time: str = "timestamp"


def build_fieldmap(schema: Dict[str, Any]) -> FieldMap:
    user_col = schema.get("user_col", "user_id")
    item_col = schema.get("item_col", "item_id")
    target_col = schema.get("target_col") or schema.get("rating_col")
    time_col = schema.get("time_col")
    return FieldMap(user_col=user_col, item_col=item_col, target_col=target_col, time_col=time_col)


def recbole_field_overrides(fm: FieldMap) -> Dict[str, Any]:
    """
    RecBole config에서 필드명 관련 설정은 여기서만 만든다.
    export 파일(.inter)은 export_* 이름으로 기록된다는 전제.
    """
    d: Dict[str, Any] = {
        "USER_ID_FIELD": fm.export_user,
        "ITEM_ID_FIELD": fm.export_item,
    }
    if fm.target_col is not None:
        d["RATING_FIELD"] = fm.export_target
    if fm.time_col is not None:
        d["TIME_FIELD"] = fm.export_time
    return d

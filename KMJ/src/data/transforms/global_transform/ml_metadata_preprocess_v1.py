from __future__ import annotations

"""
GlobalTransform: MovieLens style aux tables -> item metadata (+ optional global attribute encoding).

이 모듈은 `KMJ/preprocess.ipynb`에서 하던 메타데이터 전처리를
RecSys_RRS_Baseline의 GlobalTransform 프레임워크로 옮긴 버전입니다.

## Input (raw dict contract)
- raw["ratings"]: pd.DataFrame, columns: ["user", "item", "time"]
- raw["aux_tables"]: dict[str, pd.DataFrame] (optional)
  - "titles": columns: ["item", "title"]
  - "years": columns: ["item", "year"]
  - "genres": columns: ["item", "genre"]  (item별 multi-row)
  - "directors": columns: ["item", "director"] (item별 multi-row)
  - "writers": columns: ["item", "writer"] (item별 multi-row)
- raw["aux_paths"]: dict[str, str] (optional; aux_tables가 없으면 여기에서 로드)

## Output (raw dict augmented)
- raw["item_meta"]: pd.DataFrame
  columns: ["item", "clean_title", "year_bin", "director", "writer", "genres"]
- raw["item2attributes"]: dict[int, list[int]]  (cfg 옵션으로 enable 시 생성/덮어씀)
- raw["attribute_vocab"]: dict[str, int] (token->id)
- raw["attribute_size"]: int (max_id+1, 0 padding reserved)

## Tokenization policy (item2attributes 생성 시)
- director:  "director=<value>"
- writer:    "writer=<value>"
- year_bin:  "year=<bin>"
- genres:    "genre=<each>"  (genres 문자열을 '|'로 split 해서 개별 토큰화)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import re
import pandas as pd

from src.data.transforms.base import GlobalTransform
from src.data.transforms.registry import register_global


def _ensure_aux_tables(raw: Dict[str, Any], *, load: bool = True) -> Dict[str, pd.DataFrame]:
    """Ensure raw["aux_tables"] exists. If missing and load=True, load from raw["aux_paths"]."""
    aux_tables = raw.get("aux_tables", None)
    if isinstance(aux_tables, dict) and aux_tables:
        return aux_tables
    if not load:
        return {}
    aux_paths = raw.get("aux_paths", None) or {}
    if not isinstance(aux_paths, dict) or not aux_paths:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for k, p in aux_paths.items():
        if not p or not isinstance(p, str):
            continue
        if not os.path.exists(p):
            continue
        # tsv tables
        out[k] = pd.read_csv(p, sep="\t")
    raw["aux_tables"] = out
    return out


def _get_popular_entity(entity_df: pd.DataFrame, entity_col: str, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    영화(item)별로 가장 인기 있는(시청수 기반) 감독/작가 1명만 선정.
    동점이면 total_person_views(desc), entity_col(asc) 기준으로 안정 정렬.
    """
    item_popularity = ratings_df.groupby("item").size().reset_index(name="view_count")
    merged = entity_df.merge(item_popularity, on="item", how="left")
    merged["view_count"] = merged["view_count"].fillna(0)

    person_popularity = merged.groupby(entity_col)["view_count"].sum().reset_index(name="total_person_views")
    merged = merged.merge(person_popularity, on=entity_col, how="left")
    merged = merged.sort_values(by=["item", "total_person_views", entity_col], ascending=[True, False, True])

    rep = merged.drop_duplicates(subset=["item"], keep="first")[["item", entity_col]]
    return rep


def _build_genres_rep(genres_df: pd.DataFrame, *, top_n: int = 4, sep: str = "|") -> pd.DataFrame:
    """전체 장르 빈도 순위를 기반으로 item별 상위 N개 장르를 선택해 join."""
    genre_counts = genres_df["genre"].value_counts()
    genre_rank = {genre: i for i, genre in enumerate(genre_counts.index)}

    def filter_top(group: pd.DataFrame) -> str:
        gs = group["genre"].tolist()
        gs.sort(key=lambda x: genre_rank.get(x, 999))
        return sep.join(gs[:top_n])

    rep = genres_df.groupby("item").apply(filter_top).reset_index(name="genres")
    return rep


def _master_title_processor(df: pd.DataFrame) -> pd.DataFrame:
    """
    titles+years merge된 DF에 대해 clean_title / final_year 생성.
    notebook의 master_title_processor 로직을 최대한 그대로 이식.
    """

    def _proc_row(row: pd.Series) -> Tuple[str, Any]:
        title = row.get("title", None)
        year = row.get("year", None)

        if pd.isna(title):
            return "Unknown", year

        # 1) extract year from (1999) like patterns if year is NaN
        m = re.search(r"\((\d{4})[/-]?.*?\)", str(title))
        if m:
            extracted = int(m.group(1))
            if pd.isna(year):
                year = extracted

        # 2) basic cleaning
        t = str(title).strip().strip('"')
        t = re.sub(r"\(a\.k\.a\..*?\)", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*\(\d{4}[/-]?.*?\)", "", t)

        # 3) smart swap (foreign/english)
        m2 = re.search(r"(.*)\s*\((.*)\)", t)
        if m2:
            primary = m2.group(1).strip()
            secondary = m2.group(2).strip()
            english_articles = [", The", ", A", ", An"]
            if any(secondary.endswith(a) for a in english_articles):
                t = secondary
            else:
                t = primary

        # 4) normalize articles: "Matrix, The" -> "The Matrix"
        t = t.strip()
        for article in ["The", "A", "An"]:
            suffix = f", {article}"
            if t.endswith(suffix):
                t = f"{article} {t[:-len(suffix)]}"
                break

        return t.strip(), year

    out = df.copy()
    tmp = out.apply(lambda r: pd.Series(_proc_row(r), index=["clean_title", "final_year"]), axis=1)
    out[["clean_title", "final_year"]] = tmp

    # fill missing years with mode (same as notebook)
    if out["final_year"].notnull().any():
        out["final_year"] = out["final_year"].fillna(out["final_year"].mode()[0])
    else:
        out["final_year"] = out["final_year"].fillna(0)

    # disambiguate duplicated titles by appending year
    title_counts = out["clean_title"].value_counts()
    duplicated = set(title_counts[title_counts > 1].index)

    def _disambiguate(row: pd.Series) -> str:
        t = row["clean_title"]
        if t in duplicated:
            try:
                y = int(row["final_year"]) if pd.notnull(row["final_year"]) else "Unknown"
                return f"{t} ({y})"
            except Exception:
                return f"{t} (Unknown)"
        return t

    out["clean_title"] = out.apply(_disambiguate, axis=1)
    return out


def _bin_year(v: Any) -> str:
    try:
        y = int(v)
    except Exception:
        return "Unknown"
    if y < 1970:
        return "pre_1970"
    if 1990 <= y <= 2009:
        return f"{(y // 5) * 5}"
    return f"{(y // 10) * 10}s"


def _build_item2attributes(
    meta_df: pd.DataFrame,
    *,
    genre_sep: str = "|",
    include_unknown: bool = True,
) -> Tuple[Dict[int, List[int]], Dict[str, int], int]:
    """
    item_meta DF -> (item2attrs, vocab, vocab_size)
    vocab id: 0 is reserved (padding)
    """
    vocab: Dict[str, int] = {}
    next_id = 1

    def _get_id(tok: str) -> int:
        nonlocal next_id
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
        return vocab[tok]

    item2attrs: Dict[int, List[int]] = {}
    for _, row in meta_df.iterrows():
        item = int(row["item"])
        toks: List[str] = []

        yb = row.get("year_bin", "Unknown")
        if include_unknown or (isinstance(yb, str) and yb != "Unknown"):
            toks.append(f"year={yb}")

        d = row.get("director", "Unknown")
        if include_unknown or (isinstance(d, str) and d != "Unknown"):
            toks.append(f"director={d}")

        w = row.get("writer", "Unknown")
        if include_unknown or (isinstance(w, str) and w != "Unknown"):
            toks.append(f"writer={w}")

        g = row.get("genres", "Unknown")
        if isinstance(g, str) and g != "Unknown":
            for gg in g.split(genre_sep):
                gg = gg.strip()
                if gg:
                    toks.append(f"genre={gg}")
        elif include_unknown:
            toks.append("genre=Unknown")

        ids = [_get_id(t) for t in toks]
        item2attrs[item] = ids

    vocab_size = int(max(vocab.values(), default=0) + 1)  # +1 for padding(0)
    return item2attrs, vocab, vocab_size


@register_global("ml_metadata_preprocess_v1")
@dataclass
class MLMetadataPreprocessV1(GlobalTransform):
    """
    MovieLens train_dir용 메타데이터 전처리(GlobalTransform).

    Args:
        load_aux_if_missing: raw["aux_tables"]가 비어있으면 raw["aux_paths"]로부터 로드할지 여부.
        top_n_genres: item별 보존할 장르 개수 (노트북 기본 4)
        genre_sep: genres join/split 구분자
        build_item2attributes: True면 raw["item2attributes"]가 없거나 overwrite일 때 생성
        overwrite_item2attributes: True면 기존 raw["item2attributes"]가 있어도 덮어씀
    """

    load_aux_if_missing: bool = True
    top_n_genres: int = 4
    genre_sep: str = "|"
    build_item2attributes: bool = True
    overwrite_item2attributes: bool = False

    name: str = "ml_metadata_preprocess_v1"

    def __call__(self, cfg: Any, raw: Dict[str, Any]) -> Dict[str, Any]:
        ratings: pd.DataFrame = raw["ratings"]
        aux = _ensure_aux_tables(raw, load=self.load_aux_if_missing)

        required_keys = ["titles", "years", "genres", "directors", "writers"]
        missing = [k for k in required_keys if k not in aux]
        if missing:
            raise ValueError(
                f"Missing aux tables: {missing}. "
                f"Set cfg.dataset.load_aux_tables=True or provide aux files in train_dir."
            )

        titles_df = aux["titles"]
        years_df = aux["years"]
        genres_df = aux["genres"]
        directors_df = aux["directors"]
        writers_df = aux["writers"]

        # directors/writers representative (popular by view counts)
        directors_rep = _get_popular_entity(directors_df, "director", ratings)
        writers_rep = _get_popular_entity(writers_df, "writer", ratings)

        # genres representative
        genres_rep = _build_genres_rep(genres_df, top_n=self.top_n_genres, sep=self.genre_sep)

        # titles + years -> clean_title, final_year
        meta = titles_df.merge(years_df, on="item", how="outer")
        meta = _master_title_processor(meta)
        meta["year_bin"] = meta["final_year"].apply(_bin_year)

        # merge final item metadata
        final = meta[["item", "clean_title", "year_bin"]].merge(directors_rep, on="item", how="left")
        final = final.merge(writers_rep, on="item", how="left")
        final = final.merge(genres_rep, on="item", how="left")
        final = final.fillna({"director": "Unknown", "writer": "Unknown", "genres": "Unknown"})

        raw["item_meta"] = final.reset_index(drop=True)

        # optional: build item2attributes
        if self.build_item2attributes and (self.overwrite_item2attributes or not raw.get("item2attributes")):
            item2attrs, vocab, vocab_size = _build_item2attributes(
                raw["item_meta"], genre_sep=self.genre_sep, include_unknown=True
            )
            raw["item2attributes"] = item2attrs
            raw["attribute_vocab"] = vocab
            raw["attribute_size"] = vocab_size

        return raw



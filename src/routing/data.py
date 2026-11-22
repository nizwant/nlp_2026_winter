"""Load the prepared datasets into one item table per source.

Every source yields the same columns: item_id, source, article_id, original_set, text,
truncated, followed by one 0/1 column per label in the order given by labels.py.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import MIPD_COLUMN_RENAMES, MIPD_LABELS, SEMEVAL_LABELS, SOURCES, labels_for

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ROUTING_DATA_DIR", REPO_ROOT / "data"))

# MIPD articles are cut after this many whitespace-separated words (see PLAN.md, decision 4).
MIPD_MAX_WORDS = 2000

_WORD = re.compile(r"\S+")


def truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    """Cut text after max_words words, leaving the kept part byte-identical to the original."""
    end = None
    for count, match in enumerate(_WORD.finditer(text), start=1):
        if count == max_words:
            end = match.end()
        elif count > max_words:
            return text[:end], True
    return text, False


def _load_semeval(source: str) -> pd.DataFrame:
    language = SOURCES[source]["language"]
    df = pd.read_csv(
        DATA_DIR / "sem_eval" / f"{language}.csv",
        dtype={"article_id": str, "paragraph_number": str},
        keep_default_na=False,
    )
    missing = set(SEMEVAL_LABELS) - set(df.columns)
    if missing:
        raise ValueError(f"{source}: label columns missing from the CSV: {sorted(missing)}")
    df.insert(0, "item_id", source + ":" + df["article_id"] + ":" + df["paragraph_number"])
    df["truncated"] = False
    return df


def _load_mipd() -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "mipd" / "manipulation.csv", dtype={"article_id": str})
    labels = labels.rename(columns=MIPD_COLUMN_RENAMES)
    articles = pd.read_csv(DATA_DIR / "mipd" / "articles.csv", dtype={"article_id": str}, keep_default_na=False)
    df = labels.merge(articles, on="article_id", how="inner", validate="one_to_one")
    if len(df) != len(labels):
        raise ValueError("mipd: some labelled articles have no text")
    cut = [truncate_words(text, MIPD_MAX_WORDS) for text in df["full_article"]]
    df["text"] = [text for text, _ in cut]
    df["truncated"] = [flag for _, flag in cut]
    df.insert(0, "item_id", "mipd:" + df["article_id"])
    credibility = DATA_DIR / "mipd" / "credibility.csv"
    if credibility.exists():
        flags = pd.read_csv(credibility, dtype={"article_id": str}, usecols=["article_id", "disinformative"])
        df = df.merge(flags, on="article_id", how="left", validate="one_to_one")
    return df.drop(columns=["full_article"])


def load_items(source: str) -> pd.DataFrame:
    """Return every gold-labelled item of a source, in a stable order."""
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
    df = _load_mipd() if source == "mipd" else _load_semeval(source)
    df["source"] = source
    labels = list(labels_for(source))
    df[labels] = df[labels].astype(np.int8)
    if not df["item_id"].is_unique:
        raise ValueError(f"{source}: duplicate item ids")
    extra = [c for c in ("category", "disinformative") if c in df.columns]
    columns = ["item_id", "source", "article_id", "original_set", "text", "truncated", *extra, *labels]
    return df[columns].sort_values("item_id", ignore_index=True)


def gold_matrix(items: pd.DataFrame, source: str) -> np.ndarray:
    """Gold labels as an (items, labels) 0/1 array."""
    return items[list(labels_for(source))].to_numpy(dtype=np.int8)


def load_split(source: str) -> pd.DataFrame:
    """Read the committed split assignment (item_id, article_id, original_set, role)."""
    path = DATA_DIR / "splits" / f"{source}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m routing.splits` first")
    return pd.read_csv(path, dtype=str)


def load_role(source: str, role: str) -> pd.DataFrame:
    """Items of one source that were assigned a given role (pilot, calibration, test, ...)."""
    split = load_split(source)
    wanted = split.loc[split["role"] == role, "item_id"]
    items = load_items(source)
    return items[items["item_id"].isin(set(wanted))].reset_index(drop=True)


__all__ = [
    "DATA_DIR",
    "MIPD_LABELS",
    "MIPD_MAX_WORDS",
    "SEMEVAL_LABELS",
    "gold_matrix",
    "load_items",
    "load_role",
    "load_split",
    "truncate_words",
]

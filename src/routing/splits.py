"""Assign every item a role, using the official splits as boundaries.

SemEval: train supplies the few-shot pool, the pilot subset and the calibration set; dev is
the test set. MIPD: train supplies the few-shot pool and the pilot; validation supplies
calibration; the official test split is the test set. Whole articles are assigned together.

    python -m routing.splits            # writes data/splits/<source>.csv and prints counts
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from .data import DATA_DIR, load_items
from .labels import SOURCES, labels_for

SEED = 0

# Targets per source. Paragraph targets are reached by adding whole articles until met.
SEMEVAL_FEWSHOT_ARTICLES = 10
SEMEVAL_PILOT_PARAGRAPHS = 200
SEMEVAL_CALIBRATION_PARAGRAPHS = 1000
SEMEVAL_EN_TEST_PARAGRAPHS = 1500  # Polish dev is used in full
MIPD_PILOT_ARTICLES = 100
MIPD_FEWSHOT_ARTICLES = 200
MIPD_CALIBRATION_ARTICLES = 600

ROLES = ("fewshot", "pilot", "calibration", "test", "unused", "dropped_duplicate")

_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def _shuffled_articles(items: pd.DataFrame, stream: int) -> list[str]:
    articles = sorted(items["article_id"].unique())
    rng = np.random.default_rng([SEED, stream])
    return [articles[i] for i in rng.permutation(len(articles))]


def _take_until(order: list[str], sizes: pd.Series, start: int, target: int) -> int:
    """Index after the last article needed to reach `target` items, starting at `start`."""
    total, stop = 0, start
    while stop < len(order) and total < target:
        total += int(sizes[order[stop]])
        stop += 1
    return stop


def _assign_semeval(items: pd.DataFrame, source: str) -> pd.Series:
    role = pd.Series("unused", index=items.index)
    train = items[items["original_set"] == "train"]
    sizes = train.groupby("article_id").size()
    order = _shuffled_articles(train, stream=1)

    a = SEMEVAL_FEWSHOT_ARTICLES
    b = _take_until(order, sizes, a, SEMEVAL_PILOT_PARAGRAPHS)
    c = _take_until(order, sizes, b, SEMEVAL_CALIBRATION_PARAGRAPHS)
    for name, chosen in (("fewshot", order[:a]), ("pilot", order[a:b]), ("calibration", order[b:c])):
        role[train.index[train["article_id"].isin(chosen)]] = name

    dev = items[items["original_set"] == "dev"]
    if source == "semeval_en":
        dev_sizes = dev.groupby("article_id").size()
        dev_order = _shuffled_articles(dev, stream=2)
        stop = _take_until(dev_order, dev_sizes, 0, SEMEVAL_EN_TEST_PARAGRAPHS)
        dev = dev[dev["article_id"].isin(dev_order[:stop])]
    role[dev.index] = "test"
    return role


def _assign_mipd(items: pd.DataFrame) -> pd.Series:
    role = pd.Series("unused", index=items.index)
    train = items[items["original_set"] == "train"]
    order = _shuffled_articles(train, stream=1)
    pilot = order[:MIPD_PILOT_ARTICLES]
    fewshot = order[MIPD_PILOT_ARTICLES : MIPD_PILOT_ARTICLES + MIPD_FEWSHOT_ARTICLES]
    role[train.index[train["article_id"].isin(pilot)]] = "pilot"
    role[train.index[train["article_id"].isin(fewshot)]] = "fewshot"

    validation = items[items["original_set"] == "validation"]
    chosen = _shuffled_articles(validation, stream=2)[:MIPD_CALIBRATION_ARTICLES]
    role[validation.index[validation["article_id"].isin(chosen)]] = "calibration"

    role[items.index[items["original_set"] == "test"]] = "test"
    return role


def make_split(source: str) -> pd.DataFrame:
    """Role per item for one source, with leakage between test and the other roles removed."""
    items = load_items(source)
    role = _assign_mipd(items) if source == "mipd" else _assign_semeval(items, source)

    # Leakage rule: an item outside test whose normalised text also occurs in test is dropped.
    text = items["text"].map(normalise)
    in_test = set(text[role == "test"])
    leaked = (role != "test") & (role != "unused") & text.isin(in_test)
    role[leaked] = "dropped_duplicate"

    return pd.DataFrame(
        {
            "item_id": items["item_id"],
            "article_id": items["article_id"],
            "original_set": items["original_set"],
            "role": role,
        }
    )


def summarise(source: str, split: pd.DataFrame) -> pd.DataFrame:
    items = load_items(source)
    labels = list(labels_for(source))
    merged = split.merge(items[["item_id", *labels]], on="item_id", validate="one_to_one")
    merged["non_empty"] = merged[labels].sum(axis=1) > 0
    table = merged.groupby("role").agg(
        items=("item_id", "size"), articles=("article_id", "nunique"), non_empty=("non_empty", "sum")
    )
    table.insert(0, "source", source)
    return table.reindex([r for r in ROLES if r in table.index])


def check_split(split: pd.DataFrame) -> None:
    """Fail loudly if an article is shared between test and any other assigned role."""
    assigned = split[~split["role"].isin(["unused", "dropped_duplicate"])]
    roles_per_article = assigned.groupby("article_id")["role"].nunique()
    shared = roles_per_article[roles_per_article > 1]
    if len(shared):
        raise AssertionError(f"articles assigned to more than one role: {list(shared.index[:5])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=sorted(SOURCES), choices=sorted(SOURCES))
    args = parser.parse_args()

    out_dir = DATA_DIR / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for source in args.sources:
        split = make_split(source)
        check_split(split)
        split.to_csv(out_dir / f"{source}.csv", index=False)
        summaries.append(summarise(source, split))
    print(pd.concat(summaries).to_string())
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()

"""Download MIPD from the authors' repository and build the prepared files in data/mipd/.

    python scripts/download_mipd.py

MIPD (Modzelewski et al., EMNLP 2024) is distributed at https://github.com/ArkadiusDS/MIPD under
CC BY-NC-ND 4.0. This script fetches the files unmodified from a pinned commit into data/raw/mipd/
and derives, for local use only:

    data/mipd/articles.csv      article_id, full_article, category
    data/mipd/manipulation.csv  article_id, original_set, 11 manipulation-technique labels
    data/mipd/intention.csv     article_id, original_set, 9 intention labels
    data/mipd/credibility.csv   article_id, original_set, disinformative (1 = disinformative, 0 = credible)

The split files carry no id, so they are linked to raw_articles.csv by exact article text.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

COMMIT = "51aac23c756a3837f40cf516a8b46dea92f2952a"  # master on 2026-06-21
BASE_URL = f"https://raw.githubusercontent.com/ArkadiusDS/MIPD/{COMMIT}/data/"
SPLITS = ("train", "validation", "test")
TASKS = ("manipulation", "intention", "disinformation")

# Expected size in bytes of every file at the pinned commit.
FILES = {
    "raw_articles.csv": 89283602,
    "articles_categories.csv": 247594,
    "disinformation/train.csv": 62301963,
    "disinformation/validation.csv": 18185249,
    "disinformation/test.csv": 8746121,
    "intention/train.csv": 62855517,
    "intention/validation.csv": 18344352,
    "intention/test.csv": 8824695,
    "manipulation/train.csv": 62984179,
    "manipulation/validation.csv": 18381262,
    "manipulation/test.csv": 8842757,
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def download(raw_dir: Path) -> None:
    for name, expected in FILES.items():
        target = raw_dir / name
        if target.exists() and target.stat().st_size == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ({expected / 1e6:.1f} MB)", flush=True)
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(BASE_URL + name, timeout=120) as response, open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        if partial.stat().st_size != expected:
            raise RuntimeError(f"{name}: got {partial.stat().st_size} bytes, expected {expected}")
        partial.replace(target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build(raw_dir: Path, out_dir: Path) -> None:
    raw = pd.read_csv(raw_dir / "raw_articles.csv")
    categories = pd.read_csv(raw_dir / "articles_categories.csv")
    articles = raw.merge(categories, on="article_id", how="inner", validate="one_to_one")
    if not (len(articles) == len(raw) == len(categories)):
        raise ValueError("articles and categories do not match one to one")
    if not articles["full_article"].is_unique:
        raise ValueError("article texts are not unique, so they cannot serve as a join key")
    text_to_id = articles.set_index("full_article")["article_id"]

    tables = {}
    for task in TASKS:
        frame = pd.concat(
            [pd.read_csv(raw_dir / task / f"{split}.csv").assign(original_set=split) for split in SPLITS],
            ignore_index=True,
        )
        frame["article_id"] = frame["article"].map(text_to_id)
        if frame["article_id"].isna().any():
            raise ValueError(f"{task}: {int(frame['article_id'].isna().sum())} articles not found in raw_articles.csv")
        if not frame["article_id"].is_unique:
            raise ValueError(f"{task}: an article appears in more than one split")
        if set(frame["article_id"]) != set(articles["article_id"]):
            raise ValueError(f"{task}: the splits do not cover every article exactly once")
        labels = [c for c in frame.columns if c not in ("article", "original_set", "article_id")]
        frame["article_id"] = frame["article_id"].astype(int)
        frame[labels] = frame[labels].astype(int)
        tables[task] = frame[["article_id", "original_set", *labels]].sort_values("article_id", ignore_index=True)

    reference = tables["manipulation"][["article_id", "original_set"]]
    for task in ("intention", "disinformation"):
        if not tables[task][["article_id", "original_set"]].equals(reference):
            raise ValueError(f"{task}: split assignment differs from the manipulation task")

    out_dir.mkdir(parents=True, exist_ok=True)
    articles.to_csv(out_dir / "articles.csv", index=False)
    tables["manipulation"].to_csv(out_dir / "manipulation.csv", index=False)
    tables["intention"].to_csv(out_dir / "intention.csv", index=False)
    credibility = tables["disinformation"].rename(columns={"disinformation": "disinformative"})
    credibility.to_csv(out_dir / "credibility.csv", index=False)

    techniques = [c for c in tables["manipulation"].columns if c not in ("article_id", "original_set")]
    has_technique = tables["manipulation"][techniques].sum(axis=1) > 0
    overview = pd.crosstab(
        credibility["disinformative"].map({1: "disinformative", 0: "credible"}),
        has_technique.map({True: "with technique", False: "no technique"}),
        rownames=["article"],
        colnames=["techniques"],
        margins=True,
    )
    print("\narticles by credibility and technique labels:")
    print(overview.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "mipd")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "mipd")
    args = parser.parse_args()

    download(args.raw_dir)
    build(args.raw_dir, args.out_dir)
    print("\nsha256 of the prepared files:")
    for name in ("articles.csv", "manipulation.csv", "intention.csv", "credibility.csv"):
        print(f"  {sha256(args.out_dir / name)}  {name}")


if __name__ == "__main__":
    sys.exit(main())

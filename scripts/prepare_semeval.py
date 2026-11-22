"""Build data/sem_eval/{en,pl}.csv from the SemEval-2023 Task 3 files obtained from the organisers.

    python scripts/prepare_semeval.py --raw-dir /path/to/official/data

The data must be requested at https://propaganda.math.unipd.it/semeval2023task3/ (registration and a
team passcode) and may be used for scientific research purposes only, under the organisers' terms.

Expected layout, as distributed for subtask 3:

    <raw-dir>/<language>/train-labels-subtask-3.txt        article id, paragraph number, labels
    <raw-dir>/<language>/train-labels-subtask-3.template   article id, paragraph number, paragraph text
    <raw-dir>/<language>/dev-labels-subtask-3.txt
    <raw-dir>/<language>/dev-labels-subtask-3.template

with <language> being "en" and "pl" (the Polish folder is named "po" in the distribution; both names
are accepted). Test gold labels are not public, so only train and dev are used.

Output columns: article_id, paragraph_number, original_set, text, then one 0/1 column per technique.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from routing.labels import SEMEVAL_LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYS = ["article_id", "paragraph_number"]
FOLDER_NAMES = {"en": ("en",), "pl": ("pl", "po")}


def read_rows(path: Path) -> list[list[str]]:
    # Paragraph texts may contain tabs, so only the first two tabs separate fields.
    with open(path, encoding="utf-8") as source:
        return [line.rstrip("\r\n").split("\t", 2) for line in source if line.strip()]


def load_split(folder: Path, original_set: str) -> pd.DataFrame:
    where = f"{folder.name}/{original_set}"
    base = folder / f"{original_set}-labels-subtask-3"

    label_rows = [row + [""] * (3 - len(row)) for row in read_rows(base.with_suffix(".txt"))]
    labels = pd.DataFrame(label_rows, columns=[*KEYS, "labels"])
    if labels.duplicated(KEYS).any():
        raise ValueError(f"{where}: duplicate label keys")

    text_rows = read_rows(base.with_suffix(".template"))
    if any(len(row) != 3 for row in text_rows):
        raise ValueError(f"{where}: malformed template row")

    # Some template rows are concatenated without a newline. Their boundaries are recovered from
    # the keys that the labels file has and the template lacks.
    missing = set(map(tuple, labels[KEYS].to_numpy())) - {tuple(row[:2]) for row in text_rows}
    if missing:
        boundary = re.compile("|".join(re.escape(f"{article}\t{paragraph}\t") for article, paragraph in sorted(missing)))
        recovered = []
        for article, paragraph, text in text_rows:
            previous_end, key = 0, (article, paragraph)
            for match in boundary.finditer(text):
                recovered.append([*key, text[previous_end : match.start()]])
                key = tuple(match.group().rstrip("\t").split("\t"))
                previous_end = match.end()
            recovered.append([*key, text[previous_end:]])
        text_rows = recovered

    texts = pd.DataFrame(text_rows, columns=[*KEYS, "text"])
    joined = labels.merge(texts, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{where}: labels and paragraphs do not match")

    label_sets = joined["labels"].map(lambda value: set(filter(None, value.split(","))))
    unknown = set().union(*label_sets) - set(SEMEVAL_LABELS)
    if unknown:
        raise ValueError(f"{where}: unknown labels {sorted(unknown)}")
    for technique in SEMEVAL_LABELS:
        joined[technique] = label_sets.map(lambda values, t=technique: int(t in values))
    joined["paragraph_number"] = joined["paragraph_number"].astype(int)
    joined["original_set"] = original_set
    return joined[["article_id", "paragraph_number", "original_set", "text", *SEMEVAL_LABELS]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="folder holding the language folders")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "sem_eval")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for language, names in FOLDER_NAMES.items():
        folder = next((args.raw_dir / name for name in names if (args.raw_dir / name).is_dir()), None)
        if folder is None:
            raise FileNotFoundError(f"no folder for {language!r} in {args.raw_dir} (looked for {', '.join(names)})")
        dataset = pd.concat([load_split(folder, s) for s in ("train", "dev")], ignore_index=True)
        target = args.out_dir / f"{language}.csv"
        dataset.to_csv(target, index=False)
        counts = dataset["original_set"].value_counts().to_dict()
        print(f"{language}: {len(dataset)} paragraphs {counts}  sha256 {sha256(target)}")


if __name__ == "__main__":
    main()

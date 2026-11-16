"""Timed human annotation of a small sample, for the cost model and for agreement.

    python -m routing.annotate semeval_pl --annotator MN     # annotate (resumes where you stopped)
    python -m routing.annotate --summary                     # median time and agreement

The sample is fixed and shared by all annotators: 40 Polish and 40 English SemEval paragraphs
and 15 MIPD articles, drawn with a seed from the pilot subsets (never from a test set). Each
item is shown with the numbered list of techniques the models see; the annotator types the
numbers of the techniques that occur, or just Enter for none. The time from showing the text to
pressing Enter is recorded. Typing `q` stops; `b` takes a break without counting the time.

Files (no text is stored): data/annotation/sample_<source>.csv holds the item ids,
data/annotation/<source>_<annotator>.csv one row per annotated item with the seconds and the
chosen techniques. Gold labels are never shown.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import DATA_DIR, gold_matrix, load_role
from .labels import SOURCES, labels_for
from .prompts import _definitions

ANNOTATION_DIR = DATA_DIR / "annotation"
SAMPLE_SIZE = {"semeval_pl": 40, "semeval_en": 40, "mipd": 15}
SEED = 0


def sample(source: str) -> list[str]:
    """The fixed item ids to annotate, drawn once from the pilot subset and then kept on disk."""
    path = ANNOTATION_DIR / f"sample_{source}.csv"
    if path.exists():
        return pd.read_csv(path)["item_id"].tolist()
    pilot = load_role(source, "pilot").sort_values("item_id")
    rng = np.random.default_rng([SEED, sorted(SOURCES).index(source)])
    chosen = sorted(rng.choice(len(pilot), size=min(SAMPLE_SIZE[source], len(pilot)), replace=False))
    item_ids = pilot["item_id"].iloc[chosen].tolist()
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": item_ids}).to_csv(path, index=False)
    return item_ids


def technique_menu(source: str) -> str:
    """Numbered techniques with their definitions in the language of the text, as the models get them."""
    language, task = SOURCES[source]["language"], SOURCES[source]["task"]
    definitions = _definitions(task)
    lines = []
    for number, label in enumerate(labels_for(source), start=1):
        entry = definitions[label]
        lines.append(f"{number:>2}. {entry[f'name_{language}']}: {entry[f'def_{language}'].strip()}")
    return "\n".join(lines)


def parse_choice(answer: str, n_labels: int) -> list[int] | None:
    """Decisions in label order from an answer like '1, 4 17'; None if the answer is not valid."""
    decisions = [0] * n_labels
    for part in answer.replace(",", " ").split():
        if not part.isdigit() or not 1 <= int(part) <= n_labels:
            return None
        decisions[int(part) - 1] = 1
    return decisions


def annotate(
    source: str,
    annotator: str,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> Path:
    labels = labels_for(source)
    items = load_role(source, "pilot").set_index("item_id")
    path = ANNOTATION_DIR / f"{source}_{annotator}.csv"
    done = set(pd.read_csv(path)["item_id"]) if path.exists() else set()
    todo = [item_id for item_id in sample(source) if item_id not in done]
    menu = technique_menu(source)
    show(f"{len(todo)} items to annotate for {source}. Enter the numbers of the techniques that occur "
         f"(for example: 1 4 17), Enter alone for none, b for a break, q to stop.\n")
    for position, item_id in enumerate(todo, start=1):
        show("=" * 100)
        show(menu)
        show("-" * 100)
        show(f"[{position}/{len(todo)}]\n{items.at[item_id, 'text']}\n")
        seconds, started = 0.0, clock()
        while True:
            answer = ask("techniques> ").strip().lower()
            if answer == "q":
                return path
            if answer == "b":
                seconds += clock() - started
                ask("break; press Enter to continue")
                started = clock()
                continue
            decisions = parse_choice(answer, len(labels))
            if decisions is not None:
                break
            show(f"not understood: numbers from 1 to {len(labels)} separated by spaces")
        seconds += clock() - started
        row = pd.DataFrame(
            [{
                "item_id": item_id,
                "annotator": annotator,
                "seconds": round(seconds, 1),
                "labels": "".join(map(str, decisions)),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }]
        )
        row.to_csv(path, mode="a", header=not path.exists(), index=False)
    show(f"done; saved to {path}")
    return path


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary vectors; NaN when chance agreement is 1."""
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def summary(source: str) -> dict | None:
    """Median annotation time, agreement between annotators and agreement with gold for a source."""
    files = sorted(ANNOTATION_DIR.glob(f"{source}_*.csv"))
    if not files:
        return None
    frames = [pd.read_csv(path, dtype={"labels": str}) for path in files]
    table = pd.concat(frames, ignore_index=True).drop_duplicates(["item_id", "annotator"], keep="last")
    labels = labels_for(source)
    items = load_role(source, "pilot")
    gold = pd.DataFrame(gold_matrix(items, source), index=items["item_id"])
    result = {
        "source": source,
        "annotators": sorted(table["annotator"].unique()),
        "annotations": int(len(table)),
        "median_seconds_per_item": float(table["seconds"].median()),
        "median_seconds_by_annotator": table.groupby("annotator")["seconds"].median().round(1).to_dict(),
        "mean_seconds_per_item": float(table["seconds"].mean()),
    }
    bits = lambda column: np.array([[int(ch) for ch in value] for value in column], dtype=np.int8)  # noqa: E731
    chosen = bits(table["labels"])
    reference = gold.loc[table["item_id"]].to_numpy()
    result["exact_set_agreement_with_gold"] = float((chosen == reference).all(axis=1).mean())

    wide = table.pivot(index="item_id", columns="annotator", values="labels").dropna()
    if wide.shape[1] >= 2 and len(wide):
        first, second = bits(wide.iloc[:, 0]), bits(wide.iloc[:, 1])
        union = ((first + second) > 0).sum(axis=1)
        jaccard = np.where(union == 0, 1.0, (first * second).sum(axis=1) / np.maximum(union, 1))
        kappas = [_kappa(first[:, j], second[:, j]) for j in range(len(labels)) if first[:, j].sum() + second[:, j].sum() >= 5]
        result.update(
            {
                "items_with_two_annotators": int(len(wide)),
                "exact_set_agreement": float((first == second).all(axis=1).mean()),
                "mean_jaccard": float(jaccard.mean()),
                "mean_kappa_over_frequent_techniques": float(np.nanmean(kappas)) if kappas else float("nan"),
                "techniques_in_kappa": len(kappas),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", choices=sorted(SOURCES))
    parser.add_argument("--annotator", help="your initials; used in the file name")
    parser.add_argument("--summary", action="store_true", help="print times and agreement for all sources")
    args = parser.parse_args()
    if args.summary:
        for source in sorted(SOURCES):
            result = summary(source)
            if result is None:
                print(f"{source}: no annotations yet")
                continue
            print(f"{source}:")
            for key, value in result.items():
                print(f"  {key}: {round(value, 3) if isinstance(value, float) else value}")
            print(f"  -> configs/cost.yaml: human_seconds_per_item.{source}: {result['median_seconds_per_item']}")
        return
    if not args.source or not args.annotator:
        parser.error("give a source and --annotator, or --summary")
    annotate(args.source, args.annotator)


if __name__ == "__main__":
    main()

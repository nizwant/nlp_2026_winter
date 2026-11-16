"""Choose the fixed few-shot examples of every source and store their item ids.

    python -m routing.fewshot            # writes prompts/fewshot_<source>.yaml

Rule, applied to the few-shot pool of the source (train articles that are used for nothing else):

1. Keep items whose text has between `min_words` and `max_words` words, so the examples stay short.
2. Take `n_empty` examples without any technique. For MIPD one of them is a credible article and
   one a disinformative article without a technique, because both kinds occur in the data.
3. Fill up to `n_examples` greedily with items that carry techniques: each step takes the item
   that adds the largest share of not yet shown techniques, weighted by how often each technique
   occurs in the pool. Ties go to the item with fewer labels, then to the seeded order.
4. An article supplies at most one example.
5. Shuffle the chosen examples with the seed, so the empty ones are not all at the front.

Only item ids are written; the text is read from the dataset when a prompt is built.
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from .data import gold_matrix, load_role
from .labels import SOURCES, labels_for
from .prompts import PROMPT_DIR

SETTINGS = {
    "semeval_pl": {"n_examples": 6, "n_empty": 2, "min_words": 8, "max_words": 80},
    "semeval_en": {"n_examples": 6, "n_empty": 2, "min_words": 8, "max_words": 80},
    # Articles are long, so fewer and shorter examples keep the prompt within the context window.
    "mipd": {"n_examples": 4, "n_empty": 2, "min_words": 40, "max_words": 300},
}
SEED = 0


def select(source: str, seed: int = SEED) -> list[str]:
    settings = SETTINGS[source]
    pool = load_role(source, "fewshot")
    words = pool["text"].str.split().str.len()
    pool = pool[(words >= settings["min_words"]) & (words <= settings["max_words"])].reset_index(drop=True)
    gold = gold_matrix(pool, source)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))  # seeded order used for every tie

    article = pool["article_id"].to_numpy()

    def one_per_article(rows: list[int], taken: list[int]) -> list[int]:
        used, kept = {article[i] for i in taken}, []
        for i in rows:
            if article[i] not in used:
                used.add(article[i])
                kept.append(i)
        return kept

    empty = one_per_article([i for i in order if gold[i].sum() == 0], [])
    if source == "mipd" and "disinformative" in pool:
        credible = [i for i in empty if pool.at[i, "disinformative"] == 0]
        disinformative = [i for i in empty if pool.at[i, "disinformative"] == 1]
        chosen = credible[:1] + disinformative[:1]
        chosen += [i for i in empty if i not in chosen][: settings["n_empty"] - len(chosen)]
    else:
        chosen = empty[: settings["n_empty"]]
    if len(chosen) < settings["n_empty"]:
        raise ValueError(f"{source}: not enough short items without a technique in the few-shot pool")

    weight = gold.mean(axis=0)  # frequency of each technique in the pool
    covered = np.zeros(gold.shape[1], dtype=bool)
    candidates = [i for i in order if gold[i].sum() > 0]
    while len(chosen) < settings["n_examples"]:
        candidates = [i for i in candidates if article[i] not in {article[j] for j in chosen}]
        if not candidates:
            raise ValueError(f"{source}: not enough short items with a technique in the few-shot pool")
        gain = [float(weight[(gold[i] == 1) & ~covered].sum()) for i in candidates]
        best = max(range(len(candidates)), key=lambda k: (gain[k], -int(gold[candidates[k]].sum()), -k))
        pick = candidates.pop(best)
        chosen.append(pick)
        covered |= gold[pick] == 1

    chosen = [chosen[k] for k in rng.permutation(len(chosen))]
    return [str(pool.at[i, "item_id"]) for i in chosen]


def write(source: str) -> list[str]:
    item_ids = select(source)
    pool = load_role(source, "fewshot").set_index("item_id")
    labels = list(labels_for(source))
    listing = {
        "source": source,
        "seed": SEED,
        **SETTINGS[source],
        "item_ids": item_ids,
        # Shown for the reader only; prompts read the gold labels from the dataset.
        "gold": {i: [label for label in labels if pool.at[i, label] == 1] for i in item_ids},
    }
    with open(PROMPT_DIR / f"fewshot_{source}.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(listing, handle, sort_keys=False, allow_unicode=True)
    return item_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", default=sorted(SOURCES))
    args = parser.parse_args()
    for source in args.sources:
        item_ids = write(source)
        print(f"{source}: {len(item_ids)} examples -> prompts/fewshot_{source}.yaml")


if __name__ == "__main__":
    main()

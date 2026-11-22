"""Classical baseline: TF-IDF features with one logistic regression per technique.

    python -m routing.baseline_tfidf                    # all sources; pilot and calibration items
    python -m routing.baseline_tfidf semeval_pl --roles pilot calibration test --allow-test

The baseline is the second method class of the study (the topic excludes fine-tuning of
transformers). It is used in two ways: as a task baseline with its own predictions, and as a
cheap confidence signal for the predictions of the language models.

Training data: the official train split without the items that have another role (pilot,
calibration), and without any item whose normalised text also occurs in pilot, calibration or
test. The regularisation strength is chosen by 5-fold cross-validation over articles on that
training data, so no evaluation item influences any choice.

Output (no text): outputs/baselines/tfidf_<source>.csv with P(label = 1) per item, and
tfidf_<source>.json with the settings, the training size, the run time and the task quality.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import REPO_ROOT, gold_matrix, load_items, load_split
from .labels import SOURCES, labels_for
from .metrics import auroc, task_quality
from .splits import normalise

BASELINE_DIR = REPO_ROOT / "outputs" / "baselines"
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SEED = 0
EVALUATION_ROLES = ("pilot", "calibration", "test")
_TRAIN_SET = {"semeval_pl": "train", "semeval_en": "train", "mipd": "train"}


def build_model(c: float) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200_000)),
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=300_000),
            ),
        ]
    )
    classifier = OneVsRestClassifier(
        LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def training_items(source: str) -> pd.DataFrame:
    """Train-split items with no other role, minus texts that also occur in an evaluation role."""
    items = load_items(source).merge(load_split(source)[["item_id", "role"]], on="item_id")
    held_out = set(items.loc[items["role"].isin(EVALUATION_ROLES), "text"].map(normalise))
    train = items[(items["original_set"] == _TRAIN_SET[source]) & items["role"].isin(["fewshot", "unused"])]
    return train[~train["text"].map(normalise).isin(held_out)].reset_index(drop=True)


def _probabilities(model: Pipeline, texts: pd.Series, n_labels: int) -> np.ndarray:
    probability = np.asarray(model.predict_proba(texts), dtype=float).reshape(len(texts), n_labels)
    return np.clip(probability, 1e-4, 1.0 - 1e-4)


def _fit(model: Pipeline, texts: pd.Series, gold: np.ndarray) -> Pipeline:
    with warnings.catch_warnings():
        # A technique that never occurs in the training data gets a constant "absent" classifier.
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        return model.fit(texts, gold)


def choose_c(train: pd.DataFrame, gold: np.ndarray) -> tuple[float, dict[float, float]]:
    """Micro-F1 of every C in the grid under 5-fold cross-validation over articles."""
    folds = list(GroupKFold(n_splits=5).split(train["text"], groups=train["article_id"]))
    scores = {}
    for c in C_GRID:
        predicted = np.zeros_like(gold)
        for fit_rows, held_rows in folds:
            model = _fit(build_model(c), train["text"].iloc[fit_rows], gold[fit_rows])
            predicted[held_rows] = _probabilities(model, train["text"].iloc[held_rows], gold.shape[1]) >= 0.5
        scores[c] = task_quality(gold, predicted)["micro_f1"]
    best = max(C_GRID, key=lambda c: (scores[c], -c))
    return best, scores


def run(source: str, roles: tuple[str, ...] = ("pilot", "calibration")) -> dict:
    started = time.perf_counter()
    labels = list(labels_for(source))
    train = training_items(source)
    gold_train = gold_matrix(train, source)
    c, cv_scores = choose_c(train, gold_train)
    model = _fit(build_model(c), train["text"], gold_train)

    items = load_items(source).merge(load_split(source)[["item_id", "role"]], on="item_id")
    report = {
        "source": source,
        "features": "TF-IDF of word 1-2-grams and character 2-5-grams (within words), sublinear tf, min_df 2",
        "classifier": "one logistic regression per technique, class_weight balanced",
        "C": c,
        "cross_validation_micro_f1": {str(k): round(v, 4) for k, v in cv_scores.items()},
        "n_training_items": int(len(train)),
        "n_training_articles": int(train["article_id"].nunique()),
        "labels_without_training_positives": [l for l, n in zip(labels, gold_train.sum(axis=0)) if n == 0],
        "roles": {},
    }
    frames = []
    for role in roles:
        part = items[items["role"] == role].reset_index(drop=True)
        probability = _probabilities(model, part["text"], len(labels))
        frames.append(
            pd.concat(
                [part[["item_id", "role"]], pd.DataFrame(probability.round(5), columns=[f"p_{l}" for l in labels])],
                axis=1,
            )
        )
        if source == "mipd":  # techniques were annotated on disinformative articles only
            scored = (part["disinformative"] == 1).to_numpy()
        else:
            scored = np.ones(len(part), dtype=bool)
        gold = gold_matrix(part, source)[scored]
        predicted = (probability[scored] >= 0.5).astype(np.int8)
        own = np.where(predicted == 1, probability[scored], 1.0 - probability[scored])
        correct = (predicted == gold).all(axis=1)
        report["roles"][role] = {
            **task_quality(gold, predicted),
            "auroc_own_confidence": auroc(correct, np.log(own).mean(axis=1)),
        }
    report["seconds"] = round(time.perf_counter() - started, 1)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(BASELINE_DIR / f"tfidf_{source}.csv", index=False)
    (BASELINE_DIR / f"tfidf_{source}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", default=sorted(SOURCES))
    parser.add_argument("--roles", nargs="+", default=["pilot", "calibration"], choices=EVALUATION_ROLES)
    parser.add_argument("--allow-test", action="store_true", help="needed to score the test role")
    args = parser.parse_args()
    if "test" in args.roles and not args.allow_test:
        parser.error("the test sets stay untouched until the analysis plan is frozen; pass --allow-test once it is")
    for source in args.sources:
        report = run(source, tuple(args.roles))
        print(f"{source}: C={report['C']}, {report['n_training_items']} training items, {report['seconds']} s")
        for role, quality in report["roles"].items():
            print(
                f"  {role:<12} micro-F1 {quality['micro_f1']:.3f}  macro-F1 {quality['macro_f1']:.3f}  "
                f"exact {quality['exact_set_accuracy']:.3f}  AUROC of own confidence {quality['auroc_own_confidence']:.3f}  "
                f"(n={quality['n_items']})"
            )


if __name__ == "__main__":
    main()

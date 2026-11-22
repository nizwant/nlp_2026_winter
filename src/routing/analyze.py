"""Score one run against gold: task quality, and how well each signal predicts correctness.

    python -m routing.analyze outputs/runs/<run_name>

Prints the tables and writes two files next to the log: summary.json with every number, and
items.csv with one row per item (ids, correctness and signal values, no text) for the routing
analysis and the notebooks. All signals score the same prediction, the per-label majority vote
over the sampled runs of batch 0; the single-run policies are reported in a table of their own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import signals
from .data import REPO_ROOT, gold_matrix, load_role
from .labels import labels_for
from .metrics import auroc, cluster_bootstrap_ci, error_pr_auc, task_quality

BASELINE_DIR = REPO_ROOT / "outputs" / "baselines"

PRIMARY = "consistency: mean pairwise Jaccard"
# Signals that the model itself provides; the item-level comparison uses the items that have all of them.
MODEL_SIGNALS = (
    PRIMARY,
    "consistency: modal-set frequency",
    "consistency: Hamming agreement",
    "log-probability: voted set, teacher-forced",
    "log-probability: voted set, greedy pass",
    "verbalised confidence",
)


def load_generations(run_dir: Path) -> pd.DataFrame:
    """All generations of a run, keeping the last record when a request was retried."""
    with open(run_dir / "generations.jsonl", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    frame = pd.DataFrame(records)
    return frame.drop_duplicates(subset=["item_id", "kind", "seed"], keep="last").reset_index(drop=True)


def assemble(run_dir: Path) -> dict:
    """Join a run with gold labels and stack the outputs into arrays."""
    with open(run_dir / "config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    source, n_samples = config["source"], int(config["n_samples"])
    n_batches, base_seed = int(config.get("n_batches", 1)), int(config["base_seed"])
    labels = labels_for(source)
    generations = load_generations(run_dir)

    items = load_role(source, config["role"])
    items = items[items["item_id"].isin(set(generations["item_id"]))].reset_index(drop=True)
    gold = gold_matrix(items, source)
    row_of = {item_id: row for row, item_id in enumerate(items["item_id"])}
    n_items, n_labels = len(items), len(labels)

    samples = np.zeros((n_items, n_batches, n_samples, n_labels), dtype=np.int8)
    filled = np.zeros((n_items, n_batches, n_samples), dtype=bool)
    arrays = {
        "greedy": np.zeros((n_items, n_labels), dtype=np.int8),
        "has_greedy": np.zeros(n_items, dtype=bool),
        "p1_greedy": np.full((n_items, n_labels), np.nan),
    }
    assessed: dict[str, dict[int, dict]] = {"score": {}, "verbal": {}, "verbal_greedy": {}}
    for record in generations[generations["valid"]].to_dict("records"):
        row = row_of.get(record["item_id"])
        if row is None:
            continue
        kind = record["kind"]
        if kind == "sample":
            batch, r = divmod(int(record["seed"]) - base_seed, 100)
            if 0 <= batch < n_batches and 0 <= r < n_samples:
                samples[row, batch, r] = record["labels"]
                filled[row, batch, r] = True
        elif kind == "greedy":
            arrays["greedy"][row] = record["labels"]
            arrays["has_greedy"][row] = True
            if record.get("p1") is not None:
                arrays["p1_greedy"][row] = record["p1"]
        elif kind in assessed:
            assessed[kind][row] = record

    complete = filled.all(axis=2)  # (items, batches)
    vote = np.zeros((n_items, n_labels), dtype=np.int8)
    for row in np.flatnonzero(complete[:, 0]):
        vote[row] = signals.majority_vote(samples[row, 0])

    # Confidence queries count only if they assessed the decisions that are scored here.
    # Where the greedy answer equals the vote, the greedy pass conditioned every decision token on
    # the same prefix as teacher forcing would, so its probabilities stand in for a scoring request
    # (routing.run skips those requests unless score_when_greedy_agrees is set).
    p1_forced = np.full((n_items, n_labels), np.nan)
    forced_from_greedy = np.zeros(n_items, dtype=bool)
    verbal_overall = np.full(n_items, np.nan)
    verbal_greedy_overall = np.full(n_items, np.nan)
    for row in range(n_items):
        target = vote[row].tolist()
        record = assessed["score"].get(row)
        if record and complete[row, 0] and record["labels"] == target and record.get("p1") is not None:
            p1_forced[row] = record["p1"]
        elif (
            complete[row, 0]
            and arrays["has_greedy"][row]
            and arrays["greedy"][row].tolist() == target
            and not np.isnan(arrays["p1_greedy"][row]).any()
        ):
            p1_forced[row] = arrays["p1_greedy"][row]
            forced_from_greedy[row] = True
        record = assessed["verbal"].get(row)
        if record and complete[row, 0] and record["labels"] == target and record.get("verbal_overall") is not None:
            verbal_overall[row] = record["verbal_overall"]
        if arrays["has_greedy"][row]:
            greedy = arrays["greedy"][row].tolist()
            record = assessed["verbal_greedy"].get(row)
            if record and record["labels"] == greedy and record.get("verbal_overall") is not None:
                verbal_greedy_overall[row] = record["verbal_overall"]
            elif complete[row, 0] and greedy == target:
                verbal_greedy_overall[row] = verbal_overall[row]

    return {
        "config": config,
        "labels": labels,
        "items": items,
        "gold": gold,
        "samples": samples,
        "complete": complete,
        "vote": vote,
        "p1_forced": p1_forced,
        "forced_from_greedy": forced_from_greedy,
        "verbal_overall": verbal_overall,
        "verbal_greedy_overall": verbal_greedy_overall,
        "generations": generations,
        **arrays,
    }


def _subset(data: dict, mask: np.ndarray) -> dict:
    """The same run restricted to the items selected by a boolean mask."""
    n_items = len(data["items"])
    out = dict(data)
    out["items"] = data["items"][mask].reset_index(drop=True)
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n_items,):
            out[key] = value[mask]
    return out


def _mean_log(confidence: np.ndarray) -> np.ndarray:
    """Mean over labels of the log of a per-decision confidence (length-normalised log-probability)."""
    with np.errstate(invalid="ignore"):
        return np.log(np.clip(confidence, 1e-9, 1.0)).mean(axis=1)


def _baseline_probabilities(source: str, item_ids: pd.Series, labels) -> np.ndarray | None:
    """P(label = 1) from the TF-IDF baseline for the given items, if it has been run."""
    path = BASELINE_DIR / f"tfidf_{source}.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path).drop_duplicates("item_id").set_index("item_id")
    if not set(item_ids) <= set(table.index):
        return None
    return table.loc[list(item_ids), [f"p_{label}" for label in labels]].to_numpy(dtype=float)


def item_signals(data: dict) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """One row per item with a complete main batch: correctness and every item-level signal.

    Also returns the decision-level signals as (items, labels) arrays. Higher always means more
    confident. A signal that could not be computed for an item is NaN there.
    """
    keep = data["complete"][:, 0]
    items = data["items"][keep].reset_index(drop=True)
    gold, vote, samples = data["gold"][keep], data["vote"][keep], data["samples"][keep, 0]
    p1_forced, p1_greedy = data["p1_forced"][keep], data["p1_greedy"][keep]
    verbal_overall = data["verbal_overall"][keep]
    n = len(items)

    forced = signals.decision_confidence(p1_forced, vote)
    from_greedy = signals.decision_confidence(p1_greedy, vote)
    jaccard = np.array([signals.mean_pairwise_jaccard(s) for s in samples]) if n else np.zeros(0)
    rng = np.random.default_rng(0)

    table = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "article_id": items["article_id"],
            "correct": (vote == gold).all(axis=1),
            "gold_empty": gold.sum(axis=1) == 0,
            "n_gold": gold.sum(axis=1),
            "n_predicted": vote.sum(axis=1),
            "n_words": items["text"].str.split().str.len(),
            PRIMARY: jaccard,
            "consistency: modal-set frequency": [signals.modal_set_frequency(s) for s in samples],
            "consistency: Hamming agreement": [signals.hamming_agreement(s) for s in samples],
            "log-probability: voted set, teacher-forced": _mean_log(forced),
            "log-probability: voted set, greedy pass": _mean_log(from_greedy),
            "verbalised confidence": verbal_overall,
        }
    )
    if "disinformative" in items:
        table["disinformative"] = items["disinformative"].to_numpy()
    # Decisions as bit strings in label order, for the quality after human correction.
    table["gold_bits"] = ["".join(map(str, row)) for row in gold]
    table["vote_bits"] = ["".join(map(str, row)) for row in vote]

    # Single-run policy: the greedy answer with its own log-probability (and stated confidence).
    greedy, has_greedy = data["greedy"][keep], data["has_greedy"][keep]
    table["greedy_correct"] = np.where(has_greedy, (greedy == gold).all(axis=1), False)
    table["greedy_bits"] = ["".join(map(str, row)) if ok else "" for row, ok in zip(greedy, has_greedy)]
    table["single run: own log-probability"] = np.where(
        has_greedy, _mean_log(signals.decision_confidence(p1_greedy, greedy)), np.nan
    )
    if data["config"].get("verbal_greedy"):
        table["single run: verbalised confidence"] = np.where(has_greedy, data["verbal_greedy_overall"][keep], np.nan)
    decision = {
        "consistency: vote share": np.stack([signals.label_agreement(s) for s in samples]) if n else np.zeros((0, 0)),
        "log-probability: teacher-forced": forced,
        "log-probability: greedy pass": from_greedy,
    }
    baseline = _baseline_probabilities(data["config"]["source"], items["item_id"], data["labels"])
    if baseline is not None:
        classical = signals.decision_confidence(baseline, vote)
        table["classical: TF-IDF logistic regression"] = _mean_log(classical)
        decision["classical: TF-IDF logistic regression"] = classical
        # The baseline on its own: its prediction, scored by its own probability.
        own = (baseline >= 0.5).astype(np.int8)
        table["tfidf_correct"] = (own == gold).all(axis=1)
        table["tfidf_bits"] = ["".join(map(str, row)) for row in own]
        table["tfidf alone: own probability"] = _mean_log(signals.decision_confidence(baseline, own))
    table["control: fewer predicted labels"] = -table["n_predicted"].astype(float)
    table["control: shorter text"] = -table["n_words"].astype(float)
    table["control: random"] = rng.random(n)
    table["control: shuffled Jaccard"] = rng.permutation(jaccard)
    return table, decision


def _signal_columns(table: pd.DataFrame) -> list[str]:
    families = ("consistency", "log-probability", "verbalised confidence", "classical", "control")
    return [c for c in table.columns if c.split(":")[0] in families]


def analyse(run_dir: Path, n_resamples: int = 1000, min_support: int = 30) -> dict:
    data = assemble(run_dir)
    # MIPD: techniques were annotated on disinformative articles only. Credible articles have no
    # technique by construction, so they are reported as a false-alarm check and left out of the
    # rest (PLAN.md, decision 4).
    credible_check = None
    if "disinformative" in data["items"]:
        credible = (data["items"]["disinformative"] == 0).to_numpy()
        voted = credible & data["complete"][:, 0]
        if voted.any():
            predicted = data["vote"][voted].sum(axis=1)
            credible_check = {
                "credible_items": int(voted.sum()),
                "share_with_a_predicted_technique": float((predicted > 0).mean()),
                "mean_predicted_techniques": float(predicted.mean()),
            }
        data = _subset(data, ~credible)
    table, decision = item_signals(data)
    if table.empty:
        if credible_check:
            raise ValueError(f"only credible articles in this run, nothing to score; false-alarm check: {credible_check}")
        raise ValueError("no item has a complete set of valid sampled runs")
    keep = data["complete"][:, 0]
    gold, vote, samples = data["gold"][keep], data["vote"][keep], data["samples"][keep]
    greedy, has_greedy, p1_greedy = data["greedy"][keep], data["has_greedy"][keep], data["p1_greedy"][keep]
    labels = data["labels"]
    correct = table["correct"].to_numpy()
    correct_label = vote == gold
    groups = table["article_id"].to_numpy()
    n = len(table)

    # Item level. The comparison uses the items for which every signal of the model is available.
    names = _signal_columns(table)
    available = [name for name in MODEL_SIGNALS if table[name].notna().any()]
    common = table[available].notna().all(axis=1).to_numpy()
    strata = {
        "all": common,
        "gold empty": common & table["gold_empty"].to_numpy(),
        "gold non-empty": common & ~table["gold_empty"].to_numpy(),
    }
    primary = table[PRIMARY].to_numpy()
    item_level = []
    for name in names:
        score = table[name].to_numpy(dtype=float)
        if np.isnan(score[common]).all():
            continue
        row = {"signal": name}
        for stratum, mask in strata.items():
            row[f"AUROC {stratum}"] = auroc(correct[mask], score[mask])
        row["CI low"], row["CI high"] = cluster_bootstrap_ci(
            lambda idx, s=score: auroc(correct[idx][common[idx]], s[idx][common[idx]]), groups, n_resamples
        )
        row["error PR-AUC"] = error_pr_auc(correct[common], score[common])
        if name != PRIMARY:
            # Paired difference to the primary consistency score, on the same resampled articles.
            row["diff to Jaccard"] = row["AUROC all"] - auroc(correct[common], primary[common])
            row["diff low"], row["diff high"] = cluster_bootstrap_ci(
                lambda idx, s=score: auroc(correct[idx][common[idx]], s[idx][common[idx]])
                - auroc(correct[idx][common[idx]], primary[idx][common[idx]]),
                groups,
                n_resamples,
            )
        row["distinct values"] = int(len(np.unique(score[common & ~np.isnan(score)])))
        item_level.append(row)

    # Decision level (item x label).
    label_strata = {
        "all decisions": np.ones_like(correct_label, dtype=bool),
        "predicted positive": vote == 1,
        "gold positive": gold == 1,
    }
    decision_level = []
    for name, score in decision.items():
        if np.isnan(score).all():
            continue
        row = {"signal": name}
        for stratum, mask in label_strata.items():
            row[f"AUROC {stratum}"] = auroc(correct_label[mask], score[mask])
        decision_level.append(row)

    # Per technique, for techniques with enough gold positives among the analysed items.
    per_label = []
    for j, label in enumerate(labels):
        support = int(gold[:, j].sum())
        if support < min_support:
            continue
        row = {"label": label, "gold positives": support, "accuracy": float(correct_label[:, j].mean())}
        for name, score in decision.items():
            if not np.isnan(score).all():
                row[name] = auroc(correct_label[:, j], score[:, j])
        per_label.append(row)

    # Task quality of the voted prediction and of single runs.
    quality = {"majority vote of 5 sampled runs": task_quality(gold, vote)}
    single = [task_quality(gold, samples[:, 0, r, :]) for r in range(samples.shape[2])]
    quality["single sampled run (mean over runs)"] = {key: float(np.mean([q[key] for q in single])) for key in single[0]}
    if has_greedy.any():
        quality["single greedy run"] = task_quality(gold[has_greedy], greedy[has_greedy])

    # Complete policies: what is accepted is the prediction of that policy, scored by its own signal.
    # Each policy is costed as if run on its own: n_samples sampled runs plus one scoring or
    # stated-confidence request per item, at the measured seconds per request of each stage.
    stage_seconds = _stage_seconds(run_dir)
    k = int(data["config"]["n_samples"])
    policies = [
        _policy(f"{k} sampled runs + consistency", correct, primary, {"sample": k}, stage_seconds),
        _policy(
            f"{k} sampled runs + teacher-forced log-probability",
            correct,
            table["log-probability: voted set, teacher-forced"].to_numpy(),
            {"sample": k, "score": 1},
            stage_seconds,
        ),
        _policy(
            f"{k} sampled runs + verbalised confidence",
            correct,
            table["verbalised confidence"].to_numpy(),
            {"sample": k, "verbal": 1},
            stage_seconds,
        ),
    ]
    if has_greedy.any():
        greedy_correct = (greedy == gold).all(axis=1)
        own = np.where(has_greedy, _mean_log(signals.decision_confidence(p1_greedy, greedy)), np.nan)
        policies.append(_policy("1 greedy run + its own log-probability", greedy_correct, own, {"greedy": 1}, stage_seconds))
        stated = np.where(has_greedy, data["verbal_greedy_overall"][keep], np.nan)
        # Only when the run asked about every greedy answer; otherwise the values cover just the
        # items whose greedy answer equals the vote, which is not a fair sample.
        if data["config"].get("verbal_greedy") and not np.isnan(stated).all():
            policies.append(
                _policy("1 greedy run + verbalised confidence", greedy_correct, stated, {"greedy": 1, "verbal": 1}, stage_seconds)
            )

    # Independent batches of sampled runs: how much the main numbers move between them.
    batches = []
    for b in range(samples.shape[1]):
        done = data["complete"][:, b]
        if not done.any():
            continue
        batch_gold, batch_samples = data["gold"][done], data["samples"][done, b]
        batch_vote = np.stack([signals.majority_vote(s) for s in batch_samples])
        batch_jaccard = np.array([signals.mean_pairwise_jaccard(s) for s in batch_samples])
        batch_correct = (batch_vote == batch_gold).all(axis=1)
        batches.append(
            {
                "batch": b,
                "items": int(done.sum()),
                "micro_f1": task_quality(batch_gold, batch_vote)["micro_f1"],
                "exact_set_accuracy": float(batch_correct.mean()),
                "AUROC Jaccard": auroc(batch_correct, batch_jaccard),
                "fully consistent": float((batch_jaccard == 1.0).mean()),
            }
        )

    # Server sanity checks. A greedy decision must be the more probable of the two decision tokens;
    # if it is not, the server changes the distribution after the model (for example with a
    # repetition penalty) and its sampler settings have to be pinned before any real run.
    # An exact tie (probability 0.5, which half-precision logits produce now and then) agrees
    # with either decision; ties are counted separately.
    have_p1 = ~np.isnan(p1_greedy).any(axis=1)
    checked = has_greedy & have_p1
    greedy_matches_probability = float("nan")
    exact_ties = 0
    if checked.any():
        p, d = p1_greedy[checked], greedy[checked]
        exact_ties = int((p == 0.5).sum())
        greedy_matches_probability = float(((p == 0.5) | (d == (p > 0.5))).mean())
    # Sampled decisions should follow the same probabilities, sharpened by the temperature. The
    # two rates below agree only roughly (each sampled run conditions on its own earlier
    # decisions), but a large gap means the sampler does not do what the config says.
    temperature = float(data["config"].get("temperature", 1.0))
    expected_positive_rate = sampled_positive_rate = float("nan")
    if checked.any() and temperature > 0:
        p = np.clip(p1_greedy[checked], 1e-9, 1 - 1e-9)
        sharpened = p ** (1 / temperature) / (p ** (1 / temperature) + (1 - p) ** (1 / temperature))
        expected_positive_rate = float(sharpened.mean())
        sampled_positive_rate = float(samples[checked, 0].mean())

    generations = data["generations"]
    ordered = generations["ordered"].dropna() if "ordered" in generations else pd.Series(dtype=bool)
    by_kind = generations.groupby("kind")["valid"].agg(["size", "sum"])
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []

    summary = {
        "run": data["config"]["run_name"],
        "source": data["config"]["source"],
        "role": data["config"]["role"],
        "prompt": data["config"]["prompt"],
        "model": data["config"]["engine"].get("model"),
        "n_items_in_run": int(len(data["items"])),
        "n_items_complete": int(n),
        "n_items_all_signals": int(common.sum()),
        "n_forced_from_greedy": int(data["forced_from_greedy"][keep].sum()),
        "n_articles": int(len(np.unique(groups))),
        "requests": {kind: {"total": int(r["size"]), "invalid": int(r["size"] - r["sum"])} for kind, r in by_kind.iterrows()},
        "invalid_generations": int((~generations["valid"]).sum()),
        "generations": int(len(generations)),
        "share_gold_empty": float(table["gold_empty"].mean()),
        "share_vote_correct": float(correct.mean()),
        "share_fully_consistent": float((primary == 1.0).mean()),
        "accuracy_when_fully_consistent": float(correct[primary == 1.0].mean()) if (primary == 1.0).any() else float("nan"),
        "accuracy_when_not_fully_consistent": float(correct[primary < 1.0].mean()) if (primary < 1.0).any() else float("nan"),
        "labels_with_bounded_probability": int(generations["n_bounded"].fillna(0).sum()),
        "greedy_matches_probability": greedy_matches_probability,
        "greedy_exact_ties": exact_ties,
        "keys_in_label_order": float(ordered.astype(bool).mean()) if len(ordered) else float("nan"),
        "sampled_positive_rate": sampled_positive_rate,
        "expected_positive_rate": expected_positive_rate,
        "stage_seconds_per_item": stage_seconds,
        "output_tokens_per_second": [m.get("output_tokens_per_second") for m in meta],
        "wall_seconds": [m.get("wall_seconds") for m in meta],
        "gpu_memory_used_mib": [m.get("gpu_memory_used_mib") for m in meta],
        "credible_articles": credible_check,
        "task_quality": quality,
        "item_level": item_level,
        "decision_level": decision_level,
        "per_label": per_label,
        "policies": policies,
        "batches": batches,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    table.to_csv(run_dir / "items.csv", index=False)
    return summary


def _stage_seconds(run_dir: Path) -> dict[str, float]:
    """Wall-clock seconds per request for each kind of request, over all resumed invocations.

    Per request, not per item: a run with several seed batches, or one that scores only some
    items, would otherwise charge a policy for requests it does not make.
    """
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    seconds: dict[str, float] = {}
    requests: dict[str, int] = {}
    for entry in json.loads(meta_path.read_text(encoding="utf-8")):
        for stage, numbers in entry.get("stages", {}).items():
            seconds[stage] = seconds.get(stage, 0.0) + float(numbers["wall_seconds"])
            requests[stage] = requests.get(stage, 0) + int(numbers["generations"])
    return {stage: round(seconds[stage] / requests[stage], 4) for stage in seconds if requests[stage]}


def _policy(name: str, correct: np.ndarray, score: np.ndarray, requests: dict[str, int], seconds: dict) -> dict:
    have = ~np.isnan(score)
    known = all(stage in seconds for stage in requests)
    return {
        "policy": name,
        "requests per item": sum(requests.values()),
        "seconds per item": round(sum(seconds[s] * n for s, n in requests.items()), 3) if known else float("nan"),
        "exact-set accuracy": float(correct[have].mean()) if have.any() else float("nan"),
        "AUROC": auroc(correct[have], score[have]),
        "items": int(have.sum()),
    }


def _frame(rows: list[dict], index: str) -> str:
    if not rows:
        return "(not available)"
    return pd.DataFrame(rows).set_index(index).round(3).to_string()


def report(summary: dict) -> str:
    pd.set_option("display.width", 250)
    requests = ", ".join(f"{kind} {r['total']} ({r['invalid']} invalid)" for kind, r in summary["requests"].items())
    lines = [
        f"run {summary['run']}: {summary['model']} on {summary['source']} ({summary['role']}), prompt {summary['prompt']}",
        f"items with complete runs: {summary['n_items_complete']} of {summary['n_items_in_run']} "
        f"from {summary['n_articles']} articles; with every signal: {summary['n_items_all_signals']}",
        f"requests: {requests}",
        f"teacher-forced probabilities taken from the greedy pass (greedy answer = vote): "
        f"{summary.get('n_forced_from_greedy', 0)} items; scored separately: "
        f"{summary['requests'].get('score', {}).get('total', 0)} requests",
        f"gold empty: {summary['share_gold_empty']:.1%}; voted set exactly right: {summary['share_vote_correct']:.1%}; "
        f"all runs identical: {summary['share_fully_consistent']:.1%} "
        f"(exactly right when identical: {summary['accuracy_when_fully_consistent']:.1%}, "
        f"otherwise: {summary['accuracy_when_not_fully_consistent']:.1%})",
        f"server checks: greedy decisions agreeing with their token probability "
        f"{summary['greedy_matches_probability']:.1%} (must be 100%; {summary.get('greedy_exact_ties', 0)} exact ties); "
        f"answers with keys in label order {summary['keys_in_label_order']:.1%}",
        f"               sampled decisions that are 1: {summary['sampled_positive_rate']:.2%}; expected from the "
        f"greedy probabilities at this temperature: about {summary['expected_positive_rate']:.2%}",
        "",
        "Task quality",
        pd.DataFrame(summary["task_quality"]).T.round(3).to_string(),
        "",
        "Item level: does the score predict that the voted label set is exactly right?",
        _frame(summary["item_level"], "signal"),
        "",
        "Decision level (item x label): does the score predict that the voted decision is right?",
        _frame(summary["decision_level"], "signal"),
        "",
        "Per technique (decision-level AUROC, techniques with enough gold positives)",
        _frame(summary["per_label"], "label"),
        "",
        "Complete policies (each scored on its own prediction)",
        _frame(summary["policies"], "policy"),
    ]
    if summary.get("credible_articles"):
        check = summary["credible_articles"]
        lines.insert(
            4,
            f"credible articles (left out below): {check['credible_items']}, of which "
            f"{check['share_with_a_predicted_technique']:.1%} got at least one technique from the vote",
        )
    if len(summary["batches"]) > 1:
        lines += ["", "Independent batches of sampled runs", _frame(summary["batches"], "batch")]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples for the intervals")
    parser.add_argument("--min-support", type=int, default=30, help="gold positives a technique needs for its own row")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    print(report(analyse(run_dir, args.resamples, args.min_support)))


if __name__ == "__main__":
    main()

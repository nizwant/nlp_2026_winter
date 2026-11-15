"""Score one run against gold: task quality, and how well each signal predicts correctness.

    python -m routing.analyze outputs/runs/<run_name>

Prints the tables and writes summary.json next to the log. All signals score the same
prediction, the per-label majority vote over the sampled runs.
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
    labels = labels_for(source)
    generations = load_generations(run_dir)

    items = load_role(source, config["role"])
    items = items[items["item_id"].isin(set(generations["item_id"]))].reset_index(drop=True)
    gold = gold_matrix(items, source)

    valid = generations[generations["valid"]]
    sampled = {key: group.sort_values("seed") for key, group in valid[valid["kind"] == "sample"].groupby("item_id")}
    greedy = valid[valid["kind"] == "greedy"].set_index("item_id")

    n_items, n_labels = len(items), len(labels)
    samples = np.zeros((n_items, n_samples, n_labels), dtype=np.int8)
    complete = np.zeros(n_items, dtype=bool)
    greedy_decisions = np.zeros((n_items, n_labels), dtype=np.int8)
    p1 = np.full((n_items, n_labels), np.nan)
    has_greedy = np.zeros(n_items, dtype=bool)
    for row, item_id in enumerate(items["item_id"]):
        group = sampled.get(item_id)
        if group is not None and len(group) >= n_samples:
            samples[row] = np.array(group["labels"].tolist()[:n_samples], dtype=np.int8)
            complete[row] = True
        if item_id in greedy.index:
            record = greedy.loc[item_id]
            greedy_decisions[row] = record["labels"]
            has_greedy[row] = True
            if record["p1"] is not None:
                p1[row] = record["p1"]

    return {
        "config": config,
        "labels": labels,
        "items": items,
        "gold": gold,
        "samples": samples,
        "complete": complete,
        "greedy": greedy_decisions,
        "has_greedy": has_greedy,
        "p1": p1,
        "generations": generations,
    }


def analyse(run_dir: Path, n_resamples: int = 1000) -> dict:
    data = assemble(run_dir)
    keep = data["complete"]
    items = data["items"][keep].reset_index(drop=True)
    gold, samples = data["gold"][keep], data["samples"][keep]
    p1, greedy, has_greedy = data["p1"][keep], data["greedy"][keep], data["has_greedy"][keep]
    groups = items["article_id"].to_numpy()
    n = len(items)
    if n == 0:
        raise ValueError("no item has a complete set of valid sampled runs")

    vote = np.stack([signals.majority_vote(s) for s in samples])
    correct_item = (vote == gold).all(axis=1)
    correct_label = vote == gold
    gold_empty = gold.sum(axis=1) == 0
    have_p1 = ~np.isnan(p1).any(axis=1)

    rng = np.random.default_rng(0)
    jaccard = np.array([signals.mean_pairwise_jaccard(s) for s in samples])
    item_signals = {
        "consistency: mean pairwise Jaccard": jaccard,
        "consistency: modal-set frequency": np.array([signals.modal_set_frequency(s) for s in samples]),
        "consistency: Hamming agreement": np.array([signals.hamming_agreement(s) for s in samples]),
        "log-probability of voted set": np.array(
            [signals.joint_log_probability(p, v) if ok else np.nan for p, v, ok in zip(p1, vote, have_p1)]
        ),
        "control: fewer predicted labels": -vote.sum(axis=1).astype(float),
        "control: shorter text": -items["text"].str.split().str.len().to_numpy(dtype=float),
        "control: random": rng.random(n),
        "control: shuffled Jaccard": rng.permutation(jaccard),
    }

    strata = {"all": np.ones(n, dtype=bool), "gold empty": gold_empty, "gold non-empty": ~gold_empty}
    item_table = []
    for name, score in item_signals.items():
        row = {"signal": name}
        for stratum, mask in strata.items():
            row[f"AUROC {stratum}"] = auroc(correct_item[mask], score[mask])
        low, high = cluster_bootstrap_ci(lambda idx, s=score: auroc(correct_item[idx], s[idx]), groups, n_resamples)
        row["CI low"], row["CI high"] = low, high
        row["error PR-AUC"] = error_pr_auc(correct_item, score)
        item_table.append(row)

    agreement = np.stack([signals.label_agreement(s) for s in samples])
    label_signals = {"consistency: vote share": agreement}
    if have_p1.any():
        label_signals["log-probability of voted decision"] = np.where(
            have_p1[:, None], signals.decision_confidence(p1, vote), np.nan
        )
    label_strata = {
        "all decisions": np.ones_like(correct_label, dtype=bool),
        "predicted positive": vote == 1,
        "gold positive": gold == 1,
    }
    label_table = []
    for name, score in label_signals.items():
        row = {"signal": name}
        for stratum, mask in label_strata.items():
            row[f"AUROC {stratum}"] = auroc(correct_label[mask], score[mask])
        label_table.append(row)

    quality = {"majority vote of sampled runs": task_quality(gold, vote)}
    single = [task_quality(gold, samples[:, r, :]) for r in range(samples.shape[1])]
    quality["single sampled run (mean over runs)"] = {
        key: float(np.mean([q[key] for q in single])) for key in single[0]
    }
    if has_greedy.any():
        quality["single greedy run"] = task_quality(gold[has_greedy], greedy[has_greedy])

    # Server sanity checks. A greedy decision must be the more probable of the two decision tokens;
    # if it is not, the server changes the distribution after the model (for example with a
    # repetition penalty) and its sampler settings have to be pinned before any real run.
    checked = has_greedy & have_p1
    greedy_matches_probability = (
        float((greedy[checked] == (p1[checked] >= 0.5)).mean()) if checked.any() else float("nan")
    )
    # Sampled decisions should follow the same probabilities, sharpened by the temperature. The
    # two rates below agree only roughly (each sampled run conditions on its own earlier
    # decisions), but a large gap means the sampler does not do what the config says.
    temperature = float(data["config"].get("temperature", 1.0))
    expected_positive_rate = sampled_positive_rate = float("nan")
    if checked.any() and temperature > 0:
        p = np.clip(p1[checked], 1e-9, 1 - 1e-9)
        sharpened = p ** (1 / temperature) / (p ** (1 / temperature) + (1 - p) ** (1 / temperature))
        expected_positive_rate = float(sharpened.mean())
        sampled_positive_rate = float(samples[checked].mean())

    generations = data["generations"]
    ordered = generations["ordered"].dropna() if "ordered" in generations else pd.Series(dtype=bool)
    keys_in_label_order = float(ordered.astype(bool).mean()) if len(ordered) else float("nan")

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
        "n_articles": int(len(np.unique(groups))),
        "invalid_generations": int((~generations["valid"]).sum()),
        "generations": int(len(generations)),
        "share_gold_empty": float(gold_empty.mean()),
        "share_vote_correct": float(correct_item.mean()),
        "share_fully_consistent": float((jaccard == 1.0).mean()),
        "labels_with_bounded_probability": int(generations["n_bounded"].fillna(0).sum()),
        "greedy_matches_probability": greedy_matches_probability,
        "keys_in_label_order": keys_in_label_order,
        "sampled_positive_rate": sampled_positive_rate,
        "expected_positive_rate": expected_positive_rate,
        "output_tokens_per_second": [m.get("output_tokens_per_second") for m in meta],
        "wall_seconds": [m.get("wall_seconds") for m in meta],
        "task_quality": quality,
        "item_level": item_table,
        "label_level": label_table,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def report(summary: dict) -> str:
    pd.set_option("display.width", 200)
    lines = [
        f"run {summary['run']}: {summary['model']} on {summary['source']} ({summary['role']}), prompt {summary['prompt']}",
        f"items with complete runs: {summary['n_items_complete']} of {summary['n_items_in_run']} "
        f"from {summary['n_articles']} articles; invalid generations: {summary['invalid_generations']} "
        f"of {summary['generations']}",
        f"gold empty: {summary['share_gold_empty']:.1%}; voted set exactly right: "
        f"{summary['share_vote_correct']:.1%}; all runs identical: {summary['share_fully_consistent']:.1%}",
        f"server checks: greedy decisions agreeing with their token probability "
        f"{summary['greedy_matches_probability']:.1%} (must be 100%); answers with keys in label order "
        f"{summary['keys_in_label_order']:.1%}",
        f"               sampled decisions that are 1: {summary['sampled_positive_rate']:.2%}; expected from the "
        f"greedy probabilities at this temperature: about {summary['expected_positive_rate']:.2%}",
        "",
        "Task quality",
        pd.DataFrame(summary["task_quality"]).T.round(3).to_string(),
        "",
        "Item level: does the score predict that the voted label set is exactly right?",
        pd.DataFrame(summary["item_level"]).set_index("signal").round(3).to_string(),
        "",
        "Decision level (item x label): does the score predict that the voted decision is right?",
        pd.DataFrame(summary["label_level"]).set_index("signal").round(3).to_string(),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples for the intervals")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    print(report(analyse(run_dir, args.resamples)))


if __name__ == "__main__":
    main()

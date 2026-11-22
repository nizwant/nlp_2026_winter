"""Routing evaluation: thresholds chosen on a calibration run, applied to an evaluation run.

    python -m routing.route --calibration outputs/runs/<calibration run> --evaluation outputs/runs/<other run>

Both runs must come from the same model, source and prompt. For every policy (a prediction and
the confidence score that decides whether it is accepted) and every accuracy target:

1. the threshold is chosen on the calibration run (routing.selective.choose_threshold);
2. it is applied unchanged to the evaluation run;
3. coverage, accepted-item accuracy (with an article-level bootstrap interval), the quality
   after ideal human correction of the routed items, and the cost are reported.

A target that no threshold reaches on the calibration run is reported as infeasible. Writes
routing.json and risk_coverage.csv into the evaluation run folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import cost as cost_model
from . import selective
from .analyze import PRIMARY, analyse
from .data import REPO_ROOT
from .metrics import cluster_bootstrap_ci, task_quality

# policy name -> (column with the score, column with the correctness of the prediction it accepts,
#                 column with that prediction, stages whose GPU time the policy needs)
POLICIES = {
    "5 runs + consistency (Jaccard)": (PRIMARY, "correct", "vote_bits", ["sample"]),
    "5 runs + modal-set frequency": ("consistency: modal-set frequency", "correct", "vote_bits", ["sample"]),
    "5 runs + teacher-forced log-probability": (
        "log-probability: voted set, teacher-forced",
        "correct",
        "vote_bits",
        ["sample", "score"],
    ),
    "5 runs + verbalised confidence": ("verbalised confidence", "correct", "vote_bits", ["sample", "verbal"]),
    "5 runs + TF-IDF probability": ("classical: TF-IDF logistic regression", "correct", "vote_bits", ["sample"]),
    "5 runs + fewer predicted labels (control)": ("control: fewer predicted labels", "correct", "vote_bits", ["sample"]),
    "5 runs + random (control)": ("control: random", "correct", "vote_bits", ["sample"]),
    "1 greedy run + own log-probability": ("single run: own log-probability", "greedy_correct", "greedy_bits", ["greedy"]),
    "1 greedy run + verbalised confidence": (
        "single run: verbalised confidence",
        "greedy_correct",
        "greedy_bits",
        ["greedy", "verbal"],
    ),
    "TF-IDF alone + own probability": ("tfidf alone: own probability", "tfidf_correct", "tfidf_bits", []),
}


def _items(run_dir: Path, refresh: bool) -> tuple[pd.DataFrame, dict]:
    if refresh or not (run_dir / "items.csv").exists() or not (run_dir / "summary.json").exists():
        analyse(run_dir, n_resamples=200)
    table = pd.read_csv(run_dir / "items.csv", dtype={"gold_bits": str, "vote_bits": str, "greedy_bits": str, "tfidf_bits": str})
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return table, summary


def _bits(column: pd.Series) -> np.ndarray:
    return np.array([[int(ch) for ch in bits] for bits in column], dtype=np.int8)


def route(
    calibration_dir: Path,
    evaluation_dir: Path,
    targets: tuple[float, ...] = selective.TARGETS,
    min_accepted: int = selective.MIN_ACCEPTED,
    n_resamples: int = 1000,
    refresh: bool = False,
) -> dict:
    calibration, calibration_summary = _items(calibration_dir, refresh)
    evaluation, evaluation_summary = _items(evaluation_dir, refresh)
    for key in ("model", "source", "prompt"):
        if calibration_summary[key] != evaluation_summary[key]:
            raise ValueError(f"the two runs differ in {key}: {calibration_summary[key]} vs {evaluation_summary[key]}")
    source = evaluation_summary["source"]
    groups = evaluation["article_id"].to_numpy()
    gold = _bits(evaluation["gold_bits"])
    seconds = evaluation_summary.get("stage_seconds_per_item") or {}
    rates = cost_model.load_rates()
    human_seconds = (rates.get("human_seconds_per_item") or {}).get(source)

    rows, curves = [], []
    for name, (score_column, correct_column, bits_column, stages) in POLICIES.items():
        if score_column not in calibration or score_column not in evaluation:
            continue
        cal_score = calibration[score_column].to_numpy(dtype=float)
        cal_correct = calibration[correct_column].to_numpy(dtype=bool)
        score = evaluation[score_column].to_numpy(dtype=float)
        correct = evaluation[correct_column].to_numpy(dtype=bool)
        if np.isnan(cal_score).all() or np.isnan(score).all():
            continue
        has_prediction = evaluation[bits_column].fillna("").str.len() > 0
        predicted = np.where(
            has_prediction.to_numpy()[:, None],
            _bits(evaluation[bits_column].where(has_prediction, evaluation["gold_bits"])),
            0,
        )
        gpu_seconds = sum(seconds.get(stage, float("nan")) for stage in stages) if stages else 0.0

        curve = selective.risk_coverage(correct, score)
        curve.insert(0, "policy", name)
        curves.append(curve)
        calibrator = selective.fit_calibrator(cal_correct, cal_score)
        with np.errstate(invalid="ignore"):
            calibrated = np.where(np.isnan(score), np.nan, calibrator.predict(np.nan_to_num(score)))
        common = {
            "policy": name,
            "gpu_seconds_per_item": round(float(gpu_seconds), 3),
            "prediction_exact_accuracy": float(correct.mean()),
            "area_under_risk_coverage": selective.area_under_risk_coverage(correct, score),
            "expected_calibration_error": selective.expected_calibration_error(correct, calibrated),
        }
        for target in targets:
            threshold = selective.choose_threshold(cal_correct, cal_score, target, min_accepted)
            row = {**common, "target": target, "feasible": threshold is not None, "threshold": threshold}
            row["calibration_coverage"] = selective.evaluate(cal_correct, cal_score, threshold)["coverage"]
            row.update(selective.evaluate(correct, score, threshold))
            accepted = selective.accepted_mask(score, threshold)
            if threshold is not None and accepted.any():
                row["accuracy_low"], row["accuracy_high"] = cluster_bootstrap_ci(
                    lambda idx: float(correct[idx][accepted[idx]].mean()) if accepted[idx].any() else float("nan"),
                    groups,
                    n_resamples,
                )
                row["coverage_low"], row["coverage_high"] = cluster_bootstrap_ci(
                    lambda idx: float(accepted[idx].mean()), groups, n_resamples
                )
            # Quality after a human has corrected every routed item (gold stands in for the human).
            final = np.where(accepted[:, None], predicted, gold)
            row["final_micro_f1"] = task_quality(gold, final)["micro_f1"]
            if human_seconds is not None and not np.isnan(gpu_seconds):
                money = cost_model.policy_cost(1000, row["coverage"], human_seconds, gpu_seconds, rates)
                row["human_hours_saved_per_1000"] = round(money["human_hours_saved"], 2)
                row["gpu_hours_per_1000"] = round(money["gpu_hours"], 3)
                row["net_saving_per_1000"] = round(money["net_saving"], 2)
                row["break_even_coverage"] = round(cost_model.break_even_coverage(human_seconds, gpu_seconds, rates), 4)
            rows.append(row)

    result = {
        "calibration_run": calibration_summary["run"],
        "evaluation_run": evaluation_summary["run"],
        "model": evaluation_summary["model"],
        "source": source,
        "prompt": evaluation_summary["prompt"],
        "n_calibration_items": int(len(calibration)),
        "n_evaluation_items": int(len(evaluation)),
        "min_accepted": min_accepted,
        "human_seconds_per_item": human_seconds,
        "cost_rates": {k: v for k, v in rates.items() if k != "human_seconds_per_item"},
        "rows": rows,
    }
    (evaluation_dir / "routing.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(evaluation_dir / "risk_coverage.csv", index=False)
    return result


def report(result: dict) -> str:
    pd.set_option("display.width", 250)
    lines = [
        f"{result['model']} on {result['source']}, prompt {result['prompt']}: thresholds from "
        f"{result['calibration_run']} ({result['n_calibration_items']} items), applied to "
        f"{result['evaluation_run']} ({result['n_evaluation_items']} items); at least "
        f"{result['min_accepted']} accepted calibration items required",
    ]
    table = pd.DataFrame(result["rows"])
    if table.empty:
        return lines[0] + "\n(no policy has scores in both runs)"
    for target, part in table.groupby("target", sort=False):
        columns = ["policy", "feasible", "coverage", "accepted_accuracy", "accuracy_low", "accuracy_high",
                   "final_exact_accuracy", "final_micro_f1", "gpu_seconds_per_item"]
        columns += [c for c in ("human_hours_saved_per_1000", "net_saving_per_1000") if c in part]
        columns = [c for c in columns if c in part]
        lines += ["", f"Target: {target:.0%} of the accepted label sets exactly right", part[columns].set_index("policy").round(3).to_string()]
    overall = table.drop_duplicates("policy")[["policy", "prediction_exact_accuracy", "area_under_risk_coverage", "expected_calibration_error"]]
    lines += ["", "Independent of the target", overall.set_index("policy").round(3).to_string()]
    if result["human_seconds_per_item"] is None:
        lines += ["", "No cost columns: human_seconds_per_item is not set for this source in configs/cost.yaml."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--targets", type=float, nargs="+", default=list(selective.TARGETS))
    parser.add_argument("--min-accepted", type=int, default=selective.MIN_ACCEPTED)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--refresh", action="store_true", help="recompute items.csv of both runs first")
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else REPO_ROOT / p  # noqa: E731
    result = route(
        resolve(args.calibration), resolve(args.evaluation), tuple(args.targets), args.min_accepted, args.resamples, args.refresh
    )
    print(report(result))


if __name__ == "__main__":
    main()

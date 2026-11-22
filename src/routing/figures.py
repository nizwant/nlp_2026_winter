"""Figures for the slides and the report, from analysed runs (no model, no GPU).

    python -m routing.figures outputs/runs/bielik_semeval_pl_calibration_p1
    python -m routing.figures <run_dir> --out latex/img --name poc

Needs `python -m routing.analyze <run_dir>` first (it writes summary.json and items.csv).
Writes, as PDF for LaTeX and PNG for slides:

    <name>_signals         AUROC of every signal with its 95% article-bootstrap interval
    <name>_risk_coverage   accuracy of the accepted items against the share accepted
    <name>_set_size        how often the vote is exactly right, by number of predicted techniques

Colours follow the EDA notebook: one fixed colour per signal family across all figures, controls
in grey, text in ink and muted grey, never in a series colour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import REPO_ROOT  # noqa: E402
from .selective import risk_coverage  # noqa: E402

INK, MUTED, GRID, CONTROL = "#0b0b0b", "#52514e", "#e4e3de", "#a3a29d"
FAMILY_COLOUR = {
    "consistency": "#2a78d6",  # blue
    "probability": "#1baf7a",  # aqua
    "classical": "#eb6834",  # orange
    "stated": INK,
    "control": CONTROL,
}

# Signals in display order: (column in summary/items, label on the slide, family)
SIGNALS = [
    ("consistency: mean pairwise Jaccard", "Agreement of 5 runs (Jaccard, main)", "consistency"),
    ("consistency: modal-set frequency", "Share of runs giving the modal set", "consistency"),
    ("consistency: Hamming agreement", "Agreement per technique (Hamming)", "consistency"),
    ("log-probability: voted set, teacher-forced", "Token probability of the vote", "probability"),
    ("verbalised confidence", "Stated confidence (0-100)", "stated"),
    ("classical: TF-IDF logistic regression", "TF-IDF classifier probability", "classical"),
    ("control: fewer predicted labels", "Control: fewer predicted techniques", "control"),
    ("control: shorter text", "Control: shorter paragraph", "control"),
    ("control: random", "Control: random score", "control"),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.pdf", out / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=220)
    plt.close(fig)
    return paths


def _caption(summary: dict) -> str:
    return (
        f"{summary['model']}, SemEval PL {summary['role']} set, prompt {summary['prompt'].upper()}, "
        f"{summary['n_items_complete']:,} paragraphs from {summary['n_articles']} articles"
    )


def _footnote(ax, text: str) -> None:
    """Source line under the x-axis label, left-aligned with the plot."""
    ax.annotate(text, xy=(0, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
                ha="left", va="top", fontsize=7, color=MUTED)


def signals_figure(summary: dict, out: Path, stem: str) -> list[Path]:
    rows = {row["signal"]: row for row in summary["item_level"]}
    shown = [(rows[c], label, family) for c, label, family in SIGNALS if c in rows]
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(shown) + 0.9))
    y = np.arange(len(shown))[::-1]
    for yi, (row, label, family) in zip(y, shown):
        colour = FAMILY_COLOUR[family]
        ax.plot([row["CI low"], row["CI high"]], [yi, yi], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(row["AUROC all"], yi, "o", color=colour, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{row['AUROC all']:.2f}", (row["CI high"], yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7, color=MUTED)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.annotate("chance", (0.5, y[0] + 0.6), xytext=(3, 0), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_yticks(y, [label for _, label, _ in shown])
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.6, len(shown) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC for 'the voted label set is exactly right' (95% interval, bootstrap over articles)")
    ax.set_title("Which signal tells right answers from wrong ones?", loc="left", fontweight="bold")
    _footnote(ax, _caption(summary))
    return _save(fig, out, f"{stem}_signals")


def risk_coverage_figure(summary: dict, items: pd.DataFrame, out: Path, stem: str) -> list[Path]:
    correct = items["correct"].to_numpy(dtype=bool)
    lines = [
        ("consistency: mean pairwise Jaccard", "Agreement of 5 runs", FAMILY_COLOUR["consistency"], "-"),
        ("log-probability: voted set, teacher-forced", "Token probability", FAMILY_COLOUR["probability"], "-"),
        ("classical: TF-IDF logistic regression", "TF-IDF classifier", FAMILY_COLOUR["classical"], "-"),
        ("control: fewer predicted labels", "Control: fewer predicted techniques", CONTROL, (0, (4, 2))),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    first = {}
    for column, label, colour, style in lines:
        if column not in items:
            continue
        curve = risk_coverage(correct, items[column].to_numpy(dtype=float))
        curve = curve[curve["accepted"] >= 20]  # the first few points are single items
        ax.step(curve["coverage"], curve["accepted_accuracy"], where="post", color=colour, linestyle=style,
                linewidth=2, label=label)
        first[column] = curve.iloc[0]
    # Tied scores are accepted together, so a coarse score cannot accept a small share.
    # Notes go where no curve runs: below-left of the consistency start, right of the control's step.
    for column, text, offset, ha in [
        ("consistency: mean pairwise Jaccard", "all 5 runs identical", (-6, -6), "right"),
        ("control: fewer predicted labels", "vote predicts no technique", (84, -6), "left"),
    ]:
        if column in first:
            point = first[column]
            ax.annotate(f"{text}:\n{point['coverage']:.0%} accepted, {point['accepted_accuracy']:.0%} right",
                        (point["coverage"], point["accepted_accuracy"]), xytext=offset, textcoords="offset points",
                        ha=ha, va="top", fontsize=7, color=MUTED)
    base = correct.mean()
    ax.axhline(base, color=CONTROL, linewidth=1, linestyle=(0, (1, 2)))
    ax.annotate(f"accept everything / at random: {base:.0%}", (1.0, base), xytext=(0, -10), textcoords="offset points",
                ha="right", fontsize=7, color=MUTED)
    for target in (0.9, 0.8):
        ax.axhline(target, color=MUTED, linewidth=0.6)
        ax.annotate(f"target {target:.0%}", (1.0, target), xytext=(-2, 3), textcoords="offset points", fontsize=7,
                    ha="right", color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of paragraphs accepted without a human (most confident first)")
    ax.set_ylabel("accepted label sets exactly right")
    ax.set_title("Accepting the most confident answers first", loc="left", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, handlelength=2.6)
    _footnote(ax, _caption(summary) + "; curves start at 20 accepted paragraphs")
    return _save(fig, out, f"{stem}_risk_coverage")


def set_size_figure(summary: dict, items: pd.DataFrame, out: Path, stem: str) -> list[Path]:
    size = items["n_predicted"].clip(upper=3)
    table = items.groupby(size)["correct"].agg(["mean", "size"])
    names = {0: "none", 1: "1", 2: "2", 3: "3 or more"}
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bars = ax.bar([names[i] for i in table.index], table["mean"], width=0.6, color=FAMILY_COLOUR["consistency"],
                  linewidth=0)
    for bar, (share, n) in zip(bars, table.itertuples(index=False)):
        ax.annotate(f"{share:.0%}\n(n={n})", (bar.get_x() + bar.get_width() / 2, share), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("techniques in the voted answer")
    ax.set_ylabel("voted label set exactly right")
    ax.set_title("The vote is right mostly when it predicts nothing", loc="left", fontweight="bold")
    _footnote(ax, _caption(summary))
    return _save(fig, out, f"{stem}_set_size")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "latex" / "img")
    parser.add_argument("--name", help="file name prefix (default: the run name)")
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    items = pd.read_csv(args.run_dir / "items.csv")
    stem = args.name or summary["run"]
    _style()
    written = [
        *signals_figure(summary, args.out, stem),
        *risk_coverage_figure(summary, items, args.out, stem),
        *set_size_figure(summary, items, args.out, stem),
    ]
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

"""End-to-end checks on the real prepared data with the fake engine. Skipped when data is absent."""

import asyncio
import math

import numpy as np
import pytest

from routing import data, splits
from routing.labels import SOURCES, labels_for
from routing.prompts import build_messages, technique_list

needs_data = pytest.mark.skipif(
    not (data.DATA_DIR / "sem_eval" / "pl.csv").exists(), reason="prepared datasets are not on this machine"
)


def test_truncate_words_keeps_the_original_prefix():
    text = "one  two\nthree four"
    assert data.truncate_words(text, 2) == ("one  two", True)
    assert data.truncate_words(text, 4) == (text, False)
    assert data.truncate_words(text, 10) == (text, False)


@pytest.mark.parametrize("source", sorted(SOURCES))
@pytest.mark.parametrize("language", ["pl", "en"])
def test_every_label_has_a_name_and_definition(source, language):
    listing = technique_list(source, language)
    lines = listing.splitlines()
    assert len(lines) == len(labels_for(source))
    for label, line in zip(labels_for(source), lines):
        assert line.startswith(f"- {label} (")
        assert len(line) > len(label) + 30


def test_prompt_variants_differ_only_in_instruction_language():
    polish = build_messages("semeval_pl", "p1", "Przykładowy akapit.")[0]["content"]
    english = build_messages("semeval_pl", "p2", "Przykładowy akapit.")[0]["content"]
    assert "Przykładowy akapit." in polish and "Przykładowy akapit." in english
    assert polish.startswith("Jesteś") and english.startswith("You are")
    assert polish.rstrip().endswith(">>>")  # the text comes last, so the prefix is shared
    with pytest.raises(NotImplementedError):
        build_messages("semeval_pl", "p3", "x")


def test_text_with_dollar_signs_is_inserted_verbatim():
    content = build_messages("semeval_en", "p1", "It cost $5 and $text more.")[0]["content"]
    assert "It cost $5 and $text more." in content


@needs_data
@pytest.mark.parametrize("source", sorted(SOURCES))
def test_splits_are_disjoint_and_deterministic(source):
    first = splits.make_split(source)
    second = splits.make_split(source)
    assert first.equals(second)
    splits.check_split(first)

    assert first["item_id"].is_unique
    official_test = {"semeval_pl": "dev", "semeval_en": "dev", "mipd": "test"}[source]
    assert set(first.loc[first["role"] == "test", "original_set"]) == {official_test}
    not_test = first[first["role"].isin(["fewshot", "pilot", "calibration"])]
    assert official_test not in set(not_test["original_set"])

    # No text is shared between test and the roles used for development and calibration.
    items = data.load_items(source).merge(first[["item_id", "role"]], on="item_id")
    text = items["text"].map(splits.normalise)
    assert not set(text[items["role"] == "test"]) & set(text[items["role"].isin(["fewshot", "pilot", "calibration"])])


@needs_data
def test_polish_test_set_is_the_whole_dev_split():
    split = splits.make_split("semeval_pl")
    assert (split["role"] == "test").sum() == (split["original_set"] == "dev").sum() == 800


@needs_data
def test_run_and_analyse_with_the_fake_engine(tmp_path, monkeypatch):
    from routing import analyze, run

    if not (data.DATA_DIR / "splits" / "semeval_pl.csv").exists():
        pytest.skip("run `python -m routing.splits` first")
    monkeypatch.setattr(run, "RUNS_DIR", tmp_path)
    config = {
        **run.DEFAULTS,
        "run_name": "fake_run",
        "source": "semeval_pl",
        "role": "pilot",
        "engine": {"kind": "fake", "model": "fake"},
        "limit": 60,
    }
    run_dir = asyncio.run(run.run(config, show_progress=False))

    lines = (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 60 * 6  # five samples and one greedy pass per item
    assert "text" not in lines[0]  # the log never stores the paragraph

    # A second invocation finds everything done and adds nothing.
    asyncio.run(run.run(config, show_progress=False))
    assert len((run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()) == 360

    summary = analyze.analyse(run_dir, n_resamples=50)
    assert summary["n_items_complete"] == 60
    assert summary["invalid_generations"] == 0
    by_name = {row["signal"]: row for row in summary["item_level"]}
    assert set(by_name) >= {"consistency: mean pairwise Jaccard", "log-probability of voted set", "control: random"}
    value = by_name["consistency: mean pairwise Jaccard"]["AUROC all"]
    assert math.isnan(value) or 0.0 <= value <= 1.0
    assert "Task quality" in analyze.report(summary)


@needs_data
def test_incomplete_items_are_counted_not_analysed(tmp_path, monkeypatch):
    from routing import analyze, run
    from routing.engines import FakeEngine

    if not (data.DATA_DIR / "splits" / "semeval_pl.csv").exists():
        pytest.skip("run `python -m routing.splits` first")
    monkeypatch.setattr(run, "RUNS_DIR", tmp_path)
    config = {
        **run.DEFAULTS,
        "run_name": "fake_invalid",
        "source": "semeval_pl",
        "role": "pilot",
        "engine": {"kind": "fake", "model": "fake"},
        "limit": 80,
    }
    run_dir = asyncio.run(run.run(config, engine=FakeEngine(invalid_rate=0.1), show_progress=False))
    summary = analyze.analyse(run_dir, n_resamples=20)
    assert summary["invalid_generations"] > 0
    assert summary["n_items_complete"] < summary["n_items_in_run"] == 80
    assert np.isfinite(summary["task_quality"]["majority vote of sampled runs"]["micro_f1"])

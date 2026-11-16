"""End-to-end checks on the real prepared data with the fake engine. Skipped when data is absent."""

import asyncio
import json
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


@needs_data
@pytest.mark.parametrize("source", sorted(SOURCES))
def test_few_shot_prompt_alternates_examples_and_answers(source):
    from routing.engines import parse_decisions
    from routing.prompts import fewshot_examples

    labels = labels_for(source)
    zero_shot = build_messages(source, "p1", "Target text.")
    few_shot = build_messages(source, "p3", "Target text.")
    examples = fewshot_examples(source)
    assert len(few_shot) == 2 * len(examples) + 1
    assert [m["role"] for m in few_shot] == ["user", "assistant"] * len(examples) + ["user"]
    # Same instructions as the zero-shot prompt, and the target text comes last.
    assert few_shot[0]["content"].startswith(zero_shot[0]["content"].split("Target text.")[0][:200])
    assert "Target text." in few_shot[-1]["content"] and "Target text." not in few_shot[0]["content"]
    for (_, gold), answer in zip(examples, few_shot[1::2]):
        assert parse_decisions(answer["content"], labels) == list(gold)
    # Examples come from the few-shot pool only, never from pilot, calibration or test.
    from routing.fewshot import select

    pool = set(data.load_role(source, "fewshot")["item_id"])
    assert set(select(source)) <= pool


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
    not_test = first[first["role"].isin(["fewshot", "pilot", "calibration", "credible_check"])]
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
    from routing.engines import FakeEngine

    engine = FakeEngine()
    run_dir = asyncio.run(run.run(config, engine=engine, show_progress=False))

    lines = (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"text"' not in lines[0]  # the log never stores the paragraph
    records = [json.loads(line) for line in lines]
    by_kind = {kind: [r for r in records if r["kind"] == kind] for kind in ("sample", "greedy", "score", "verbal")}
    # five samples, one greedy pass and one stated-confidence query per item ...
    assert [len(by_kind[k]) for k in ("sample", "greedy", "verbal")] == [300, 60, 60]
    # ... and a scoring request only where the vote differs from the greedy answer.
    votes = {r["item_id"]: r["labels"] for r in by_kind["verbal"]}  # the verbal query assesses the vote
    disagree = {r["item_id"] for r in by_kind["greedy"] if r["labels"] != votes[r["item_id"]]}
    assert 0 < len(disagree) < 60
    assert {r["item_id"] for r in by_kind["score"]} == disagree
    assert len(lines) == 60 * 7 + len(disagree)
    # Every answer shown back to the model (scoring, stated confidence) uses its own greedy layout.
    assert len(engine.templates_seen) == len(by_kind["score"]) + len(by_kind["verbal"])
    assert all(template and template.startswith("{\n    ") for template in engine.templates_seen)
    assert all(r["answer"] for r in by_kind["greedy"]) and not any(r.get("answer") for r in by_kind["sample"])

    # A second invocation finds everything done and adds nothing.
    asyncio.run(run.run(config, show_progress=False))
    assert len((run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()) == len(lines)

    summary = analyze.analyse(run_dir, n_resamples=50)
    assert summary["n_items_complete"] == 60
    assert summary["invalid_generations"] == 0
    by_name = {row["signal"]: row for row in summary["item_level"]}
    assert set(by_name) >= {
        "consistency: mean pairwise Jaccard",
        "log-probability: voted set, teacher-forced",
        "verbalised confidence",
        "control: random",
    }
    assert summary["n_items_all_signals"] == 60  # teacher-forced signal for every item
    assert summary["n_forced_from_greedy"] == 60 - len(disagree)
    # Policies are costed per request: five sampled runs cost five sample requests.
    seconds = analyze._stage_seconds(run_dir)
    policy = {row["policy"]: row for row in summary["policies"]}
    assert policy["5 sampled runs + consistency"]["seconds per item"] == pytest.approx(5 * seconds["sample"], abs=1e-3)
    assert policy["5 sampled runs + teacher-forced log-probability"]["requests per item"] == 6
    assert {row["policy"] for row in summary["policies"]} >= {
        "5 sampled runs + consistency",
        "1 greedy run + its own log-probability",
    }
    assert (run_dir / "items.csv").exists()
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
    assert np.isfinite(summary["task_quality"]["majority vote of 5 sampled runs"]["micro_f1"])


@needs_data
def test_batches_few_shot_and_the_test_lock(tmp_path, monkeypatch):
    from routing import analyze, run

    monkeypatch.setattr(run, "RUNS_DIR", tmp_path)
    config = {
        "run_name": "fake_batches",
        "source": "mipd",
        "role": "pilot",
        "prompt": "p3",
        "n_batches": 3,
        "verbal_greedy": True,
        "engine": {"kind": "fake", "model": "fake"},
        "limit": 40,
    }
    run_dir = asyncio.run(run.run({**config, "score_when_greedy_agrees": True}, show_progress=False))
    summary = analyze.analyse(run_dir, n_resamples=20)
    assert summary["requests"]["score"]["total"] == 40  # every vote scored when asked to
    assert summary["n_forced_from_greedy"] == 0
    assert len(summary["batches"]) == 3
    assert summary["requests"]["sample"]["total"] == 40 * 15
    assert summary["n_items_complete"] == 40
    assert summary["credible_articles"] is None  # the MIPD pilot holds disinformative articles only


    with pytest.raises(PermissionError):
        asyncio.run(run.run({**config, "role": "test"}, show_progress=False))


@needs_data
def test_mipd_roles_follow_the_credibility_label():
    split = splits.make_split("mipd")
    items = data.load_items("mipd").merge(split[["item_id", "role"]], on="item_id")
    by_role = items.groupby("role")["disinformative"].agg(["size", "mean"])
    assert by_role.loc["pilot"].tolist() == [100, 1.0]
    assert by_role.loc["calibration"].tolist() == [600, 1.0]
    assert by_role.loc["credible_check"].tolist() == [200, 0.0]
    assert by_role.loc["test", "size"] == 1521

import pandas as pd
import pytest

from routing import annotate, data

needs_data = pytest.mark.skipif(
    not (data.DATA_DIR / "splits" / "semeval_pl.csv").exists(), reason="prepared datasets are not on this machine"
)


def test_parse_choice():
    assert annotate.parse_choice("", 4) == [0, 0, 0, 0]
    assert annotate.parse_choice("1, 4", 4) == [1, 0, 0, 1]
    assert annotate.parse_choice("2 2", 4) == [0, 1, 0, 0]
    assert annotate.parse_choice("5", 4) is None
    assert annotate.parse_choice("x", 4) is None


@needs_data
def test_two_annotators_are_timed_and_compared(tmp_path, monkeypatch):
    monkeypatch.setattr(annotate, "ANNOTATION_DIR", tmp_path)
    monkeypatch.setitem(annotate.SAMPLE_SIZE, "semeval_pl", 6)
    item_ids = annotate.sample("semeval_pl")
    assert len(item_ids) == 6 and annotate.sample("semeval_pl") == item_ids  # fixed once drawn
    assert set(item_ids) <= set(data.load_role("semeval_pl", "pilot")["item_id"])

    def run(annotator, answers, seconds_per_item):
        answers = iter(answers)
        now = {"t": 0.0}

        def ask(_prompt):
            now["t"] += seconds_per_item
            return next(answers)

        annotate.annotate("semeval_pl", annotator, ask=ask, show=lambda _text: None, clock=lambda: now["t"])

    run("A", ["1", "bad", "", "1 2", "q"], 10.0)  # an invalid answer is asked again; q stops after 3 items
    saved = pd.read_csv(tmp_path / "semeval_pl_A.csv", dtype={"labels": str})
    assert len(saved) == 3 and saved["seconds"].tolist() == [10.0, 20.0, 10.0]
    run("A", ["", "", ""], 10.0)  # resumes with the remaining three
    run("B", ["1", "", "2", "", "", ""], 30.0)

    result = annotate.summary("semeval_pl")
    assert result["annotators"] == ["A", "B"] and result["annotations"] == 12
    assert result["items_with_two_annotators"] == 6
    assert result["median_seconds_by_annotator"] == {"A": 10.0, "B": 30.0}
    assert result["exact_set_agreement"] == pytest.approx(5 / 6)  # they differ on the third item only
    assert "Loaded_Language" not in (tmp_path / "semeval_pl_A.csv").read_text()  # bits only, no text

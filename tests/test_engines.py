import asyncio
import math

import pytest

from routing.engines import (
    FakeEngine,
    TokenAlternative,
    TokenLogprob,
    decision_probabilities,
    decision_schema,
    keys_in_label_order,
    parse_decisions,
)

LABELS = ("A", "B")


def tok(text, alternatives=()):
    """A token with probability-labelled alternatives; the chosen token is the first alternative."""
    alts = [TokenAlternative(t, math.log(p)) for t, p in alternatives]
    return TokenLogprob(text, alts[0].logprob if alts else 0.0, alts)


def test_schema_requires_every_label():
    schema = decision_schema(LABELS)
    assert schema["required"] == ["A", "B"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["A"]["enum"] == [0, 1]


def test_parse_decisions_accepts_only_complete_binary_answers():
    assert parse_decisions('{"A": 1, "B": 0}', LABELS) == [1, 0]
    assert parse_decisions('{"B": 0, "A": 1}', LABELS) == [1, 0]  # key order does not matter
    assert parse_decisions('{"A": 1}', LABELS) is None  # missing label
    assert parse_decisions('{"A": 1, "B": 0, "C": 1}', LABELS) is None  # unknown label
    assert parse_decisions('{"A": 2, "B": 0}', LABELS) is None
    assert parse_decisions('{"A": true, "B": 0}', LABELS) is None
    assert parse_decisions("not json", LABELS) is None
    assert parse_decisions("[1, 0]", LABELS) is None


def test_decision_probabilities_separate_digit_tokens():
    content = [
        tok('{"'), tok("A"), tok('":'),
        tok(" 1", [(" 1", 0.8), (" 0", 0.2)]),
        tok(', "'), tok("B"), tok('":'),
        tok(" 0", [(" 0", 0.9), (" 1", 0.1)]),
        tok("}"),
    ]
    probabilities, bounded = decision_probabilities(content, LABELS)
    assert probabilities == pytest.approx([0.8, 0.1])
    assert bounded == 0


def test_decision_probabilities_renormalise_over_the_two_digits():
    content = [
        tok('{"A":'), tok("1", [("1", 0.6), ("0", 0.2), ("2", 0.2)]),
        tok(',"B":'), tok("0", [("0", 0.5), ("1", 0.25), (" ", 0.25)]),
        tok("}"),
    ]
    probabilities, _ = decision_probabilities(content, LABELS)
    assert probabilities == pytest.approx([0.75, 1 / 3])


def test_decision_probabilities_digit_merged_with_punctuation():
    content = [
        tok('{"A'),
        tok('":1', [('":1', 0.7), ('":0', 0.3)]),
        tok(',"B'),
        tok('":0', [('":0', 0.6), ('":1', 0.4)]),
        tok("}"),
    ]
    probabilities, bounded = decision_probabilities(content, LABELS)
    assert probabilities == pytest.approx([0.7, 0.4])
    assert bounded == 0


def test_decision_probabilities_bound_a_missing_alternative():
    content = [
        tok('{"A":'), tok("1", [("1", 0.97), ("x", 0.02)]),  # "0" is outside the returned top-k
        tok(',"B":'), tok("0", [("0", 0.5), ("1", 0.5)]),
        tok("}"),
    ]
    probabilities, bounded = decision_probabilities(content, LABELS)
    assert bounded == 1
    # "0" can hold at most the 0.01 of mass the alternatives leave over
    assert probabilities[0] == pytest.approx(0.97 / 0.98)
    assert probabilities[1] == pytest.approx(0.5)


def test_decision_probabilities_do_not_depend_on_key_order():
    # Ollama sorts keys alphabetically, so "B" may come before "A".
    content = [
        tok('{"B":'), tok("0", [("0", 0.9), ("1", 0.1)]),
        tok(',"A":'), tok("1", [("1", 0.8), ("0", 0.2)]),
        tok("}"),
    ]
    probabilities, _ = decision_probabilities(content, LABELS)
    assert probabilities == pytest.approx([0.8, 0.1])


def test_keys_in_label_order():
    assert keys_in_label_order('{"A": 1, "B": 0}', LABELS) is True
    assert keys_in_label_order('{"B": 0, "A": 1}', LABELS) is False
    assert keys_in_label_order("not json", LABELS) is None
    assert keys_in_label_order("[1, 0]", LABELS) is None


def test_decision_probabilities_none_when_a_label_is_missing():
    content = [tok('{"A":'), tok("1", [("1", 1.0)]), tok("}")]
    assert decision_probabilities(content, LABELS) is None


def test_fake_engine_is_deterministic_and_seed_sensitive():
    engine = FakeEngine()
    messages = [{"role": "user", "content": "some text"}]
    labels = tuple(f"L{i}" for i in range(12))

    def call(seed, temperature):
        return asyncio.run(
            engine.generate(messages, labels, seed=seed, temperature=temperature, top_p=1.0, want_probabilities=True)
        )

    assert call(1, 0.7).labels == call(1, 0.7).labels
    assert call(1, 0.0).labels == call(2, 0.0).labels  # greedy ignores the seed
    assert any(call(seed, 0.7).labels != call(1, 0.7).labels for seed in range(2, 30))
    greedy = call(1, 0.0)
    assert [int(p >= 0.5) for p in greedy.p1] == greedy.labels

"""Model back ends behind one interface.

OpenAIEngine talks to any OpenAI-compatible chat endpoint (vLLM, llama.cpp server, Ollama), so
the serving engine is a config value. FakeEngine needs no model and exists for tests and for
checking the pipeline end to end without a GPU.

Every generation is a JSON object with one 0/1 decision per label in fixed order. For greedy
passes the engines also return, per label, the probability of the decision "1" normalised over
the two decision tokens.

Two further calls assess a given set of decisions (in the study: the majority vote):
    score      teacher forcing: the answer is appended to the prompt and the model's probability
               of every decision token is read off, without generating anything
    verbalise  a follow-up turn asks the model to state, in percent, how likely the whole
               answer is to be correct
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class TokenAlternative:
    token: str
    logprob: float


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top_logprobs: list[TokenAlternative] = field(default_factory=list)


@dataclass
class Generation:
    labels: list[int] | None  # decisions in label order; None when the output is invalid
    p1: list[float] | None = None  # P(decision = 1) per label, greedy passes only
    n_bounded: int = 0  # labels whose other decision token was outside the returned top-k
    ordered: bool | None = None  # keys came back in label order (None when unknown)
    n_prompt_tokens: int | None = None
    n_output_tokens: int | None = None
    latency_s: float = 0.0
    error: str | None = None
    raw: str | None = None  # kept only for invalid outputs, for debugging
    verbal_overall: float | None = None  # stated probability (0-1) that the whole answer is right


class Engine(Protocol):
    name: str

    async def generate(
        self,
        messages: list[dict[str, str]],
        labels: Sequence[str],
        *,
        seed: int,
        temperature: float,
        top_p: float,
        want_probabilities: bool,
    ) -> Generation: ...

    async def score(
        self, messages: list[dict[str, str]], labels: Sequence[str], decisions: Sequence[int]
    ) -> Generation: ...

    async def verbalise(
        self,
        messages: list[dict[str, str]],
        labels: Sequence[str],
        decisions: Sequence[int],
        followup: str,
        *,
        seed: int,
    ) -> Generation: ...


def decision_schema(labels: Sequence[str]) -> dict:
    """JSON schema: every label is a required key with value 0 or 1, nothing else allowed."""
    return {
        "type": "object",
        "properties": {label: {"type": "integer", "enum": [0, 1]} for label in labels},
        "required": list(labels),
        "additionalProperties": False,
    }


def answer_json(labels: Sequence[str], decisions: Sequence[int]) -> str:
    """A set of decisions written the way the models write their answers."""
    return json.dumps({label: int(value) for label, value in zip(labels, decisions)})


CONFIDENCE_KEY = "confidence"


def verbal_schema() -> dict:
    """JSON schema of the stated-confidence answer: one percentage for the whole answer."""
    return {
        "type": "object",
        "properties": {CONFIDENCE_KEY: {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": [CONFIDENCE_KEY],
        "additionalProperties": False,
    }


def parse_verbal(text: str) -> float | None:
    """The stated probability (0-1) that the whole answer is right, or None if not valid."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {CONFIDENCE_KEY}:
        return None
    value = obj[CONFIDENCE_KEY]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
        return None
    return float(value) / 100.0


def parse_decisions(text: str, labels: Sequence[str]) -> list[int] | None:
    """Return the decisions in label order, or None if the text is not a complete valid answer."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(labels):
        return None
    decisions = []
    for label in labels:
        value = obj[label]
        if isinstance(value, bool) or value not in (0, 1):
            return None
        decisions.append(int(value))
    return decisions


def keys_in_label_order(text: str, labels: Sequence[str]) -> bool | None:
    """Whether the answer lists its keys in the order of `labels`; None if it is not a JSON object.

    Decisions are generated one after another, so the order is part of the experimental setting.
    vLLM and the llama.cpp server follow the schema order; Ollama sorts the keys alphabetically.
    """
    try:
        pairs = json.loads(text, object_pairs_hook=list)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(pairs, list) or any(not isinstance(pair, tuple) for pair in pairs):
        return None
    return [key for key, _ in pairs] == list(labels)


def decision_probabilities(
    content: Sequence[TokenLogprob], labels: Sequence[str], last: bool = False
) -> tuple[list[float], int] | None:
    """P(decision = 1) for each label, read from the token alternatives at each decision.

    The decision digit is located in the generated text, the token covering it is found, and the
    probability mass of the alternatives that would write "1" there is divided by the mass of
    those that would write "0" or "1". If one of the two digits is absent from the returned
    alternatives, its mass is bounded from above by the smallest returned probability (and by the
    mass the alternatives leave over); the second return value counts how often that happened.

    With `last`, the last occurrence of each key is used. That is needed when the tokens cover a
    whole prompt in which earlier turns (few-shot examples) contain answers of the same form.
    """
    text = "".join(token.token for token in content)
    starts, position = [], 0
    for token in content:
        starts.append(position)
        position += len(token.token)

    probabilities, bounded = [], 0
    for label in labels:
        # Keys are searched independently: some servers emit them in their own order.
        matches = list(re.finditer(r'"%s"\s*:\s*([01])' % re.escape(label), text))
        if not matches:
            return None
        match = matches[-1] if last else matches[0]
        digit_at = match.start(1)
        index = max(i for i, start in enumerate(starts) if start <= digit_at)
        token = content[index]
        offset = digit_at - starts[index]
        prefix = token.token[:offset]

        alternatives = {alt.token: alt.logprob for alt in token.top_logprobs}
        alternatives.setdefault(token.token, token.logprob)
        mass = {"0": 0.0, "1": 0.0}
        for alt_text, logprob in alternatives.items():
            if prefix.strip() == "":  # the digit is the first visible character of the token
                digit = alt_text.lstrip()[:1]
            else:  # the digit is merged with preceding characters, e.g. '":1'
                digit = alt_text[offset : offset + 1] if alt_text.startswith(prefix) else ""
            if digit in mass:
                mass[digit] += math.exp(logprob)

        if mass["0"] == 0.0 or mass["1"] == 0.0:
            bounded += 1
            listed = [math.exp(lp) for lp in alternatives.values()]
            ceiling = min(min(listed), max(1.0 - sum(listed), 1e-9))
            mass = {digit: (value if value > 0.0 else ceiling) for digit, value in mass.items()}
        probabilities.append(mass["1"] / (mass["0"] + mass["1"]))
    return probabilities, bounded


class OpenAIEngine:
    """Chat completions against an OpenAI-compatible server with a JSON-schema constrained answer."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout_s: float = 600.0,
        top_logprobs: int = 10,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        prompt_logprobs: int = 20,
    ) -> None:
        from openai import AsyncOpenAI  # imported here so tests do not need the package

        self.name = model
        self.model = model
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens
        # Server-specific request fields. Models that reason before answering must have that
        # switched off here (Ollama: {"reasoning_effort": "none"}), or the hidden reasoning
        # consumes the token budget and produces free text the study must not generate.
        self.extra_body = extra_body or {}
        # Number of alternatives requested per prompt position when scoring (vLLM only).
        self.prompt_logprobs = prompt_logprobs
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=2)

    async def generate(self, messages, labels, *, seed, temperature, top_p, want_probabilities) -> Generation:
        request = {
            "model": self.model,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.max_tokens or 40 * len(labels) + 64,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decisions", "schema": decision_schema(labels), "strict": True},
            },
        }
        if want_probabilities:
            request["logprobs"] = True
            request["top_logprobs"] = self.top_logprobs
        if self.extra_body:
            request["extra_body"] = self.extra_body

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as error:  # network, server and timeout errors all become a failed generation
            return Generation(labels=None, latency_s=time.perf_counter() - started, error=f"request: {error}"[:300])
        latency = time.perf_counter() - started

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        result = Generation(
            labels=parse_decisions(text, labels),
            ordered=keys_in_label_order(text, labels),
            n_prompt_tokens=getattr(usage, "prompt_tokens", None),
            n_output_tokens=getattr(usage, "completion_tokens", None),
            latency_s=latency,
        )
        if result.labels is None:
            result.error = f"invalid_output (finish_reason={choice.finish_reason})"
            result.raw = text[:500]
            return result

        if want_probabilities:
            logprobs = getattr(choice, "logprobs", None)
            content = getattr(logprobs, "content", None) if logprobs is not None else None
            if not content:
                result.error = "no_logprobs_returned"
                return result
            tokens = [
                TokenLogprob(
                    token=item.token,
                    logprob=item.logprob,
                    top_logprobs=[TokenAlternative(alt.token, alt.logprob) for alt in (item.top_logprobs or [])],
                )
                for item in content
            ]
            extracted = decision_probabilities(tokens, labels)
            if extracted is None:
                result.error = "decision_tokens_not_found"
            else:
                result.p1, result.n_bounded = extracted
        return result


    async def score(self, messages, labels, decisions) -> Generation:
        """Teacher forcing: P(decision = 1) at every decision position of the given answer.

        The answer is sent as the unfinished final assistant turn and the server returns the
        log-probabilities of the prompt tokens (`prompt_logprobs`, a vLLM extension). Nothing is
        generated apart from the one token the API requires. Servers without this extension
        return an error, which is logged; the signal is then reported as unavailable.
        """
        answer = answer_json(labels, decisions)
        request = {
            "model": self.model,
            "messages": [*messages, {"role": "assistant", "content": answer}],
            "temperature": 0.0,
            "max_tokens": 1,
            "extra_body": {
                **self.extra_body,
                "add_generation_prompt": False,
                "continue_final_message": True,
                "prompt_logprobs": self.prompt_logprobs,
            },
        }
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as error:
            return Generation(labels=None, latency_s=time.perf_counter() - started, error=f"request: {error}"[:300])
        latency = time.perf_counter() - started
        usage = response.usage
        result = Generation(
            labels=[int(value) for value in decisions],
            n_prompt_tokens=getattr(usage, "prompt_tokens", None),
            n_output_tokens=0,
            latency_s=latency,
        )
        positions = (response.model_extra or {}).get("prompt_logprobs")
        if positions is None:  # the completions API puts it on the choice
            positions = (response.choices[0].model_extra or {}).get("prompt_logprobs")
        if not positions:
            result.error = "no_prompt_logprobs_returned"
            return result
        # Only the end of the prompt holds the answer; a generous window keeps the search cheap.
        window = positions[-(40 * len(labels) + 64) :]
        tokens = []
        for alternatives in window:
            if not alternatives:
                continue
            # The first entry is the token that is actually in the prompt, the rest are the most
            # probable tokens at that position.
            entries = [
                TokenAlternative(str(entry.get("decoded_token") or ""), float(entry["logprob"]))
                for entry in alternatives.values()
            ]
            tokens.append(TokenLogprob(entries[0].token, entries[0].logprob, entries))
        extracted = decision_probabilities(tokens, labels, last=True)
        if extracted is None:
            result.error = "decision_tokens_not_found"
        else:
            result.p1, result.n_bounded = extracted
        return result

    async def verbalise(self, messages, labels, decisions, followup, *, seed) -> Generation:
        """Ask in a follow-up turn how likely the given answer is to be correct as a whole.

        Only one value is asked for. A per-technique variant was tried on 12 pilot paragraphs
        (15 Nov 2026): Bielik gave one constant value for all 23 techniques in six of them and
        otherwise switched between "probability that the technique is present" and "probability
        that the decision is right", under both wordings of the question. Such values cannot be
        interpreted, so stated confidence is available at item level only.
        """
        request = {
            "model": self.model,
            "messages": [
                *messages,
                {"role": "assistant", "content": answer_json(labels, decisions)},
                {"role": "user", "content": followup},
            ],
            "seed": seed,
            "temperature": 0.0,
            "max_tokens": 32,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "confidence", "schema": verbal_schema(), "strict": True},
            },
        }
        if self.extra_body:
            request["extra_body"] = self.extra_body
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**request)
        except Exception as error:
            return Generation(labels=None, latency_s=time.perf_counter() - started, error=f"request: {error}"[:300])
        latency = time.perf_counter() - started
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        result = Generation(
            labels=[int(value) for value in decisions],
            n_prompt_tokens=getattr(usage, "prompt_tokens", None),
            n_output_tokens=getattr(usage, "completion_tokens", None),
            latency_s=latency,
        )
        result.verbal_overall = parse_verbal(text)
        if result.verbal_overall is None:
            result.labels = None
            result.error = f"invalid_output (finish_reason={choice.finish_reason})"
            result.raw = text[:500]
        return result


class FakeEngine:
    """Deterministic stand-in for a model.

    Each prompt gets a latent probability per label derived from a hash of the prompt. A greedy
    pass rounds it, a sampled pass draws from it with the request seed. Outputs therefore look
    like a real run (items differ in how much their samples agree) but carry no information
    about the gold labels.
    """

    def __init__(self, name: str = "fake", invalid_rate: float = 0.0) -> None:
        self.name = name
        self.invalid_rate = invalid_rate

    @staticmethod
    def _latent(messages, n_labels: int) -> np.ndarray:
        digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        # Most labels are clearly absent, a few are uncertain or clearly present.
        return rng.beta(0.3, 1.5, size=n_labels)

    async def generate(self, messages, labels, *, seed, temperature, top_p, want_probabilities) -> Generation:
        latent = self._latent(messages, len(labels))
        rng = np.random.default_rng([seed, int(latent[0] * 1e9)])
        if rng.random() < self.invalid_rate:
            return Generation(labels=None, error="invalid_output (fake)", raw="not json")
        if temperature == 0:
            decisions = (latent >= 0.5).astype(int)
        else:
            decisions = (rng.random(len(labels)) < latent).astype(int)
        text = json.dumps({label: int(value) for label, value in zip(labels, decisions)})
        result = Generation(
            labels=parse_decisions(text, labels),
            ordered=True,
            n_prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
            n_output_tokens=len(text) // 4,
            latency_s=0.0,
        )
        if want_probabilities:
            result.p1 = [float(min(max(p, 1e-6), 1 - 1e-6)) for p in latent]
        return result

    async def score(self, messages, labels, decisions) -> Generation:
        latent = self._latent(messages, len(labels))
        return Generation(
            labels=[int(value) for value in decisions],
            p1=[float(min(max(p, 1e-6), 1 - 1e-6)) for p in latent],
            ordered=True,
            n_prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
            n_output_tokens=0,
        )

    async def verbalise(self, messages, labels, decisions, followup, *, seed) -> Generation:
        latent = self._latent(messages, len(labels))
        rng = np.random.default_rng([seed, int(latent[0] * 1e9), 7])
        if rng.random() < self.invalid_rate:
            return Generation(labels=None, error="invalid_output (fake)", raw="not json")
        confidence = np.where(np.asarray(decisions) == 1, latent, 1.0 - latent)
        return Generation(
            labels=[int(value) for value in decisions],
            # stated values come in steps of five percent
            verbal_overall=float(np.round(float(np.prod(confidence)) * 20) / 20),
            ordered=True,
            n_prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
            n_output_tokens=8,
        )


def build_engine(config: dict) -> Engine:
    kind = config.get("kind", "openai")
    if kind == "fake":
        return FakeEngine(name=config.get("model", "fake"), invalid_rate=config.get("invalid_rate", 0.0))
    if kind == "openai":
        return OpenAIEngine(
            base_url=config["base_url"],
            model=config["model"],
            api_key=config.get("api_key", "not-needed"),
            timeout_s=config.get("timeout_s", 600.0),
            top_logprobs=config.get("top_logprobs", 10),
            max_tokens=config.get("max_tokens"),
            extra_body=config.get("extra_body"),
            prompt_logprobs=config.get("prompt_logprobs", 20),
        )
    raise KeyError(f"unknown engine kind {kind!r}")

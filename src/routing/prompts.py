"""Build the chat messages for one item.

Prompt variants (PLAN.md, decision 7):
    p1  zero-shot, instructions in the language of the text
    p2  zero-shot, instructions in the other language
    p3  few-shot, instructions in the language of the text

A template in prompts/ has two parts separated by its last blank line: the instructions with the
technique definitions, and the block that holds the text. In the zero-shot prompts both parts form
one user message with the text last, so all items of a source share the same prefix and the
server can reuse it from its prefix cache. In the few-shot prompt each example is a user turn
(the text block) followed by an assistant turn (the gold answer in the output format); the
examples are fixed per source, so the shared prefix covers them as well.

Few-shot examples are stored as item ids in prompts/fewshot_<source>.yaml (written by
`python -m routing.fewshot`) and their text is filled in at run time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
import yaml

from .data import REPO_ROOT
from .engines import answer_json
from .labels import SOURCES, labels_for

PROMPT_DIR = REPO_ROOT / "prompts"
PROMPT_IDS = ("p1", "p2", "p3")
_OTHER = {"pl": "en", "en": "pl"}


@lru_cache(maxsize=None)
def _definitions(task: str) -> dict:
    with open(PROMPT_DIR / f"labels_{task}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=None)
def _template(task: str, language: str) -> tuple[Template, Template]:
    """The instruction part and the text block of a template."""
    raw = Path(PROMPT_DIR / f"{task}.{language}.txt").read_text(encoding="utf-8").strip()
    instructions, _, block = raw.rpartition("\n\n")
    if "$techniques" not in instructions or "$text" not in block:
        raise ValueError(f"{task}.{language}.txt: expected instructions, a blank line, then the text block")
    return Template(instructions), Template(block)


def instruction_language(source: str, prompt_id: str) -> str:
    text_language = SOURCES[source]["language"]
    if prompt_id in ("p1", "p3"):
        return text_language
    if prompt_id == "p2":
        return _OTHER[text_language]
    raise KeyError(f"unknown prompt {prompt_id!r}; expected one of {PROMPT_IDS}")


def technique_list(source: str, language: str) -> str:
    """One line per technique: key, name and definition in the instruction language."""
    task = SOURCES[source]["task"]
    definitions = _definitions(task)
    lines = []
    for label in labels_for(source):
        entry = definitions[label]
        lines.append(f"- {label} ({entry[f'name_{language}']}): {entry[f'def_{language}'].strip()}")
    return "\n".join(lines)


@lru_cache(maxsize=None)
def fewshot_examples(source: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """(text, gold decisions) of the fixed few-shot examples of a source, in prompt order."""
    from .data import load_items  # local import: data loading is not needed for zero-shot prompts

    path = PROMPT_DIR / f"fewshot_{source}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m routing.fewshot` first")
    with open(path, encoding="utf-8") as handle:
        item_ids = yaml.safe_load(handle)["item_ids"]
    items = load_items(source).set_index("item_id")
    missing = [item_id for item_id in item_ids if item_id not in items.index]
    if missing:
        raise KeyError(f"{path.name}: unknown item ids {missing}")
    labels = list(labels_for(source))
    return tuple((items.at[i, "text"], tuple(int(v) for v in items.loc[i, labels])) for i in item_ids)


def build_messages(source: str, prompt_id: str, text: str) -> list[dict[str, str]]:
    language = instruction_language(source, prompt_id)
    instructions, block = _template(SOURCES[source]["task"], language)
    head = instructions.substitute(techniques=technique_list(source, language))
    if prompt_id != "p3":
        return [{"role": "user", "content": head + "\n\n" + block.substitute(text=text.strip())}]

    labels = labels_for(source)
    messages: list[dict[str, str]] = []
    for position, (example_text, decisions) in enumerate(fewshot_examples(source)):
        content = block.substitute(text=example_text.strip())
        messages.append({"role": "user", "content": head + "\n\n" + content if position == 0 else content})
        messages.append({"role": "assistant", "content": answer_json(labels, decisions)})
    messages.append({"role": "user", "content": block.substitute(text=text.strip())})
    return messages


def verbal_followup(source: str, prompt_id: str) -> str:
    """The follow-up turn that asks for stated confidence, in the instruction language."""
    language = instruction_language(source, prompt_id)
    return Path(PROMPT_DIR / f"verbal.{language}.txt").read_text(encoding="utf-8").strip()

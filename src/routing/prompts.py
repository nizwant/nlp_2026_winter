"""Build the chat messages for one item.

Prompt variants (PLAN.md, decision 7):
    p1  zero-shot, instructions in the language of the text
    p2  zero-shot, instructions in the other language
    p3  few-shot in the language of the text (not implemented yet)

The whole prompt is one user message with the text last, so every item of a source shares the
same prefix (instructions and definitions) and servers can reuse it from their prefix cache.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

import yaml

from .data import REPO_ROOT
from .labels import SOURCES, labels_for

PROMPT_DIR = REPO_ROOT / "prompts"
PROMPT_IDS = ("p1", "p2", "p3")
_OTHER = {"pl": "en", "en": "pl"}


@lru_cache(maxsize=None)
def _definitions(task: str) -> dict:
    with open(PROMPT_DIR / f"labels_{task}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=None)
def _template(task: str, language: str) -> Template:
    return Template(Path(PROMPT_DIR / f"{task}.{language}.txt").read_text(encoding="utf-8"))


def instruction_language(source: str, prompt_id: str) -> str:
    text_language = SOURCES[source]["language"]
    if prompt_id == "p1":
        return text_language
    if prompt_id == "p2":
        return _OTHER[text_language]
    if prompt_id == "p3":
        raise NotImplementedError("the few-shot prompt p3 is not written yet")
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


def build_messages(source: str, prompt_id: str, text: str) -> list[dict[str, str]]:
    language = instruction_language(source, prompt_id)
    template = _template(SOURCES[source]["task"], language)
    content = template.substitute(techniques=technique_list(source, language), text=text.strip())
    return [{"role": "user", "content": content}]

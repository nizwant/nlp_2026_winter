"""Pinned model specs from configs/models.yaml, and how to start vLLM with them.

Every script that loads a model goes through here, so the offline engine (`make_llm`) and the
server (`serve_args`) get the same revision, engine arguments and chat-template settings.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str | None
    served_name: str
    study: bool = True  # False: smoke tests and speed comparisons only
    engine_args: dict[str, Any] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


def load_models() -> dict[str, ModelSpec]:
    raw = yaml.safe_load(MODELS_FILE.read_text())
    return {
        key: ModelSpec(key=key, served_name=entry.pop("served_name", key), **entry)
        for key, entry in raw.items()
    }


def get_model(key: str) -> ModelSpec:
    """A model by its key in models.yaml. A raw `org/name` repo id is accepted, unpinned."""
    models = load_models()
    if key in models:
        return models[key]
    if "/" in key:
        return ModelSpec(key=key, repo=key, revision=None, served_name=key, study=False)
    raise KeyError(f"unknown model {key!r}; known: {', '.join(models)}")


def machine_limits() -> dict[str, int]:
    """Optional per-machine caps from .env (MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS); unset = vLLM default."""
    env = {"max_num_seqs": "MAX_NUM_SEQS", "max_num_batched_tokens": "MAX_NUM_BATCHED_TOKENS"}
    return {arg: int(os.environ[var]) for arg, var in env.items() if os.environ.get(var)}


def llm_kwargs(spec: ModelSpec, **overrides: Any) -> dict[str, Any]:
    """Arguments for vllm.LLM. Memory settings default to the per-machine values in .env."""
    return {
        "model": spec.repo,
        "revision": spec.revision,
        "tokenizer_revision": spec.revision,
        "seed": 0,
        "gpu_memory_utilization": float(os.environ.get("GPU_MEM_UTIL", "0.82")),
        "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "8192")),
        **machine_limits(),
        **spec.engine_args,
        **overrides,
    }


def make_llm(spec: ModelSpec, **overrides: Any):
    # vLLM forks its engine process by default, which fails if this process has already touched
    # CUDA (torch.cuda.* in a script or notebook cell). Spawn always works; scripts that call
    # this need an `if __name__ == "__main__":` guard.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    return LLM(**llm_kwargs(spec, **overrides))


def serve_args(spec: ModelSpec) -> list[str]:
    """`vllm serve` arguments for the model-specific part (memory, host, port are added by the caller)."""
    args = [spec.repo, "--served-model-name", spec.served_name]
    if spec.revision:
        args += ["--revision", spec.revision, "--tokenizer-revision", spec.revision]
    for name, value in spec.engine_args.items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is False:
            args.append("--no-" + name.replace("_", "-"))
        else:
            args += [flag, json.dumps(value) if isinstance(value, dict | list) else str(value)]
    if spec.chat_template_kwargs:
        args += ["--default-chat-template-kwargs", json.dumps(spec.chat_template_kwargs)]
    return args

"""Measure generation speed of the models in configs/models.yaml on this GPU.

    docker compose run --rm dev python scripts/bench.py                    # every model
    docker compose run --rm dev python scripts/bench.py --models gemma4_e2b bielik

The workload is shaped like the study: SemEval PL train paragraphs (a fixed sample) inside a
Polish instruction prompt, sent through each model's chat template. Two measurements:

  single  one request, greedy: the decode speed a lone request gets (memory-bandwidth bound)
  batch   every paragraph x 5 samples at temperature 0.7: throughput with vLLM's batching,
          the regime of the actual reruns

Every completion is forced to exactly --max-tokens new tokens (ignore_eos), so all models do the
same work per request. Tokens are not words, and the vocabularies differ, so the table also
gives prompt tokens per request and paragraphs per second (each with 5 samples).

Each model runs in its own process, because vLLM does not reliably free GPU memory within one.
Results go to outputs/bench/<gpu>.json; no completions are stored.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = (
    "Przeanalizuj poniższy akapit z artykułu prasowego. Opisz, jakie techniki perswazji "
    "lub manipulacji w nim występują, i uzasadnij swoją odpowiedź.\n\nAkapit:\n{text}"
)
N_SAMPLES = 5


def gpu_info() -> dict:
    import pynvml  # no CUDA context in the parent process, unlike torch

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    return {
        "name": name.decode() if isinstance(name, bytes) else name,
        "memory_total_gib": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30, 2),
        "memory_used_before_gib": round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30, 2),
    }


def paragraphs(n: int) -> list[str]:
    import pandas as pd

    df = pd.read_csv(ROOT / "data" / "sem_eval" / "pl.csv")
    df = df[(df.original_set == "train") & (df.text.str.split().str.len() >= 5)]
    return df.sample(n=n, random_state=0).text.tolist()


def worker(key: str, args: argparse.Namespace) -> dict:
    import pynvml
    from vllm import SamplingParams

    from routing.models import get_model, machine_limits, make_llm

    spec = get_model(key)
    chats = [[{"role": "user", "content": INSTRUCTION.format(text=t)}]
             for t in paragraphs(args.paragraphs)]

    explicit = {k: getattr(args, k) for k in ("max_num_seqs", "max_num_batched_tokens")
                if getattr(args, k) is not None}
    limits = {**machine_limits(), **explicit}  # recorded as used, whether from .env or flags
    t0 = time.perf_counter()
    llm = make_llm(spec, gpu_memory_utilization=args.gpu_mem_util,
                   max_model_len=args.max_model_len, **limits)
    load_s = time.perf_counter() - t0

    def chat(batch, params):
        return llm.chat(batch, params, use_tqdm=False,
                        chat_template_kwargs=spec.chat_template_kwargs or None)

    chat(chats[:8], SamplingParams(temperature=0.0, max_tokens=16))  # warm-up, not timed

    single = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    out = chat(chats[:1], single)
    single_s = time.perf_counter() - t0
    single_tokens = len(out[0].outputs[0].token_ids)

    batch = SamplingParams(n=N_SAMPLES, temperature=0.7, top_p=0.95,
                           max_tokens=args.max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = chat(chats, batch)
    batch_s = time.perf_counter() - t0
    out_tokens = sum(len(c.token_ids) for o in outs for c in o.outputs)
    prompt_tokens = [len(o.prompt_token_ids) for o in outs]

    pynvml.nvmlInit()
    used = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used
    return {
        "model": spec.key,
        "repo": spec.repo,
        "revision": spec.revision,
        "vocab_size": len(llm.get_tokenizer()),
        "gpu_mem_util": args.gpu_mem_util,
        **limits,
        "load_seconds": round(load_s, 1),
        "single_tokens_per_second": round(single_tokens / single_s, 1),
        "batch_output_tokens_per_second": round(out_tokens / batch_s, 1),
        "batch_total_tokens_per_second": round((out_tokens + sum(prompt_tokens)) / batch_s, 1),
        "batch_paragraphs_per_second": round(len(chats) / batch_s, 2),
        "batch_seconds": round(batch_s, 1),
        "prompt_tokens_mean": round(sum(prompt_tokens) / len(prompt_tokens), 1),
        "gpu_memory_in_use_gib": round(used / 2**30, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", metavar="KEY", help="keys in configs/models.yaml (default: all)")
    ap.add_argument("--paragraphs", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=256, help="new tokens per completion")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.82")))
    ap.add_argument("--max-num-seqs", type=int, help="cap on parallel sequences (vLLM default if unset)")
    ap.add_argument("--max-num-batched-tokens", type=int, help="cap on tokens per step (prefill chunk)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        args.result.write_text(json.dumps(worker(args.worker, args)))
        return 0

    from routing.models import load_models

    known = load_models()
    keys = args.models or list(known)
    unknown = [k for k in keys if k not in known and "/" not in k]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; known: {', '.join(known)}")
    gpu = gpu_info()
    workload = {k: getattr(args, k) for k in ("paragraphs", "max_tokens", "max_model_len")}
    engine = {k: getattr(args, k) for k in ("gpu_mem_util", "max_num_seqs", "max_num_batched_tokens")
              if getattr(args, k) is not None}
    settings = {**workload, **engine}
    results = []
    for key in keys:
        print(f"\n=== {key} ===", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.json"
            cmd = [sys.executable, __file__, "--worker", key, "--result", str(result)]
            cmd += [f"--{k.replace('_', '-')}={v}" for k, v in settings.items()]
            rc = subprocess.run(cmd, check=False).returncode
            results.append(json.loads(result.read_text()) if rc == 0 and result.exists()
                           else {"model": key, "error": f"exit code {rc}, see the log above"})

    batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    workload = {**workload, "samples_per_paragraph": N_SAMPLES, "batch_invariant": batch_invariant}
    slug = re.sub(r"[^a-z0-9]+", "-", gpu["name"].lower()).strip("-")
    suffix = "_batch-invariant" if batch_invariant else ""
    out = ROOT / "outputs" / "bench" / f"{slug}{suffix}.json"
    # Re-running some models replaces their rows and keeps the others, if the workload matches.
    merged = {}
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("workload") == workload:
            merged = {r["model"]: r for r in old["results"]}
    merged.update({r["model"]: r for r in results})
    report = {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": gpu,
        "workload": workload,
        "results": list(merged.values()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{gpu['name']}, {args.paragraphs} paragraphs x {N_SAMPLES} samples, "
          f"{args.max_tokens} new tokens each, batch-invariant {'on' if batch_invariant else 'off'}\n")
    print("| model | load s | single tok/s | batch out tok/s | paragraphs/s | prompt tok | GPU GiB | engine limits |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in report["results"]:
        if "error" in r:
            print(f"| {r['model']} | {r['error']} |||||||")
            continue
        limits = ", ".join(f"{k}={r[k]}" for k in ("gpu_mem_util", "max_num_seqs", "max_num_batched_tokens") if k in r)
        print(f"| {r['model']} | {r['load_seconds']:.0f} | {r['single_tokens_per_second']:.0f} | "
              f"{r['batch_output_tokens_per_second']:.0f} | {r['batch_paragraphs_per_second']:.2f} | "
              f"{r['prompt_tokens_mean']:.0f} | {r['gpu_memory_in_use_gib']:.1f} | {limits} |")
    print(f"\nSaved {out.relative_to(ROOT)}")
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

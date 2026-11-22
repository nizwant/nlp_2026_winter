"""Check that this machine runs the project's stack, and record what it ran.

    docker compose run --rm dev python scripts/check_env.py               # full check, Bielik
    docker compose run --rm dev python scripts/check_env.py --model pllum
    docker compose run --rm dev python scripts/check_env.py --no-model    # no model load

Checks: installed packages match uv.lock; the NVIDIA driver is new enough for CUDA 13; torch has
kernels for this GPU; vLLM loads a pinned model, generates, returns log-probabilities, and gives
identical samples when a request is repeated with the same seed.

Writes outputs/env_check/<gpu>.json. Commit it, and compare the files from the two machines:
`packages` must be identical; `greedy_fingerprint` shows whether the two GPUs produce the same
greedy tokens for the same prompts (they may not, because the kernels differ per architecture).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
import time
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
MIN_DRIVER = 580  # torch 2.14 cu130 wheels ship CUDA 13.0 user-space libraries
KEY_PACKAGES = ["vllm", "torch", "transformers", "flashinfer-python", "triton", "compressed-tensors"]
PROMPTS = [
    "Odpowiedz jednym zdaniem: czym zajmuje się lingwistyka komputerowa?",
    "Wymień trzy największe miasta w Polsce, oddzielone przecinkami.",
]

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def lock_mismatches() -> list[str]:
    """Packages whose installed version differs from uv.lock (ones not installed are skipped)."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    bad = []
    for pkg in lock["package"]:
        source = pkg.get("source", {})
        if "editable" in source or "virtual" in source:
            continue
        try:
            installed = metadata.version(pkg["name"])
        except metadata.PackageNotFoundError:
            continue
        if Version(installed) != Version(pkg["version"]):
            bad.append(f"{pkg['name']} {installed} (lock: {pkg['version']})")
    return bad


def gpu_report() -> dict:
    import pynvml
    import torch

    pynvml.nvmlInit()
    driver = pynvml.nvmlSystemGetDriverVersion()
    driver = driver.decode() if isinstance(driver, bytes) else driver
    check(int(driver.split(".")[0]) >= MIN_DRIVER, "NVIDIA driver", f"{driver} (need >= {MIN_DRIVER})")

    if not check(torch.cuda.is_available(), "torch sees a CUDA GPU"):
        return {"driver": driver}
    major, minor = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    # A cubin built for sm_XY runs on any GPU of the same major version with minor >= Y.
    nums = [int(re.sub(r"\D", "", a)) for a in archs if a.startswith("sm_")]
    supported = any(n // 10 == major and n % 10 <= minor for n in nums)
    name = torch.cuda.get_device_name(0)
    check(supported, "torch has kernels for this GPU", f"{name}, sm_{major}{minor}; built for {archs}")

    x = torch.randn(1024, 1024, device="cuda")
    check(bool(torch.isfinite(x @ x).all()), "CUDA matmul runs")

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"  GPU memory: {mem.used / 2**30:.1f} of {mem.total / 2**30:.1f} GiB already in use")
    return {
        "name": name,
        "compute_capability": f"{major}.{minor}",
        "memory_total_gib": round(mem.total / 2**30, 2),
        "memory_used_before_gib": round(mem.used / 2**30, 2),
        "driver": driver,
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": archs,
    }


def model_report(key: str, gpu_mem_util: float, max_model_len: int) -> dict:
    import pynvml
    from vllm import SamplingParams

    from routing.models import get_model, make_llm

    spec = get_model(key)
    print(f"\nLoading {spec.repo} @ {spec.revision} "
          f"(gpu_memory_utilization={gpu_mem_util}, max_model_len={max_model_len})")
    t0 = time.perf_counter()
    llm = make_llm(spec, gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)
    load_s = time.perf_counter() - t0

    chats = [[{"role": "user", "content": p}] for p in PROMPTS]
    greedy = SamplingParams(temperature=0.0, max_tokens=48, logprobs=1)
    sampled = SamplingParams(n=5, temperature=0.7, top_p=0.95, max_tokens=48, seed=1234)

    def chat(params):
        return llm.chat(chats, params, use_tqdm=False,
                        chat_template_kwargs=spec.chat_template_kwargs or None)

    t0 = time.perf_counter()
    g = chat(greedy)
    s1 = chat(sampled)
    gen_s = time.perf_counter() - t0
    s2 = chat(sampled)

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**30

    greedy_ids = [list(o.outputs[0].token_ids) for o in g]
    runs1 = [[list(c.token_ids) for c in o.outputs] for o in s1]
    runs2 = [[list(c.token_ids) for c in o.outputs] for o in s2]
    n_tokens = sum(len(t) for t in greedy_ids) + sum(len(t) for r in runs1 for t in r)

    print()
    check(all(o.outputs[0].logprobs for o in g), "vLLM returns token log-probabilities")
    check(runs1 == runs2, "same seed gives the same 5 samples on a repeat")
    distinct = [len({tuple(t) for t in r}) for r in runs1]
    check(any(d > 1 for d in distinct), "samples at temperature 0.7 differ", f"distinct per prompt: {distinct}")
    for prompt, o in zip(PROMPTS, g):
        print(f"  > {prompt}\n    {o.outputs[0].text.strip()[:200]}")
    print(f"  load {load_s:.0f} s, generation {gen_s:.1f} s ({n_tokens / gen_s:.0f} tok/s), "
          f"GPU memory in use {used:.1f} GiB")

    return {
        "model": {"key": spec.key, "repo": spec.repo, "revision": spec.revision},
        "gpu_memory_utilization": gpu_mem_util,
        "max_model_len": max_model_len,
        "load_seconds": round(load_s, 1),
        "generation_tokens_per_second": round(n_tokens / gen_s, 1),
        "gpu_memory_in_use_gib": round(used, 2),
        "greedy_fingerprint": hashlib.sha256(json.dumps(greedy_ids).encode()).hexdigest()[:16],
        "sampled_fingerprint": hashlib.sha256(json.dumps(runs1).encode()).hexdigest()[:16],
        "distinct_samples_per_prompt": distinct,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="bielik", help="key in configs/models.yaml or a HF repo id")
    ap.add_argument("--no-model", action="store_true", help="skip loading a model")
    ap.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("GPU_MEM_UTIL", 0.85)))
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/env_check/<gpu>.json")
    args = ap.parse_args()

    print("Environment")
    versions = {p: metadata.version(p) for p in KEY_PACKAGES}
    print(f"  Python {platform.python_version()}, " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    mismatched = lock_mismatches()
    check(not mismatched, "installed packages match uv.lock", "; ".join(mismatched[:5]))
    check(os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "0", "VLLM_USE_FLASHINFER_SAMPLER=0")

    print("\nGPU")
    gpu = gpu_report()

    report = {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "in_docker": Path("/.dockerenv").exists(),
        "packages": {d.metadata["Name"].lower(): d.version for d in metadata.distributions()},
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "gpu": gpu,
    }
    report["packages"] = dict(sorted(report["packages"].items()))
    if not args.no_model and not failures:
        report["smoke_test"] = model_report(args.model, args.gpu_mem_util, args.max_model_len)
    report["failures"] = failures

    slug = re.sub(r"[^a-z0-9]+", "-", gpu.get("name", "no-gpu").lower()).strip("-")
    name = slug if args.model == "bielik" or args.no_model else f"{slug}_{args.model.replace('/', '-')}"
    out = args.out or ROOT / "outputs" / "env_check" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{'All checks passed' if not failures else 'FAILED: ' + ', '.join(failures)}. Report: {out.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

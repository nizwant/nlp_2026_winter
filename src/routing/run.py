"""Run one experiment: repeated sampled generations, one greedy pass, and two confidence queries.

    python -m routing.run configs/bielik.yaml                              # SemEval PL pilot, prompt P1
    python -m routing.run configs/bielik.yaml --source mipd --prompt p3    # other data or prompt
    python -m routing.run configs/bielik.yaml --limit 20                   # quick check
    python -m routing.run configs/bielik.yaml --fake                       # no model needed

A config names the model and the sampling settings; source, role and prompt can be overridden on
the command line. Unless the config gives a run_name, the run is called
<model_key>_<source>_<role>_<prompt>.

Per item the run produces, in this order (each kind is one stage, timed separately):

    sample         n_samples generations at the configured temperature, one seed each;
                   n_batches independent batches of them (batch 0 is the main one)
    greedy         one generation at temperature 0 with the probability of every decision
    score          the majority vote of batch 0 is scored by teacher forcing (no generation); by
                   default only where it differs from the greedy answer, because where the two are
                   equal the greedy pass has already computed the same probabilities
    verbal         a follow-up turn asks how likely the majority vote is to be correct as a whole
    verbal_greedy  the same question about the greedy answer, where it differs from the vote

Results go to outputs/runs/<run_name>/: the resolved config, run metadata and one JSON line per
generation. The log holds item ids and decisions only, never the text. Rerunning the same command
continues an interrupted run: finished generations are skipped, failed requests are retried.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .data import REPO_ROOT, load_role
from .engines import Engine, Generation, build_engine
from .labels import labels_for
from .prompts import build_messages, verbal_followup
from .signals import majority_vote

RUNS_DIR = REPO_ROOT / "outputs" / "runs"
STAGES = ("sample", "greedy", "score", "verbal", "verbal_greedy")

DEFAULTS = {
    "prompt": "p1",
    "n_samples": 5,
    "n_batches": 1,
    "temperature": 0.7,
    "top_p": 0.95,
    "base_seed": 1000,
    "greedy": True,
    "score_voted": True,
    "score_when_greedy_agrees": False,
    "verbal": True,
    "verbal_greedy": False,
    "max_concurrency": 8,
    "limit": None,
    "allow_test": False,
}


def load_config(path: Path, **overrides) -> dict:
    """Read a config file; `overrides` (source, role, prompt, ...) replace its values when not None."""
    with open(path, encoding="utf-8") as handle:
        config = {**DEFAULTS, **yaml.safe_load(handle)}
    config.update({key: value for key, value in overrides.items() if value is not None})
    for key in ("source", "role", "engine"):
        if key not in config:
            raise KeyError(f"{path}: missing required key {key!r}")
    if "run_name" not in config:
        if "model_key" not in config:
            raise KeyError(f"{path}: give run_name or model_key")
        config["run_name"] = "_".join([config["model_key"], config["source"], config["role"], config["prompt"]])
    return config


def sample_seed(config: dict, batch: int, run: int) -> int:
    """Seed of one sampled generation; batches are 100 apart so they never overlap."""
    if int(config["n_samples"]) > 100:
        raise ValueError("n_samples above 100 would make the seeds of two batches overlap")
    return int(config["base_seed"]) + 100 * batch + run


def _git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "openai", "pyyaml"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


class GpuMonitor:
    """Samples memory, utilisation and power draw of the first GPU while a stage runs.

    The model server reserves its memory at start-up, so memory hardly moves during a run; the
    power draw is what the cost model needs. Without nvidia-smi the monitor records nothing.
    """

    QUERY = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]

    def __init__(self, interval_s: float = 5.0) -> None:
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float]] = []
        self._task: asyncio.Task | None = None

    async def _sample(self) -> None:
        process = await asyncio.create_subprocess_exec(
            *self.QUERY, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await process.communicate()
        try:
            memory, utilisation, power = (float(x) for x in out.decode().strip().splitlines()[0].split(","))
        except (IndexError, ValueError):
            return
        self.samples.append((memory, utilisation, power))

    async def _loop(self) -> None:
        while True:
            await self._sample()
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        self.samples = []
        if shutil.which("nvidia-smi") is not None:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> dict:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            await self._sample()  # a short stage still gets one sample
        if not self.samples:
            return {}
        memory, utilisation, power = zip(*self.samples)
        return {
            "gpu_memory_peak_mib": int(max(memory)),
            "gpu_utilisation_mean_percent": round(sum(utilisation) / len(utilisation), 1),
            "gpu_power_mean_watts": round(sum(power) / len(power), 1),
        }


def _finished(log_path: Path) -> dict[tuple[str, str, int], dict]:
    """Last record of every generation that needs no retry (everything except failed requests)."""
    done: dict[tuple[str, str, int], dict] = {}
    if not log_path.exists():
        return done
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["item_id"], record["kind"], record["seed"])
            if (record.get("error") or "").startswith("request:"):
                done.pop(key, None)
            else:
                done[key] = record
    return done


def _record(config: dict, item_id: str, kind: str, seed: int, batch: int, temperature: float, result: Generation) -> dict:
    engine = config["engine"]
    return {
        "run": config["run_name"],
        "model": engine.get("model"),
        "engine": engine.get("server", engine.get("kind", "openai")),
        "quantization": engine.get("quantization"),
        "source": config["source"],
        "role": config["role"],
        "prompt": config["prompt"],
        "item_id": item_id,
        "kind": kind,
        "seed": seed,
        "batch": batch,
        "temperature": temperature,
        "top_p": config["top_p"],
        "valid": result.labels is not None,
        "labels": result.labels,
        "p1": None if result.p1 is None else [round(p, 6) for p in result.p1],
        "verbal_overall": result.verbal_overall,
        "n_bounded": result.n_bounded,
        "ordered": result.ordered,
        "n_prompt_tokens": result.n_prompt_tokens,
        "n_output_tokens": result.n_output_tokens,
        "latency_s": round(result.latency_s, 4),
        "error": result.error,
        "raw": result.raw,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _votes(config: dict, item_ids: list[str], done: dict) -> dict[str, list[int]]:
    """Majority vote of batch 0 for every item whose sampled runs are all valid."""
    votes = {}
    for item_id in item_ids:
        records = [done.get((item_id, "sample", sample_seed(config, 0, r))) for r in range(int(config["n_samples"]))]
        if all(record is not None and record["valid"] for record in records):
            votes[item_id] = majority_vote(np.array([record["labels"] for record in records])).tolist()
    return votes


async def run(config: dict, engine: Engine | None = None, show_progress: bool = True) -> Path:
    config = {**DEFAULTS, **config}
    source = config["source"]
    if config["role"] == "test" and not config["allow_test"]:
        raise PermissionError(
            "the test sets stay untouched until the analysis plan is frozen (PLAN.md, decision 2); "
            "set allow_test: true in the config once it is"
        )
    labels = labels_for(source)
    items = load_role(source, config["role"])
    if config["limit"]:
        items = items.head(int(config["limit"]))
    if items.empty:
        raise ValueError(f"no items with role {config['role']!r} for {source}")

    run_dir = RUNS_DIR / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "generations.jsonl"
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    engine = engine or build_engine(config["engine"])
    item_ids = list(items["item_id"])
    messages = {
        item_id: build_messages(source, config["prompt"], text) for item_id, text in zip(item_ids, items["text"])
    }
    followup = verbal_followup(source, config["prompt"])
    base_seed = int(config["base_seed"])
    semaphore = asyncio.Semaphore(int(config["max_concurrency"]))
    stages: dict[str, dict] = {}
    monitor = GpuMonitor()

    def jobs_for(stage: str, done: dict) -> list[tuple]:
        """(item_id, seed, batch, temperature, decisions to assess) for everything still to do."""
        jobs: list[tuple] = []
        if stage == "sample":
            for item_id in item_ids:
                for batch in range(int(config["n_batches"])):
                    for r in range(int(config["n_samples"])):
                        seed = sample_seed(config, batch, r)
                        if (item_id, "sample", seed) not in done:
                            jobs.append((item_id, seed, batch, float(config["temperature"]), None))
        elif stage == "greedy" and config["greedy"]:
            jobs = [(i, base_seed, 0, 0.0, None) for i in item_ids if (i, "greedy", base_seed) not in done]
        elif stage in ("score", "verbal") and config["score_voted" if stage == "score" else "verbal"]:
            for item_id, vote in _votes(config, item_ids, done).items():
                if stage == "score" and not config["score_when_greedy_agrees"]:
                    greedy = done.get((item_id, "greedy", base_seed))
                    # Same decisions as the vote, so the same prefix at every decision token: the
                    # greedy pass already holds the teacher-forced probabilities (routing.analyze
                    # takes them from there).
                    if greedy is not None and greedy["valid"] and greedy["labels"] == vote and greedy.get("p1"):
                        continue
                previous = done.get((item_id, stage, base_seed))
                # Redo the query if the vote changed since it was made (a sampled run was retried).
                if previous is None or (previous["valid"] and previous["labels"] != vote):
                    jobs.append((item_id, base_seed, 0, 0.0, vote))
        elif stage == "verbal_greedy" and config["verbal_greedy"]:
            votes = _votes(config, item_ids, done)
            for item_id in item_ids:
                greedy = done.get((item_id, "greedy", base_seed))
                if greedy is None or not greedy["valid"] or greedy["labels"] == votes.get(item_id):
                    continue  # the answer about the vote covers an identical greedy answer
                previous = done.get((item_id, stage, base_seed))
                if previous is None or (previous["valid"] and previous["labels"] != greedy["labels"]):
                    jobs.append((item_id, base_seed, 0, 0.0, greedy["labels"]))
        return jobs

    async def call(stage: str, item_id: str, seed: int, temperature: float, decisions) -> Generation:
        if stage == "score":
            return await engine.score(messages[item_id], labels, decisions)
        if stage in ("verbal", "verbal_greedy"):
            return await engine.verbalise(messages[item_id], labels, decisions, followup, seed=seed)
        return await engine.generate(
            messages[item_id],
            labels,
            seed=seed,
            temperature=temperature,
            top_p=float(config["top_p"]),
            want_probabilities=(stage == "greedy"),
        )

    with open(log_path, "a", encoding="utf-8") as log:
        for stage in STAGES:
            jobs = jobs_for(stage, _finished(log_path))
            if not jobs:
                continue
            counts = {"generations": 0, "invalid": 0, "prompt_tokens": 0, "output_tokens": 0}
            progress = tqdm(total=len(jobs), desc=f"{config['run_name']} {stage}", disable=not show_progress)

            async def one(item_id: str, seed: int, batch: int, temperature: float, decisions) -> None:
                async with semaphore:
                    result = await call(stage, item_id, seed, temperature, decisions)
                record = _record(config, item_id, stage, seed, batch, temperature, result)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                counts["generations"] += 1
                counts["invalid"] += result.labels is None
                counts["prompt_tokens"] += result.n_prompt_tokens or 0
                counts["output_tokens"] += result.n_output_tokens or 0
                progress.update(1)

            monitor.start()
            started = time.perf_counter()
            await asyncio.gather(*(one(*job) for job in jobs))
            progress.close()
            wall = time.perf_counter() - started
            stages[stage] = {
                **counts,
                "wall_seconds": round(wall, 2),
                "output_tokens_per_second": round(counts["output_tokens"] / wall, 1) if wall > 0 else None,
                **(await monitor.stop()),
            }

    total = {key: sum(stage[key] for stage in stages.values()) for key in ("generations", "invalid", "prompt_tokens", "output_tokens")}
    wall = sum(stage["wall_seconds"] for stage in stages.values())
    meta = {
        "run_name": config["run_name"],
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_items": int(len(items)),
        "max_concurrency": int(config["max_concurrency"]),
        "wall_seconds": round(wall, 2),
        "output_tokens_per_second": round(total["output_tokens"] / wall, 1) if wall > 0 else None,
        **total,
        "stages": stages,
        "gpu_memory_used_mib": max((s.get("gpu_memory_peak_mib", 0) for s in stages.values()), default=None) or None,
        "git": _git_state(),
        "versions": _versions(),
        "platform": platform.platform(),
    }
    # Keep one metadata entry per invocation so a resumed run keeps its history.
    meta_path = run_dir / "meta.json"
    history = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    history.append(meta)
    meta_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--source", help="override the data source")
    parser.add_argument("--role", help="override the role (pilot, calibration, ...)")
    parser.add_argument("--prompt", help="override the prompt (p1, p2, p3)")
    parser.add_argument("--batches", type=int, help="override the number of seed batches")
    parser.add_argument("--concurrency", type=int, help="override the number of parallel requests")
    parser.add_argument("--limit", type=int, help="only the first N items")
    parser.add_argument("--run-name", help="override the run name")
    parser.add_argument("--fake", action="store_true", help="use the built-in fake engine instead of a model")
    args = parser.parse_args()

    config = load_config(
        args.config,
        source=args.source,
        role=args.role,
        prompt=args.prompt,
        n_batches=args.batches,
        max_concurrency=args.concurrency,
    )
    if args.limit:
        config["limit"] = args.limit
    if args.fake:
        config["engine"] = {"kind": "fake", "model": "fake"}
        config["run_name"] += "_fake"
    if args.run_name:
        config["run_name"] = args.run_name

    run_dir = asyncio.run(run(config))
    last = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))[-1]
    print(
        f"{last['generations']} requests in {last['wall_seconds']} s, "
        f"{last['invalid']} invalid, {last['output_tokens_per_second']} output tokens/s"
    )
    for stage, numbers in last["stages"].items():
        print(f"  {stage:<14} {numbers['generations']:>6} requests  {numbers['wall_seconds']:>8} s")
    print(f"log: {run_dir / 'generations.jsonl'}")
    print(f"next: python -m routing.analyze {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

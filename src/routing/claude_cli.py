"""Run Claude Opus 5.5 through the Claude Code CLI on a SemEval subset, as a state-of-the-art reference.

    python -m routing.claude_cli                        # SemEval PL pilot, prompt P1, 5 repeats
    python -m routing.claude_cli --limit 3 --repeats 1

Every request is a separate `claude -p` call that behaves as close to a plain API call as the CLI
allows: an empty working directory, no tools, no settings, no MCP servers, no saved session, a short
system prompt instead of the Claude Code one, and the answer constrained by the same JSON schema as
in the other runs. The prompt is P1, the same as for Bielik. The CLI has no temperature or seed;
the repeats are independent calls, so their agreement is the self-consistency signal.

Requires the `claude` CLI logged in (it uses the subscription, not an API key).

Output: outputs/runs/<run_name>/generations.jsonl, one line per call (item id, repeat, decisions,
token usage, list-price cost estimate, latency). The text is never logged. A rerun skips finished calls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .data import REPO_ROOT, load_role
from .engines import decision_schema, parse_decisions
from .labels import labels_for
from .prompts import build_messages

MODEL = "opus"
EFFORT = "low"
# SYSTEM_PROMPT is used to override default claude code system prompt, the prompt from the text is used as a user message just like in bielik
SYSTEM_PROMPT = "Jesteś asystentem do anotacji tekstów. Wykonaj zadanie z wiadomości użytkownika."
RUNS_DIR = REPO_ROOT / "outputs" / "runs"


def ask(prompt: str, labels: tuple[str, ...], workdir: Path) -> dict:
    command = [
        "claude", "-p",
        "--model", MODEL,
        "--effort", EFFORT,
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
        "--system-prompt", SYSTEM_PROMPT,
        "--json-schema", json.dumps(decision_schema(labels)),
        "--output-format", "json",
    ]
    started = time.perf_counter()
    process = subprocess.run(command, input=prompt, capture_output=True, text=True, cwd=workdir, timeout=600)
    latency = time.perf_counter() - started
    try:
        out = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"labels": None, "error": f"request: exit {process.returncode}: {process.stderr[-300:]}", "latency_s": round(latency, 3)}

    answer = out.get("structured_output")
    decisions = parse_decisions(json.dumps(answer), labels) if answer is not None else None
    usage = out.get("usage") or {}
    error = None
    if out.get("is_error"):
        error = f"request: {out.get('subtype')} {out.get('api_error_status')}"
    elif decisions is None:
        error = f"invalid_output ({out.get('stop_reason')})"
    return {
        "labels": decisions,
        "error": error,
        "models": list((out.get("modelUsage") or {}).keys()),
        "input_tokens": (usage.get("input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0),
        "cached_tokens": usage.get("cache_read_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd_list_price": out.get("total_cost_usd"),
        "latency_s": round(latency, 3),
    }


def finished(log_path: Path) -> set[tuple[str, int]]:
    done = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not (record["error"] or "").startswith("request:"):
                done.add((record["item_id"], record["repeat"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="semeval_pl")
    parser.add_argument("--role", default="pilot")
    parser.add_argument("--prompt", default="p1")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.role == "test":
        raise PermissionError("the test set stays locked (PLAN.md, decision 2)")

    labels = labels_for(args.source)
    items = load_role(args.source, args.role)
    if args.limit:
        items = items.head(args.limit)

    run_dir = RUNS_DIR / f"opus55cli_{args.source}_{args.role}_{args.prompt}"
    workdir = run_dir / "empty_cwd"  # the CLI runs here, so it cannot see the repository
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "generations.jsonl"
    done = finished(log_path)
    jobs = [
        (item_id, text, repeat)
        for repeat in range(args.repeats)
        for item_id, text in zip(items["item_id"], items["text"])
        if (item_id, repeat) not in done
    ]

    def one(job: tuple[str, str, int]) -> dict:
        item_id, text, repeat = job
        messages = build_messages(args.source, args.prompt, text)
        result = ask(messages[0]["content"], labels, workdir)
        return {
            "model": "claude-opus-5-5 via Claude Code CLI",
            "effort": EFFORT,
            "source": args.source,
            "role": args.role,
            "prompt": args.prompt,
            "item_id": item_id,
            "repeat": repeat,
            **result,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    if args.prompt == "p3":
        raise ValueError("p3 is a multi-turn few-shot prompt, not supported by this script")

    started = time.perf_counter()
    errors = 0
    with open(log_path, "a", encoding="utf-8") as log, ThreadPoolExecutor(args.workers) as pool:
        for record in tqdm(pool.map(one, jobs), total=len(jobs)):
            errors += record["error"] is not None
            log.write(json.dumps(record) + "\n")
            log.flush()
    print(f"{len(jobs)} calls in {time.perf_counter() - started:.1f} s, {errors} with errors")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()

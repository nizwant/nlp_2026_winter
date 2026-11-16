"""Run the OpenAI Decisions API (gpt-6-luna) on a SemEval subset, as a state-of-the-art reference.

    python -m routing.decisions_api                     # SemEval PL pilot, 5 repeats
    python -m routing.decisions_api --limit 3 --repeats 1

Each request holds one paragraph and 23 predicate questions, one per technique. The API returns a
probability per technique. The API has no temperature or seed, so self-consistency is measured by
sending the same request several times (repeats).

The instructions with technique definitions are the same as in prompt P1 and go first as a
developer message, so they form a shared prefix for the prompt cache. The API key is read from
OPENAI_API_KEY or from the .env file in the repository root.

Output: outputs/runs/<run_name>/generations.jsonl, one line per request (item id, repeat,
probabilities, token usage, latency). The text is never logged. A rerun skips finished requests.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from openai import OpenAI
from tqdm import tqdm

from .data import REPO_ROOT, load_role
from .labels import labels_for
from .prompts import _definitions, _template, technique_list

MODEL = "gpt-6-luna"
RUNS_DIR = REPO_ROOT / "outputs" / "runs"


def load_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for line in (REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY":
            return value.strip().strip('"').strip("'")
    raise KeyError("OPENAI_API_KEY not found in the environment or in .env")


def build_request(source: str, text: str) -> dict:
    """Shared instructions as a developer message, the paragraph as a user message, one question per technique."""
    instructions, block = _template("semeval", "pl")
    # The P1 rule about answering with a JSON object does not apply here, the API returns typed answers.
    head = "\n".join(
        line for line in instructions.substitute(techniques=technique_list(source, "pl")).splitlines()
        if not line.startswith("- Odpowiedz wyłącznie")
    )
    definitions = _definitions("semeval")
    questions = [
        {
            "type": "predicate",
            "name": label,
            "instructions": f"W akapicie występuje technika {label} ({definitions[label]['name_pl']}).",
        }
        for label in labels_for(source)
    ]
    return {
        "model": MODEL,
        "input": [
            {"role": "developer", "content": head},
            {"role": "user", "content": block.substitute(text=text.strip())},
        ],
        "questions": questions,
        "prompt_cache_key": f"{source}_decisions",
    }


def ask(client: OpenAI, request: dict, labels: tuple[str, ...]) -> dict:
    started = time.perf_counter()
    try:
        response = client.decisions.create(**request)
    except Exception as error:
        return {"p1": None, "error": f"request: {error}"[:300], "latency_s": time.perf_counter() - started}
    latency = time.perf_counter() - started

    probabilities = {}
    refused = []
    for answer in response.answers:
        if answer.type == "predicate":
            probabilities[answer.name] = float(answer.probability)
        else:
            refused.append(answer.name)
    usage = response.usage
    details = getattr(usage, "input_tokens_details", None)
    return {
        "p1": [probabilities.get(label) for label in labels],
        "error": f"refused: {refused}" if refused else None,
        "input_tokens": getattr(usage, "input_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", None),
        "latency_s": round(latency, 3),
    }


def finished(log_path) -> set[tuple[str, int]]:
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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.source != "semeval_pl":
        raise ValueError("only semeval_pl is prepared for the Decisions API")
    if args.role == "test":
        raise PermissionError("the test set stays locked (PLAN.md, decision 2)")

    labels = labels_for(args.source)
    items = load_role(args.source, args.role)
    if args.limit:
        items = items.head(args.limit)

    run_dir = RUNS_DIR / f"gpt6luna_{args.source}_{args.role}_decisions"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "generations.jsonl"
    done = finished(log_path)
    jobs = [
        (item_id, text, repeat)
        for repeat in range(args.repeats)
        for item_id, text in zip(items["item_id"], items["text"])
        if (item_id, repeat) not in done
    ]

    client = OpenAI(api_key=load_api_key(), max_retries=3)

    def one(job: tuple[str, str, int]) -> dict:
        item_id, text, repeat = job
        result = ask(client, build_request(args.source, text), labels)
        return {
            "model": MODEL,
            "source": args.source,
            "role": args.role,
            "item_id": item_id,
            "repeat": repeat,
            **result,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    started = time.perf_counter()
    errors = 0
    with open(log_path, "a", encoding="utf-8") as log, ThreadPoolExecutor(args.workers) as pool:
        for record in tqdm(pool.map(one, jobs), total=len(jobs)):
            errors += record["error"] is not None
            log.write(json.dumps(record) + "\n")
            log.flush()
    print(f"{len(jobs)} requests in {time.perf_counter() - started:.1f} s, {errors} with errors")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()

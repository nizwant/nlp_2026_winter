"""Serve a model from configs/models.yaml with vLLM's OpenAI-compatible server.

    docker compose up vllm                       # bielik, http://127.0.0.1:8000/v1
    MODEL=pllum docker compose up vllm
    python scripts/serve.py pllum --max-num-seqs 64   # native; extra args go to `vllm serve`

Memory and context come from GPU_MEM_UTIL and MAX_MODEL_LEN (see .env.example).
"""

import os
import sys

from routing.models import get_model, serve_args


def main() -> None:
    try:
        spec = get_model(sys.argv[1] if len(sys.argv) > 1 else "bielik")
    except KeyError as e:
        sys.exit(str(e))
    cmd = [
        "vllm", "serve", *serve_args(spec),
        "--host", os.environ.get("VLLM_HOST", "127.0.0.1"),
        "--port", os.environ.get("VLLM_PORT", "8000"),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "8192"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEM_UTIL", "0.82"),
        "--seed", "0",
        *sys.argv[2:],
    ]
    print("+", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Serve one of the study's models with vLLM on http://127.0.0.1:8000/v1.
#
#   scripts/serve_vllm.sh bielik                 # Bielik-11B-v3.0-Instruct, 4-bit AWQ
#   MAX_MODEL_LEN=12288 scripts/serve_vllm.sh bielik
#
# Needs the `vllm` environment (see README). Settings that matter for the experiments:
#   - the model revision is pinned, so the weights cannot change under a run;
#   - VLLM_USE_FLASHINFER_SAMPLER=0 selects vLLM's own PyTorch top-p sampler. The FlashInfer
#     sampler compiles CUDA code on first use and needs ninja and a matching CUDA toolkit;
#   - GPU_UTIL is the share of the whole card vLLM may take. On a 12 GB card with a desktop
#     session and a browser about 9.9 GiB are free, so the default is 0.82 (14,336 KV-cache tokens
#     for Bielik). With only the desktop shell running (about 0.2 GB), GPU_UTIL=0.90 works and
#     gives 19,200 tokens, enough for 40 parallel requests (configs/bielik.yaml).
set -euo pipefail

MODEL_KEY="${1:-bielik}"
case "$MODEL_KEY" in
  bielik)
    REPO="speakleash/Bielik-11B-v3.0-Instruct-awq"
    REVISION="0de6dd10ddcbd570b3844f1ae0e98f8e4800b548"
    NAME="Bielik-11B-v3.0-Instruct"
    ;;
  *)
    echo "unknown model key: $MODEL_KEY (known: bielik)" >&2
    exit 2
    ;;
esac

VLLM_BIN="${VLLM_BIN:-vllm}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

exec "$VLLM_BIN" serve "$REPO" \
  --revision "$REVISION" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.82}" \
  --seed 0

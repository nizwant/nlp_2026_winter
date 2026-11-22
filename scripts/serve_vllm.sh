#!/usr/bin/env bash
# Serve one of the study's models with vLLM on http://127.0.0.1:8000/v1.
#
#   scripts/serve_vllm.sh bielik                 # Bielik-11B-v3.0-Instruct, 4-bit AWQ
#   scripts/serve_vllm.sh pllum                  # PLLuM-12B-chat-2512, 4-bit AWQ
#   scripts/serve_vllm.sh mistral_nemo           # Mistral-Nemo-Instruct-2407, 4-bit AWQ
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
#   - MAX_NUM_SEQS caps the parallel sequences vLLM reserves sampler memory for (default 256).
#     PLLuM and Mistral-Nemo have a 131k-token vocabulary, so this reserve is four times
#     Bielik's; the client never sends more than 40 requests at once.
set -euo pipefail

MODEL_KEY="${1:-bielik}"
case "$MODEL_KEY" in
  bielik)
    REPO="speakleash/Bielik-11B-v3.0-Instruct-awq"
    REVISION="0de6dd10ddcbd570b3844f1ae0e98f8e4800b548"
    NAME="Bielik-11B-v3.0-Instruct"
    ;;
  pllum)
    # Community AWQ build (llm-compressor, W4A16 symmetric) of CYFRAGOVPL/PLLuM-12B-chat-2512;
    # no official 4-bit build exists. Base model: Mistral-Nemo-Base-2407.
    REPO="mozarcik/PLLuM-12B-chat-2512-awq"
    REVISION="2ad2863e47f26fcb75f0f2dad40d3e3959a15931"
    NAME="PLLuM-12B-chat-2512"
    ;;
  mistral_nemo)
    # Community AWQ build (llm-compressor, W4A16 asymmetric) of mistralai/Mistral-Nemo-Instruct-2407.
    REPO="pbhappliedsystems/Mistral_Nemo_Instruct_2407_4bit_AWQ_Async"
    REVISION="98d07b0643f435a75e115cffbda6daacacbd6216"
    NAME="Mistral-Nemo-Instruct-2407"
    ;;
  *)
    echo "unknown model key: $MODEL_KEY (known: bielik, pllum, mistral_nemo)" >&2
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
  ${MAX_NUM_SEQS:+--max-num-seqs "$MAX_NUM_SEQS"} \
  --seed 0

# Is LLM self-consistency a usable signal for routing annotation work?

## Project description

The project is a part of the Natural Language Processing course at the Warsaw University of Technology. The project's goal is to check whether self-consistency across LLM reruns predicts correctness of multilabel annotations of manipulation techniques in English and Polish texts and supports cost-effective human-model routing. Negative results are considered valid research outcomes.

Authors: Tymoteusz Urban, Sebastian Trojan, Mateusz Nizwantowski  
Supervisor: Soveatin Kuntur - soveatin.kuntur.dokt@pw.edu.pl

## Useful Resources

- course main repo: https://github.com/grant-TraDA/NLP-2026W
- overleaf lind with reports: https://www.overleaf.com/project/6ae711011f26b6a0c20dceb7

| Phase | Deliverables | Points | Deadline |
| --- | --- | ---: | --- |
| 1. Literature review and concept | Report (max. 4 meaningful pages): state of the art (SOTA) and relevant datasets | 10 | 4.11 |
| 2. Reproducible PoC | Merge request with source code; EDA notebooks; preliminary models that reproduce; presentation with revised SOTA; updated report | 20 | 2.12 |
| 3. Complete solution | Extended report covering all phases, results, and analysis; merge request; final presentation | 30 | 13.01.2027 |

### Self-check before you submit

- [ ] Contribution table, one row per person, with hours — without it the report is not accepted, in every phase
- [ ] LLM-use declaration — same rule
- [ ] README that lets a stranger run one experiment
- [ ] Runtime and memory measured and reported
- [ ] Every number in every table says which split it comes from
- [ ] Every preprint checked for a published version
- [ ] Data statement present
- [ ] Rebuttal answers every review point AND every Q&A question, at a named place, checkably
- [ ] Correct template, submitted on time
- [ ] Folder named surname_surname_surname, same order as Milestone 1

## Environment

One Docker image holds everything: vLLM (as a Python library or as an OpenAI-compatible server), the analysis stack, JupyterLab and pytest. Python packages are pinned in `uv.lock`; model weights are pinned to a Hugging Face commit in `configs/models.yaml`. Both team GPUs are supported: RTX 4070 SUPER (12 GB, sm_89) and RTX 5070 Ti (16 GB, sm_120).

| Component | Version |
| --- | --- |
| vLLM | 0.33.0 |
| PyTorch | 2.14.0, CUDA 13.0 |
| Python | 3.12 (uv-managed) |
| Base image | `nvidia/cuda:13.0.3-devel-ubuntu24.04` |

### Host requirements

- Linux, or Windows with WSL2.
- NVIDIA driver **580 or newer**: `nvidia-smi` must report `CUDA Version: 13.0` or higher.
- Docker with the NVIDIA runtime. On Ubuntu with apt-installed Docker: install `nvidia-container-toolkit`, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. The snap Docker package already includes the runtime. Test with:
  `docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu:24.04 nvidia-smi`
- About 20 GB of disk for the image (18 GB measured), plus 6–8 GB per model.

### First run

```bash
cp .env.example .env            # set GPU_MEM_UTIL for your card; HOST_UID/HOST_GID if not 1000
docker compose build
docker compose run --rm dev python scripts/check_env.py
```

`check_env.py` checks that the installed packages match `uv.lock`, that the driver and GPU are supported, and that vLLM loads Bielik, generates text, returns log-probabilities and reproduces seeded samples. It writes `outputs/env_check/<gpu>.json`. Commit that file so the reports from both machines can be compared.

### Daily use

```bash
docker compose run --rm dev                   # shell with the GPU; the repo is mounted at /workspace
docker compose up jupyter                     # JupyterLab, URL with token in the log; http://127.0.0.1:8888
MODEL=pllum docker compose up vllm            # OpenAI-compatible API at http://127.0.0.1:8000/v1
docker compose run --rm dev pytest
```

The repository is bind-mounted, so code edits take effect without rebuilding. The host Hugging Face cache (`~/.cache/huggingface`) is mounted too, so weights are downloaded once. vLLM, Triton and torch.compile caches persist in the `cache` volume. Rebuild only when `pyproject.toml` or `uv.lock` changes.

### Models

`configs/models.yaml` pins each model's repository and commit. Load models in code through `routing.models` (`make_llm(get_model("bielik"))`), so every script and notebook uses the same revision, engine arguments and chat-template settings.

| Key | Weights | Role |
| --- | --- | --- |
| `bielik` | speakleash/Bielik-11B-v3.0-Instruct-awq (official AWQ) | study |
| `pllum` | mozarcik/PLLuM-12B-chat-2512-awq (community AWQ) | study |
| `mistral_nemo` | pbhappliedsystems/Mistral_Nemo_Instruct_2407_4bit_AWQ_Async (community AWQ) | study |
| `gemma4_e2b` | google/gemma-4-E2B-it-qat-w4a16-ct (official QAT, 4-bit) | smoke tests and speed only; text-only, thinking off |

```bash
docker compose run --rm dev python scripts/check_env.py --model gemma4_e2b   # smoke test any model
docker compose run --rm dev python scripts/bench.py                          # tokens/s for all models
```

`bench.py` runs each model on 100 SemEval PL paragraphs × 5 samples, with every completion forced to 256 new tokens, plus a single-request decode test. It writes `outputs/bench/<gpu>.json`.

Measured on the RTX 4070 SUPER (12 GB), 22 Nov 2026, `max_model_len` 2048:

| Model | single request, tok/s | batch, output tok/s | paragraphs/s (5 samples each) | batch-invariant: single / batch tok/s | prompt tokens | engine limits |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gemma4_e2b | 181 | 5,521 | 4.31 | 31 / 2,435 | 153 | util 0.82 |
| bielik | 71 | 1,291 | 1.01 | 8 / 334 | 209 | util 0.82 |
| pllum | 61 | 936 | 0.73 | does not fit | 155 | util 0.86, 32 seqs, 2048 batched tokens |
| mistral_nemo | 63 | 922 | 0.72 | does not fit | 155 | util 0.86, 32 seqs, 2048 batched tokens |

- Bielik's tokenizer needs about 35% more tokens than the others for the same paragraphs (prompt tokens column), so compare models in paragraphs/s, not tok/s.
- On 12 GB, the 12B models run only with the limits shown (see `.env.example`). Their KV cache then holds about 4,100 tokens, roughly 9 sequences in flight, so batch throughput is capped by memory. In batch-invariant mode they do not start at all. Run them on the 16 GB card.
- Batch-invariant mode costs 4× batch throughput for Bielik and 2.3× for Gemma.

### Changing dependencies

Edit `pyproject.toml`, then refresh the lock and rebuild:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e UV_CACHE_DIR=/uvc -e HOME=/uvc \
  -v ~/.cache/uv:/uvc -v "$PWD":/w -w /w ghcr.io/astral-sh/uv:0.13.9-python3.12-trixie-slim uv lock
docker compose build
```

Commit `pyproject.toml` and `uv.lock` together, and both of us rebuild. Changing the vLLM version also changes torch, CUDA and the kernels, so rerun `check_env.py` on both GPUs afterwards.

### Without Docker

On Linux with driver 580 or newer, `uv sync && source .venv/bin/activate` installs the same locked packages. Also `export VLLM_USE_FLASHINFER_SAMPLER=0`, which `compose.yaml` sets for you. Docker is still the reference environment.

### Reproducibility

**A seed alone does not make vLLM's sampled outputs reproducible.** vLLM gives each of the `n` samples its own seed, so the random draws are fixed. The logits are not: GPU kernels give slightly different numbers depending on how many sequences share a batch, and the batch composition at each step depends on request timing. When a random draw lands near a token boundary, those tiny differences flip the sampled token, and the completion diverges from there. Measured on the 4070 SUPER with Bielik (5 samples × 2 prompts, seed 1234, repeated in one process and across processes):

| `VLLM_BATCH_INVARIANT` | greedy | seeded `n=1` | seeded `n=5`, same process | seeded `n=5`, new process |
| --- | --- | --- | --- | --- |
| `0` (default) | identical | identical | 1 of 4 repeats differed | differed |
| `1` | identical | identical | identical | identical (same hash in 3 processes) |

`VLLM_BATCH_INVARIANT=1` (in `.env`, or `docker compose run -e VLLM_BATCH_INVARIANT=1 ...`) switches vLLM to batch-invariant kernels; 4-bit weights are then dequantized for a deterministic matmul instead of using the fast Marlin kernel. `outputs/bench/` holds the speed in both modes. `check_env.py` reports whether seeded samples reproduced.

Across the two GPUs, outputs are not expected to match even in batch-invariant mode, because each architecture uses different kernels. Compare `greedy_fingerprint` in the two env reports to see. In practice, run all reruns for a given model × prompt × item set on one GPU, and record the GPU in every run log.

## Datasets

- SemEval-2023 Task 3:
  - pl - https://github.com/yrf1/SemEval23-Task-3-UIUC-Team/tree/main/data/po
  - en - https://github.com/yrf1/SemEval23-Task-3-UIUC-Team/tree/main/data/en

- MIPD:
  - intention and manipulation tasks - https://github.com/ArkadiusDS/MIPD/tree/master/data

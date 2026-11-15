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

The full plan to the end of the project, including the answer to every Phase 1 review point, is in [PLAN.md](PLAN.md).

## Setup

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate nlp-routing
pip install -e .
pytest -q                               # runs without a GPU and without a model
```

This environment holds the client, the evaluation code and the tests. The model itself is served separately by any OpenAI-compatible server (Ollama, llama.cpp server or vLLM); which one is used is a line in the config.

## Data

The prepared files are in `data/`. Both sources are used for non-commercial research only; the two scripts below rebuild the prepared files from the original distributions.

- **MIPD** (Modzelewski et al., EMNLP 2024), CC BY-NC-ND 4.0. `python scripts/download_mipd.py` downloads the files from the authors' repository (https://github.com/ArkadiusDS/MIPD, pinned commit) into `data/raw/mipd/` and builds `data/mipd/`.
- **SemEval-2023 Task 3, subtask 3** (Piskorski et al., 2023), English and Polish, for scientific research purposes only under the organisers' terms. Access is by registration at https://propaganda.math.unipd.it/semeval2023task3/; `python scripts/prepare_semeval.py --raw-dir <downloaded data folder>` builds `data/sem_eval/{en,pl}.csv` from the official files.

`python -m routing.splits` then writes `data/splits/<source>.csv` with the role of every item (few-shot pool, pilot, calibration, test). The official splits are the boundaries: SemEval dev and the MIPD test split are the test sets.

## Run the proof of concept

```bash
# 1. serve a model with llama.cpp's server (GPU build)
llama-server -hf speakleash/Bielik-11B-v3.0-Instruct-GGUF:Q4_K_M \
    --host 127.0.0.1 --port 8080 -ngl 99 -c 12288 -np 4 --no-webui

# 2. five sampled runs and one greedy run per item on the SemEval PL pilot subset
python -m routing.run configs/poc_semeval_pl.yaml

# 3. task quality, and how well each confidence signal predicts correctness
python -m routing.analyze outputs/runs/poc_semeval_pl_bielik
```

`python -m routing.run configs/poc_semeval_pl.yaml --fake` runs the same pipeline with a built-in fake engine, which needs no model and is useful for checking an installation. Its numbers carry no meaning.

The analysis prints three server checks before the results: greedy decisions must agree with their own token probabilities, answers must list the techniques in the configured order, and the share of positive sampled decisions should be close to what the probabilities predict. Ollama's `/v1` endpoint fails the first two (it applies a fixed repetition penalty and sorts the keys), so it is not used for real runs.

Each run writes `outputs/runs/<run_name>/generations.jsonl` with one line per generation: item id, seed, the 0/1 decision per technique and, for the greedy run, the probability of each decision. The log never contains the text. `outputs/runs/` is not committed.

## Repository layout

| Path | Content |
| --- | --- |
| `src/routing/` | `data`, `splits`, `prompts`, `engines`, `run`, `signals`, `metrics`, `analyze` |
| `prompts/` | prompt templates and technique definitions in Polish and English |
| `configs/` | one YAML file per experiment |
| `scripts/` | dataset download and preparation |
| `tests/` | unit tests and an end-to-end test with the fake engine |
| `notebooks/` | data preparation and exploration |
| `latex/`, `deliverables/` | report sources and submitted PDFs |

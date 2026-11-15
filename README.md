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

## What is in the repository

The pipeline asks a language model which persuasion or manipulation techniques occur in a text, five times with different seeds, takes the majority vote per technique, and then checks which confidence signal best predicts that the voted label set is exactly right: agreement between the five runs (self-consistency), the model's token log-probability, its stated confidence, or trivial controls. The signals are then used to decide which items a model may label on its own and which go to a human.

| Path | Content |
| --- | --- |
| `src/routing/` | `data`, `splits`, `prompts`, `fewshot`, `engines`, `run`, `signals`, `metrics`, `analyze`, `selective`, `route`, `cost`, `baseline_tfidf`, `annotate` |
| `prompts/` | prompt templates, technique definitions in Polish and English, ids of the few-shot examples |
| `configs/` | one YAML file per model, and the rates of the cost model |
| `scripts/` | dataset download and preparation, vLLM start script |
| `tests/` | unit tests and an end-to-end test with a fake engine (no GPU, no model) |
| `notebooks/01_eda.ipynb` | exploratory data analysis, committed with outputs |
| `data/` | prepared datasets and `data/splits/` with the role of every item |
| `outputs/public/` | released model outputs (decisions only, no text) and `outputs/eda/` tables |
| `latex/`, `deliverables/` | report sources and submitted PDFs |

## Setup

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate nlp-routing
pip install -e .
pytest -q                               # 54 tests, about 25 s, no GPU and no model needed
```

The same environment as a container: `docker build -t nlp-routing . && docker run --rm nlp-routing` runs the tests.

## Reproduce one result without a GPU

The outputs of the first pilot run are in the repository (item ids and 0/1 decisions, no text), so its analysis can be repeated on any machine in under a minute:

```bash
python -m routing.baseline_tfidf semeval_pl                              # optional: adds the TF-IDF signal, 35 s
python -m routing.analyze outputs/public/bielik_semeval_pl_pilot_p1
```

This prints the task quality of Bielik-11B-v3.0 on the Polish SemEval pilot subset (219 paragraphs) and, for every confidence signal, the AUROC for predicting that the voted label set is exactly right. Expected: micro-F1 0.285, exact-set accuracy 0.311, AUROC 0.569 for the Jaccard consistency score and 0.722 for the teacher-forced log-probability.

## Run an experiment on a GPU

The model is served by vLLM in an environment of its own; the client above talks to it over the OpenAI-compatible API.

```bash
# once: environment for the server (about 9 GB with PyTorch and CUDA libraries)
mamba create -n vllm python=3.12 pip && conda activate vllm && pip install vllm==0.33.0

# terminal 1: serve Bielik-11B-v3.0-Instruct, official 4-bit AWQ build (6.2 GB download on first start)
scripts/serve_vllm.sh bielik

# terminal 2: five sampled runs, one greedy run, scoring and stated confidence for every pilot paragraph
conda activate nlp-routing
python -m routing.run configs/bielik.yaml                       # about 14 min on an RTX 4070 SUPER (12 GB)
python -m routing.analyze outputs/runs/bielik_semeval_pl_pilot_p1
```

Other data, roles and prompts use the same config: `python -m routing.run configs/bielik.yaml --source mipd --role calibration --prompt p3`. Add `--limit 20` for a quick check, or `--fake` to run the whole pipeline with a built-in fake engine (its numbers carry no meaning). An interrupted run continues where it stopped when the command is repeated.

Routing, with thresholds chosen on one run and applied to another:

```bash
python -m routing.route --calibration outputs/runs/bielik_semeval_pl_calibration_p1 \
                        --evaluation  outputs/runs/bielik_semeval_pl_pilot_p1
```

Measured on the 12 GB card with a desktop session running: 5.8 GiB for the weights, 9.5 GiB reserved by vLLM in total, context 8,192 tokens, about 620 output tokens per second at 16 parallel requests, 3.8 s per paragraph for all eight requests.

Every run writes `outputs/runs/<run>/generations.jsonl` with one line per request (item id, seed, the 0/1 decision per technique, probabilities, token counts, latency) and `meta.json` with timings, GPU power draw and memory, package versions and the git commit. The log never contains the text. Before the results, the analysis prints three server checks: greedy decisions must agree with their own token probabilities, answers must list the techniques in the configured order, and the share of positive sampled decisions should be close to what the probabilities predict. Ollama's `/v1` endpoint fails the first two (fixed repetition penalty, sorted keys) and is not used.

The test sets are locked: `routing.run` and `routing.baseline_tfidf` refuse the `test` role until `allow_test: true` (or `--allow-test`) is set, which happens only after the analysis plan is frozen in the Phase 2 report.

## Data

The prepared files are in `data/`. Both sources are used for non-commercial research only; the two scripts below rebuild the prepared files from the original distributions.

- **MIPD** (Modzelewski et al., EMNLP 2024), CC BY-NC-ND 4.0. `python scripts/download_mipd.py` downloads the files from the authors' repository (https://github.com/ArkadiusDS/MIPD, pinned commit) into `data/raw/mipd/` and builds `data/mipd/`, including the credible/disinformative label.
- **SemEval-2023 Task 3, subtask 3** (Piskorski et al., 2023), English and Polish, for scientific research purposes only under the organisers' terms. Access is by registration at https://propaganda.math.unipd.it/semeval2023task3/; `python scripts/prepare_semeval.py --raw-dir <downloaded data folder>` builds `data/sem_eval/{en,pl}.csv` from the official files.

`python -m routing.splits` writes `data/splits/<source>.csv` with the role of every item. The official splits are the boundaries: SemEval train supplies the few-shot pool, the pilot subset and the calibration set, and SemEval dev is the test set; for MIPD the pilot comes from train, calibration from validation, and the official test split is the test set. `python -m routing.fewshot` chooses the few-shot examples (stored as item ids). `notebooks/01_eda.ipynb` describes the data.

## Other commands

| Command | Purpose |
| --- | --- |
| `python -m routing.baseline_tfidf` | TF-IDF with logistic regression: task baseline and a cheap confidence signal |
| `python -m routing.annotate semeval_pl --annotator XY` | timed human annotation of a fixed sample, for the cost model |
| `python -m routing.annotate --summary` | median annotation time and agreement between annotators |
| `jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb` | regenerate the EDA and its figures |

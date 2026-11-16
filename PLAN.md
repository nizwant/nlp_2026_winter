# Project plan: LLM self-consistency as a routing signal

Status date: Sunday 15 Nov 2026. Covers everything from today to the final submission on 13 Jan 2027.
Team: Sebastian Trojan (S), Tymoteusz Urban (T), Mateusz Nizwantowski (M). Supervisor: Soveatin Kuntur (available Wednesdays).

## Context

Phase 1 scored 6.5/10. Phase 2 (20 pts, due 2 Dec) and Phase 3 (30 pts, due 13 Jan) are still open, and a 10-minute mid-term video is due in three days (18 Nov). The repository holds only data preparation; there is no environment, prompt, inference or evaluation code, and the teammates have nothing outside the repo. The Phase 1 review also left about 30 concrete points that Phase 2 must answer one by one (criterion 2.4, 3 pts). This plan fixes the open design decisions, orders the work by deadline, and maps every review point and rubric line to a task.

## Where the project stands

- **Graded so far:** 1.1 SOTA 1.5/3, 1.2 datasets 1.5/3, 1.3 concept 2/2, 1.4 text 0.5/1, 1.5 data statement 1/1 in the score table (the written justification says 0.5/1; the fix is needed either way).
- **Code:** `notebooks/02_data_preparation.ipynb` builds `data/mipd/{articles,manipulation,intention}.csv` and `data/sem_eval/{en,pl}.csv`. `src/` is empty. `notebooks/01_label_distribution.ipynb` no longer runs because it reads raw files deleted in commits `92e4fcf` and `fdd6052`.
- **Data facts measured today:** SemEval PL 3,110 paragraphs / 194 articles (the proposal said 3,094; the 16 rows were recovered), 46.8% with no label. SemEval EN 12,625 / 536, 61.3% empty, 1,072 rows share a text with another row, 29 duplicated texts carry conflicting gold labels, 30 texts appear in both train and dev. MIPD 15,356 articles, 3,702 with at least one technique, 1,265 with an intention label but no technique; 5% of articles exceed about 1,900 words, the longest has 25,018.
- **Machine (M):** RTX 4070 SUPER, 12 GB. `nvidia-smi` currently fails (kernel module 580.173, userspace 580.178) until a reboot. Docker is installed, the NVIDIA container toolkit is not. No Python environment for the project. Ollama 0.32.6 is installed.
- **Compute available:** 12 GB (M) and a 16 GB RTX 5070 Ti (teammate). 4-bit quantized models are accepted by the team.
- **Sibling team (idczak_twardowski_wnek):** the teams work independently (team decision). Their Phase 1 report plans Bielik, PLLuM and Gemma-3-12B on SemEval PL dev (800 paragraphs) and 1,000 MIPD test articles, with signals per item and label.

## Two points to settle before any experiment

1. **Data access (settled on 15 Nov).** The team has access to the official SemEval-2023 Task 3 data through registration at the task site. The organisers' terms allow use "only for scientific research purposes"; MIPD is CC BY-NC-ND 4.0. The prepared CSVs stay in `data/` (team decision), so no change to the repository history is needed. This was the single largest Phase 1 deduction (1.2 and 1.5), so the report must state the access route, the terms and the access status of each source.
   - `scripts/download_mipd.py` fetches MIPD from the authors' repository at a pinned commit and rebuilds `data/mipd/`; `scripts/prepare_semeval.py --raw-dir <official files>` rebuilds `data/sem_eval/{en,pl}.csv`.
   - Still to do: run `prepare_semeval.py` on the official download and confirm that the result is identical to the current CSVs (the script prints the checksums); state the result in the report.
   - The repo stays public so graders can check the contribution table against commit history (criterion 2.8).
2. **Independence from the sibling team must be on record.** The review asks for an agreed division of roles. Working independently is a legitimate answer only if it is documented: get the supervisor's written confirmation on Wed 18 Nov and quote the date in the rebuttal. The mitigation is in the design below (standard splits, so their evaluation items are a subset of ours; released item lists and log schema).

## Calendar

| Date | Event | What is due |
|---|---|---|
| Wed 18 Nov | Mid-term video | 10 minutes: problem, SOTA, method, PoC progress, first numbers, next steps. Not graded, but feeds 2.1, 2.3, 2.5. Meet supervisor. |
| Wed 25 Nov | Mid-term Q&A (meeting 8) | Each member brings at least 5 questions for other teams (1 pt each, max 10 over both sessions). Write down every question we receive, verbatim. |
| Mon 30 Nov | Internal freeze | Phase 2 content complete; analysis plan frozen. |
| Wed 2 Dec | Phase 2 (20 pts) | Report (about 8 pages plus appendices), rebuttal, EDA, PoC, README, pull request to the course repo. |
| December, date to confirm | Survey presentation and tutorial (10 pts, separate from the project) | Slides, 3-5 quiz questions, demo notebook under 10 minutes. |
| Sun 3 Jan | Internal | Complete final-report draft. |
| Wed 13 Jan | Phase 3 (30 pts) and final Q&A (meeting 13) | Final report (about 9 pages, max 12, plus reproducibility appendix), merge request, presentation, rebuttal. |

## Design decisions

| # | Decision | Reason |
|---|---|---|
| 1 | Use the official splits as boundaries. SemEval: train supplies few-shot examples, a pilot subset and the calibration set; dev is the test set. MIPD: train supplies few-shot examples and the pilot; validation supplies calibration; official test is the test set. | Standard and nameable for every number, article-disjoint (verified: zero shared articles), comparable with published MIPD test results and with the sibling team. Replaces "pool and re-split" from the proposal; stated as a deviation. |
| 2 | Test sets stay unanalysed until the analysis plan is frozen in the Phase 2 report. All Phase 2 numbers come from pilot and calibration data. | Metrics and operating-point rules are fixed before results exist. |
| 3 | The three sources are never pooled. Every table reports SemEval PL, SemEval EN and MIPD separately. | Review point; different label sets and units. |
| 4 | MIPD: add the credible/disinformative label from the dataset's `data/disinformation/` folder. Primary results on disinformative articles, where techniques were annotated; credible articles reported separately as a false-alarm check. Articles are cut at 2,000 words (about 5% affected), flagged, with a with/without comparison. | Answers the empty-label point and the supervisor's length remark with a precise, model-independent rule. |
| 5 | Output is a JSON object with one 0/1 value per technique in fixed order, enforced by structured output. No free text, no reasoning. | No parsing ambiguity; the probability of each decision token is directly readable; outputs contain no generated prose, so label outputs can be released. A list-of-labels format is run on the pilot as an ablation. |
| 6 | Per item, prompt and model: 5 samples at temperature 0.7, top-p 0.95, logged seeds; 1 greedy pass with token log-probabilities; teacher-forced scoring of the majority vote where it differs from the greedy answer (where they are equal, the greedy pass already holds the same probabilities); 1 stated-confidence query for the whole answer (0-100). | "Fixed decoding" means fixed settings; greedy reruns would be identical. Stated confidence per technique was dropped after the pilot: Bielik's per-technique values could not be interpreted (see findings). Temperatures 0.3 and 1.0 on the pilot as an ablation. |
| 7 | Three prompts per source: P1 zero-shot with instructions in the language of the text; P2 zero-shot with instructions in the other language; P3 few-shot (6 examples) in the language of the text. Same label order and schema everywhere. Few-shot examples are referenced by item ID and filled in at run time. | Defines "language-appropriate" for English data; keeps SemEval text out of released prompt files. |
| 8 | Models: Bielik-11B-v3.0-Instruct (official AWQ build) and PLLuM-12B-chat-2512 (community AWQ build; the proposal named the 2412 release, 2512 is the current one), both Apache 2.0, plus a size-matched 12B multilingual model in place of Gemma 4 E2B. Recommended: Mistral-Nemo-Instruct-2407 (community AWQ build), the base model of PLLuM-12B, so that pair differs only by Polish adaptation. Alternative: Gemma-3-12B-it. To confirm with the team. | Removes the size gap the review criticised. Gemma 4 has no 12B variant (E2B, E4B, 26B, 31B), which the rebuttal should state. |
| 9 | One engine and one quantization family for all main runs: vLLM 0.33.0 with 4-bit AWQ weights (W4A16), settled on 15 Nov after the PoC passed the server checks. The code talks to any OpenAI-compatible endpoint, so the engine is a config value. Ollama is excluded (see the findings below). | As the proposal promised; AWQ builds exist for all three models; vLLM gives prefix caching, continuous batching, per-request seeds and prompt log-probabilities for teacher forcing. |
| 10 | Second method class: TF-IDF with one-vs-rest logistic regression trained on the train split, used as a task baseline and as a confidence signal. Fine-tuned encoders are cited from the literature only. | Criterion 2.3 needs two method classes; the topic card excludes fine-tuning. |
| 11 | Negative controls: random routing, shuffled scores, and two trivial signals (number of predicted labels, text length). | If "accept every empty prediction" matches consistency, consistency adds nothing. |
| 12 | English SemEval is a secondary source with a reduced sample and is the first thing cut if compute runs short. | The review flagged the added compute. |

## Technical design

**Repository layout**

```
configs/      models.yaml, prompts.yaml, one YAML per experiment
prompts/      templates (PL, EN), label definitions (PL, EN), few-shot example IDs
data/         prepared CSVs; data/splits/*.csv holds the role of every item (IDs only); data/raw/ is git-ignored
src/routing/  data, splits, prompts, schema, client, run, signals, metrics, bootstrap, routing, cost, baseline_tfidf
scripts/      download_mipd, prepare_semeval, serve_model, measure_gpu
tests/        signals, metrics, splits, a fake engine for GPU-free end-to-end tests
notebooks/    EDA, PoC results, figures, demo
outputs/      runs/<run_id>/ (git-ignored), public/ (label outputs and tables, no text)
Dockerfile, environment.yml, Makefile, README.md, PLAN.md
```

**Splits (targets; the split script reports exact counts, whole articles only, seed 0)**

| Source | Few-shot pool | Pilot | Calibration | Test |
|---|---|---|---|---|
| SemEval PL | 10 train articles | about 200 train paragraphs | about 1,000 train paragraphs | dev, all 800 paragraphs |
| SemEval EN | 10 train articles | about 200 train paragraphs | about 1,000 train paragraphs | about 1,500 dev paragraphs |
| MIPD | train | 100 train articles | 600 validation articles | official test, 1,521 articles |

Leakage rule: a calibration item whose normalised text also occurs in test is dropped. Labels are reported individually when they have at least 30 gold positives in the test split; the rest are reported as one "rare labels" group.

**Run log.** One JSON line per generation: run ID, model, engine, quantization, model revision, prompt ID, item ID, source, split, kind (sample, greedy, verbal), seed, temperature, top-p, 0/1 per label, P(1) per label for greedy passes, stated confidence for verbal passes, validity flag, token counts, latency. No text. Each run folder also stores the config, git commit and package versions. This schema is agreed on day one so that metrics code can be written in parallel with the runner.

**Signals, all scoring the same majority-vote prediction**

| Signal | Per item and label | Per item |
|---|---|---|
| Self-consistency | share of the 5 runs agreeing with the vote | mean pairwise Jaccard, empty-vs-empty counted as 1 (primary, as declared in Phase 1); also modal-set frequency and mean pairwise Hamming distance |
| Token log-probability | P(voted decision) from the greedy pass, normalised over the two decision tokens | sum of the log-probabilities over labels |
| Verbalised confidence | stated probability for the voted decision | stated overall value |
| Classical | logistic-regression probability | product over labels |
| Controls | random, shuffled | number of predicted labels, text length |

Complete policies are also compared by cost: one greedy run with its own log-probability (1 generation), one run plus verbalised confidence (2), five-run vote with consistency (5).

**Evaluation**

- Task quality: micro-F1, macro-F1, exact-set accuracy, per-label F1, failure rate.
- Does the signal predict correctness: AUROC and error-detection PR-AUC, at item level (exact-set correctness) and at item-and-label level, per model, source, prompt and supported label; stratified by gold-empty and gold-non-empty. Accuracy alone is never reported.
- Uncertainty: paired bootstrap over articles, 2,000 resamples, 95% intervals, including intervals on differences between signals. Three independent five-run batches (15 samples) on the test sets for prompt P1, with the spread across batches.
- Calibration: every signal is mapped to a correctness probability on the same calibration items; expected calibration error on test.
- Ties and failures: AUROC uses mid-ranks; items tied at a threshold are accepted together; an item with any invalid output goes to the human and stays in the coverage denominator.

**Routing and cost (rule fixed before test results)**

- For each source, model and signal, choose on calibration data the threshold with the highest coverage whose accepted-item exact-set accuracy is at least 95% (also 90% and 80%), with at least 50 accepted items. If none exists, report the target as infeasible.
- On test: coverage, accepted accuracy with interval, quality after ideal human correction, hours saved, extra GPU time.
- The named operating point is the feasible setting with the largest net saving on calibration data. If no setting is feasible, the result is the negative finding, stated as such.
- Human time comes from a timed sample to collect by 9 Dec: 40 SemEval PL paragraphs, 40 SemEval EN paragraphs, 15 MIPD articles, each labelled independently by two of the three team members with the definitions the models see. Report median seconds per item and inter-annotator agreement.

**Error analysis.** At least 60 wrong predictions (20 per source), balanced between consistent-and-wrong and inconsistent-and-wrong, coded by two members into named types with counts. SemEval items appear by ID and paraphrase only.

**Compute estimate (from the PoC measurements of 16 Nov; MIPD still to be measured)**

Measured on the 12 GB card, all stages per item (five sampled runs, greedy run, scoring where needed, stated confidence):

| Run | Parallel requests | Seconds per paragraph |
|---|---|---|
| Bielik, SemEval PL calibration (1,002 paragraphs), P1 | 40 | 2.3 |
| Bielik, SemEval PL pilot, P3 (few-shot, 4,600-token prefix) | 32 | 3.4 |
| PLLuM, SemEval PL pilot, P1 / P2 | 20 | 2.8 / 2.4 |

| Quantity | Estimate per model |
|---|---|
| SemEval paragraphs still to run (PL calibration and test, EN pilot, calibration and test) | about 4,500 |
| SemEval, three prompts | about 8-9 s per paragraph, 10-12 GPU-hours |
| Two extra seed batches on the test sets, P1 only | about 2 GPU-hours |
| MIPD (2,421 articles, three prompts; prompts up to about 7,500 tokens for P1 and 11,000 for P3) | not measured; about 5 GPU-hours assumed |
| Total | about 18 GPU-hours, about 55 for three models |

An answer takes 262 output tokens (Bielik) or 272 (PLLuM) for SemEval, not the 200 the first plan assumed; the time per item absorbs this. The 12B models need GPU_UTIL 0.95 and a 6,144-token context to fit on the 12 GB card, so MIPD and most 12B runs belong on the 16 GB card. If the total goes above 80 GPU-hours, cut in this order: English sample size, replicates, shorter label keys in the output.

**Findings from the PoC (15-16 Nov)**

- Serving. vLLM 0.33.0 on the 12 GB card runs Bielik with 5.8 GiB of weights. With GPU_UTIL 0.90 the KV cache holds 19,200 tokens; 40 parallel requests give about 880 output tokens/s (16 requests: 520). The prefix-cache hit rate is 98%, because every prompt shares the instructions and the five runs of a paragraph are sent together. Two start-up errors and their fixes are in `scripts/serve_vllm.sh` (memory share, FlashInfer sampler). PLLuM needs GPU_UTIL 0.95 and a 6,144-token context to fit; with 28 parallel requests it gives 651 output tokens/s, and its calibration run (7,796 requests) took 43 minutes.
- Server checks pass for both models run so far: no invalid output in 26,072 requests, greedy decisions agree with their token probabilities (exact ties at 0.5 excepted), keys come in label order, and the share of sampled positives matches what the probabilities predict.
- Answer layout. Models write the JSON differently: Bielik compactly, PLLuM one key per line. Scoring and the stated-confidence turn must show the vote in the model's own layout; with Bielik's layout PLLuM's teacher-forced probabilities were off by up to 0.56. Fixed before any PLLuM result was used.
- Stated confidence. Asked per technique, Bielik's values could not be interpreted (constant over all techniques in half of the paragraphs, otherwise switching meaning), so it is asked once per answer. Bielik then gives values between 50 and 100. PLLuM answers 0 for every paragraph to the Polish question and almost only 0 or 100 to the English one, with or without the format constraint.
- Ollama's OpenAI-compatible endpoint is unsuitable: a fixed repetition penalty of 1.1 flips decisions away from the repeated "0", and it sorts the keys alphabetically. Reasoning-capable models (Gemma 4) must have reasoning switched off.
- First results (SemEval PL; pilot 219 paragraphs, calibration 1,002). Exact-set accuracy of the vote is 0.28-0.32 for both models, below the 0.45-0.46 of always answering "no technique". AUROC for exact-set correctness:

| | Bielik P1 pilot | Bielik P2 pilot | Bielik P3 pilot | Bielik P1 calibration | PLLuM P1 pilot | PLLuM P2 pilot | PLLuM P1 calibration |
|---|---|---|---|---|---|---|---|
| Jaccard consistency (primary) | 0.57 | 0.70 | 0.60 | 0.61 | 0.71 | 0.71 | 0.61 |
| Modal-set frequency | 0.68 | 0.74 | 0.65 | 0.73 | 0.84 | 0.80 | 0.78 |
| Token probability (vote) | 0.72 | 0.81 | 0.71 | 0.80 | 0.89 | 0.84 | 0.84 |
| Stated confidence | 0.69 | 0.76 | 0.54 | 0.69 | 0.50 | 0.50 | 0.50 |
| TF-IDF probability | 0.88 | 0.91 | 0.83 | 0.88 | 0.85 | 0.85 | 0.80 |
| Control: fewer predicted techniques | 0.89 | 0.88 | 0.82 | 0.93 | 0.86 | 0.83 | 0.87 |

- With Bielik, every model-based signal is below the trivial control: the vote is exactly right for 78% of paragraphs where it predicts no technique and for under 10% of the others. Fifteen runs instead of five do not change the Jaccard AUROC (0.569 both). With PLLuM, token probability and modal-set frequency reach or pass the control on the pilot, but not on its calibration run: token probability 0.84 (0.80-0.87) and modal-set frequency 0.78 stay below the control (0.87), although token probability beats TF-IDF (0.80). Jaccard consistency is 0.61 (0.56-0.66) for both models on calibration, and for PLLuM 0.35 on paragraphs that contain a technique, below chance. Of the 1,002 calibration paragraphs, all five PLLuM runs agree on 14.5%, and 71% of those are right.
- Routing (thresholds from Bielik's calibration run, applied to its pilot): no model-based signal accepts 50 calibration paragraphs at 90% exact-set accuracy; Jaccard consistency is infeasible even at 80%. Only TF-IDF is feasible at 90%, covering 12% of the pilot at 85% accepted accuracy. With PLLuM's thresholds applied to its pilot, token probability is feasible at 80% and accepts 15% of the pilot at 88% accepted accuracy (0.67-0.97); no signal is feasible at 90%, and Jaccard consistency at no target. Exact-set routing of whole paragraphs is therefore unlikely to work for either model; decision-level routing (accepting individual technique decisions) is the planned secondary analysis and should be reported next to it.

## Work plan

**Sun 15 - Wed 18 Nov: first number and the video**

| Day | Task | Owner |
|---|---|---|
| Sun | Reboot to fix the GPU driver; vLLM environment; Bielik AWQ download | M |
| Sun-Mon | Conda environment, package skeleton, split script, pilot subsets, fake engine | M with Claude |
| Mon | Polish and English prompt P1 with translated definitions; schema | T, reviewed by S |
| Mon | Runner against a local endpoint; Bielik-11B-v3.0 Q4 on 200 SemEval PL pilot paragraphs, 5 samples plus greedy | M with Claude |
| Mon-Tue | Signals and metrics module with unit tests: F1, Jaccard consistency, AUROC, log-probability baseline | S |
| Tue | First table and two figures; second model if time allows; slides and recording | all; T leads the video |
| Wed | Submit video; supervisor meeting: independence, third model, MIPD handling, in writing | all |

**Wed 18 - Wed 25 Nov: complete PoC**

- Engine test and freeze; Dockerfile, `environment.yml`, NVIDIA container toolkit; pinned model revisions (M).
- All three models, three prompts, pilot subsets of all three sources; measured tokens/s, peak VRAM, wall time (M, 16 GB owner for MIPD).
- Prompts P2 and P3; few-shot example selection by ID (T).
- EDA notebook: label and length distributions, duplicates, leakage between splits, MIPD credibility by technique, comparison across the three sources. Replaces the broken notebook 01 (T).
- TF-IDF baseline; bootstrap module; verbalised-confidence parsing (S).
- Watch the other teams' videos and prepare questions by Tue 24 Nov (all).

**Wed 25 Nov - Wed 2 Dec: Phase 2**

- Calibration-split runs for all models (M, 16 GB owner).
- Report in the final-report template (`latex/02_project-final-report-structure.tex` as the base; fix the placeholder author block): revised SOTA table, datasets with licences and per-label counts, EDA, method, PoC results, experiment table, measured runtime and memory, pre-registered analysis plan, ethics and data statement, contribution table, LLM-use declaration (now including Claude Code for planning and code). Rebuttal table in an appendix, with changes also made in the text (T leads, all write their sections).
- README that reproduces one number from a fresh clone: download MIPD, run the analysis on the released pilot outputs, print the AUROC. A second path runs the pilot on a GPU (M).
- Pull request to the course repo under `Projects/Phase 2/urban_trojan_nizwantowski/`, without datasets or raw outputs (M).
- Mon 30 Nov freeze, Tue 1 Dec proofreading against the self-check list, Wed 2 Dec submit.

**Wed 2 Dec - Wed 13 Jan: main study and final report**

| Week | Tasks | Owner |
|---|---|---|
| 2-9 Dec | Test-split runs: samples, greedy, verbalised. Output validation and reruns of failures. Timed annotation sample. | M, 16 GB owner; all for annotation |
| 9-16 Dec | Signals, AUROC and PR-AUC with intervals, per-label and stratified tables, seed-batch replicates, GPU measurements. Survey tutorial preparation. | S, M |
| 16-23 Dec | Routing curves, operating points, cost model. Error coding. Ablations: temperature, output format, 3 versus 5 versus 15 runs. | S, T |
| 23-30 Dec | Figures, results text, reproducibility appendix from `latex/03_reproducibility-checklist.tex`, fresh-clone reproduction test. Light week. | all |
| 30 Dec - 6 Jan | Full draft by 3 Jan; literature refresh for anything published since November; internal review; slides. | T leads |
| 6-13 Jan | Rebuttal to Phase 2 reviews and Q&A; final merge request; proofreading; presentation; questions for other teams. | all |

## Rebuttal matrix for Phase 2 (criterion 2.4)

Each row becomes one line of the rebuttal appendix with the section, table or file where the change is visible. Questions received on 25 Nov are appended.

| # | Review point | Answer | Concrete change |
|---|---|---|---|
| 1 | SOTA table rows without a split (Gilardi; Atıl; Pangakis and Wolken; Merdjanovska Yahoo/ESCI) | Fixed | Split added per number, checked in each paper |
| 2 | Tian row has no number | Fixed | ECE values for GPT-3.5-turbo on TriviaQA, verbalised versus token probability |
| 3 | APatt called an ensemble for Polish | Fixed after checking Sec. 4.1 | Row split: English ensemble, Polish single model |
| 4 | No row for the models we run | Fixed | Reported Polish-benchmark numbers from the Bielik and PLLuM reports, plus our pilot numbers |
| 5 | "No study checks..." claim too broad | Fixed | Narrowed to multilabel, Polish, open-weight models, three-signal comparison; cites Li 2023, Wang 2024, Gligorić 2025 |
| 6 | Six suggested works | Fixed | Added after verifying metadata: Xiong 2024, Farquhar 2024, Wang 2024, Gligorić 2025, Geng 2024, Piskorski 2025 |
| 7 | SemEval licence and access | Fixed | Official access by registration, terms quoted, access status stated, preparation script |
| 8 | Per-label balance only as a figure; "adequate support" undefined | Fixed | Per-label count table per split; threshold of 30 gold positives in test |
| 9 | Timed annotation sample unspecified | Fixed | 40 + 40 + 15 items, two annotators each, by 9 Dec |
| 10 | Split sizes and drawing rule missing | Fixed | Split table and rule |
| 11 | MIPD empty label sets are unannotated credible articles | Fixed | Credibility label added; primary results on disinformative articles; credible reported separately |
| 12 | Exact-set correctness inflated by empty sets | Fixed | Item-and-label AUROC, gold-empty and non-empty strata, trivial-signal control |
| 13 | Prompt languages for English data undefined | Fixed | Prompt table |
| 14 | Gemma 4 E2B far smaller than the Polish models | Fixed, with a correction | Size-matched 12B model; Gemma 4 has no 12B variant |
| 15 | No compute estimate | Fixed | Generations and GPU-hours from measured pilot throughput |
| 16 | Undefined abbreviations | Fixed | Each defined at first use |
| 17 | Models and vLLM not cited | Fixed | Bielik, PLLuM, third-model and vLLM references |
| 18 | Three language errors | Fixed | Sentences rewritten |
| 19 | Data statement omits model outputs | Fixed | Storage location, access, what is released |
| 20 | Agree a division of roles with the sibling team | Rejected with justification | Independent work confirmed by the supervisor (date); standard splits and released item lists keep results comparable; one paragraph on how the designs differ |
| 21 | Ethics paragraph; licence of each model | Fixed | New paragraph; licence column in the model table |
| 22 | Do not pool MIPD and SemEval | Fixed | Separate reporting everywhere |
| 23 | Label-set normalisation before voting; same calibration cases for all signals | Fixed | Fixed-order 0/1 vector, per-label majority; one calibration set for all signals |
| 24 | Operating point hedged with "if feasible"; pre-register the rule | Fixed | Rule stated; commitment to an operating point or the negative finding |
| 25 | Compute and hour-cost pilot | Fixed in part, rest deferred | Pilot compute in Phase 2; full cost analysis in Phase 3 |
| 26 | Meaningful negative control | Fixed | Random, shuffled and trivial signals |
| 27 | Question: how are the three signals compared on identical items | Answered | Same items, same voted prediction, same calibration set |
| 28 | Question: hours versus compute at the operating point, per task | Deferred to Phase 3 | Pilot estimate now; method stated |
| 29 | Supervisor: MIPD length and cross-dataset merging | Fixed | 2,000-word rule; no merging |
| 30 | English data adds compute | Answered | Secondary source, reduced sample, first to be cut |

## Rubric coverage

**Phase 2:** 2.1 pilot AUROC produced by the committed pipeline. 2.2 EDA notebook and section. 2.3 experiment table with two method classes, settings, splits, metrics. 2.4 matrix above plus Q&A rows. 2.5 extended SOTA table. 2.6 README path tested on a fresh clone. 2.7 measured time and VRAM. 2.8 contribution table matching commit history; each member commits their own work under their own account. 2.9 abbreviations and published versions.

**Phase 3:** 3.1 results tables across sources, models and method classes. 3.2 measurement procedures section. 3.3 seeds, versions, configs, commands, data version. 3.4 rebuttal. 3.5 presentation. 3.6 three seed batches and bootstrap intervals. 3.7 coded errors. 3.8 deviations: splits, third model, quantization, engine if changed. 3.9 literature refresh. 3.10 limitations with consequences: quantized models, gold-label noise, simulated human correction, small rare-label support. 3.11 data statement. 3.12 metric choice. 3.13-3.16 references, language, captions, code.

## Risks

| Risk | Response |
|---|---|
| vLLM does not serve all three models in 12 GB | Fallback engine, decided by 20 Nov |
| Models score near zero, or samples never disagree | Seen on the pilot before 25 Nov; revise prompts, check the temperature ablation; a null result is still reportable |
| Test results influence design | Test lock until the Phase 2 freeze |
| Holiday weeks | Main runs finished by 16 Dec; draft by 3 Jan |
| Contribution table does not match history | Own commits per member; hours noted weekly |

## Right after approval

1. Write this plan to `PLAN.md` in the repo and replace `TODO.md` with a short checklist pointing to it.
2. Build the PoC in this order: `environment.yml`; `src/routing/` skeleton and tests; `splits.py` writing `data/splits/*.csv`; MIPD credibility label in the preparation step; prompt P1 and schema; fake engine; runner; signals and metrics; first run on the SemEval PL pilot once the GPU is back after the reboot.

Existing code to reuse: the loading and validation logic in `notebooks/02_data_preparation.ipynb` moves into `scripts/`; the plotting cell in `notebooks/01_label_distribution.ipynb` moves into the EDA notebook; `data/sem_eval/task3_def.csv` supplies the English definitions.

## Verification

- Unit tests with known answers: Jaccard of two empty sets is 1, majority vote, AUROC with ties, bootstrap on a synthetic case, split disjointness by article.
- End-to-end run against the fake engine, no GPU, in under a minute.
- Pilot acceptance: zero invalid outputs with structured output, both decision tokens found for every label, tokens/s and peak VRAM recorded.
- Reproduction test: fresh clone on the teammate's machine, README only.
- Every number in the report is generated from run folders by a script; `latexmk` builds the PDF without warnings about undefined references.

## Open points for you

- Date and topic of the survey presentation and tutorial.
- Whether the final report is due before or after the 13 Jan Q&A, since its rebuttal must cover that session (ask the instructor).
- Confirm the third model: Mistral-Nemo-Instruct-2407 or Gemma-3-12B-it.
- Operating system of the 16 GB machine, for the setup instructions.

## Sources checked today

- [SemEval-2023 Task 3 site, access and terms](https://propaganda.math.unipd.it/semeval2023task3/)
- [Bielik-11B-v3.0-Instruct model card](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) and [SpeakLeash collection](https://huggingface.co/collections/speakleash/bielik-11b-v30)
- [PLLuM-12B-chat model card](https://huggingface.co/CYFRAGOVPL/PLLuM-12B-chat)
- [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [MIPD repository](https://github.com/ArkadiusDS/MIPD)
- [vLLM OpenAI-compatible server documentation](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [Course repository](https://github.com/grant-TraDA/NLP-2026W), including the sibling team's Phase 1 report

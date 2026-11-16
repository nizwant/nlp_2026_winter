# TODO

Pełny plan do końca projektu: [PLAN.md](PLAN.md). Tutaj tylko to, co na teraz.

## Do środy 18.11 (wideo mid-term)

- [x] dostęp do oficjalnych danych SemEval-2023 Task 3 (rejestracja zespołu)
- [ ] uruchomić `scripts/prepare_semeval.py` na oficjalnych plikach i potwierdzić, że wynik jest identyczny z `data/sem_eval/`
- [ ] pierwszy run Bielika (vLLM, AWQ) na pilocie SemEval PL
- [x] M - środowisko conda, szkielet `src/routing/`, splity, fake engine, runner
- [ ] T - sprawdzić polskie definicje technik i prompt P1 (`prompts/`), teraz to szkic
- [ ] S - przejąć i rozbudować `signals.py` / `metrics.py` (bootstrap, routing), testy
- [ ] wszyscy - slajdy i nagranie (T prowadzi)
- [ ] wszyscy - środa: pisemne potwierdzenie od S. Kuntur, że pracujemy niezależnie od drugiego zespołu; wybór trzeciego modelu

## Do środy 25.11 (Q&A)

- [ ] M - wybór silnika (vLLM + AWQ albo llama.cpp + Q4_K_M), Dockerfile, pomiar czasu i VRAM
- [ ] T - prompty P2 i P3, notebook EDA (zastępuje zepsuty `01_label_distribution.ipynb`)
- [ ] S - baseline TF-IDF + regresja logistyczna, pewność werbalna
- [ ] wszyscy - po 5 pytań do innych zespołów; zapisać dosłownie pytania zadane nam

## Do środy 2.12 (Phase 2)

- [ ] raport, rebuttal (30 punktów z recenzji, tabela w PLAN.md), README, pull request do repo kursu

## Notatki

- Miara spójności z sumy odległości Hamminga między parami runów jest w planie jako miara dodatkowa (`signals.hamming_agreement`); główną zostaje średni Jaccard po parach.
- Nie używać endpointu `/v1` Ollamy do prawdziwych runów: ma na stałe repeat penalty 1.1 (odwraca decyzje) i sortuje klucze alfabetycznie. Serwer llama.cpp z przypiętym samplerem przeszedł testy na CPU. Szczegóły w PLAN.md.
- Wersja GPU serwera llama.cpp (albo vLLM) jeszcze niezainstalowana; środowisko conda `llama-cpp` ma tylko wersję CPU.
- `data/sem_eval/task3_def.csv`: wiersz `Appeal_to_Authority` ma wklejoną definicję `Appeal_to_Hypocrisy`. Prompty używają poprawionej definicji z `prompts/`.
- Każdy commituje swoją pracę ze swojego konta (tabela wkładu musi zgadzać się z historią repo).

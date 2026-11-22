# TODO

Pełny plan do końca projektu: [PLAN.md](PLAN.md). Tutaj tylko to, co na teraz.

## Do środy 18.11 (wideo mid-term)

- [x] dostęp do oficjalnych danych SemEval-2023 Task 3 (rejestracja zespołu)
- [x] vLLM + Bielik (AWQ) na karcie 12 GB; pilot SemEval PL z P1 (3 batche), P2, P3; zbiór kalibracyjny P1
- [x] PLLuM-12B na pilocie SemEval PL (P1, P2); zbiór kalibracyjny i pilot Mistral-Nemo w toku
- [x] pierwsza analiza i routing kalibracja -> pilot; wykresy w `latex/img/poc_bielik_calibration_*`
- [ ] treść slajdów i notatki do nagrania (wyniki z PLAN.md, sekcja "Findings from the PoC")
- [ ] uruchomić `scripts/prepare_semeval.py` na oficjalnych plikach i potwierdzić, że wynik jest identyczny z `data/sem_eval/`
- [ ] sprawdzić polskie definicje technik i prompt P1 (`prompts/`), teraz to szkic
- [ ] potwierdzić trzeci model: Mistral-Nemo-Instruct-2407 (rekomendacja) czy Gemma-3-12B-it
- [ ] wszyscy - nagranie (10 min) i wrzucenie na kanał Teams
- [ ] wszyscy - środa: pisemne potwierdzenie od S. Kuntur, że pracujemy niezależnie od drugiego zespołu

## Do środy 25.11 (Q&A)

- [x] silnik: vLLM 0.33 + AWQ; Dockerfile; pomiar czasu, pamięci i poboru mocy w każdym runie
- [ ] piloty SemEval EN i MIPD (MIPD na karcie 16 GB), baseline TF-IDF dla EN i MIPD
- [x] prompty P2 i P3, notebook EDA (`notebooks/01_eda.ipynb`)
- [x] baseline TF-IDF + regresja logistyczna (SemEval PL), pewność werbalna
- [ ] wszyscy - po 5 pytań do innych zespołów; zapisać dosłownie pytania zadane nam

## Do środy 2.12 (Phase 2)

- [ ] raport, rebuttal (30 punktów z recenzji, tabela w PLAN.md), README, pull request do repo kursu

## Notatki

- Miara spójności z sumy odległości Hamminga między parami runów jest w planie jako miara dodatkowa (`signals.hamming_agreement`); główną zostaje średni Jaccard po parach.
- Nie używać endpointu `/v1` Ollamy do prawdziwych runów: ma na stałe repeat penalty 1.1 (odwraca decyzje) i sortuje klucze alfabetycznie. Używamy vLLM; szczegóły w PLAN.md.
- `data/sem_eval/task3_def.csv`: wiersz `Appeal_to_Authority` ma wklejoną definicję `Appeal_to_Hypocrisy`. Prompty używają poprawionej definicji z `prompts/`.
- Każdy commituje swoją pracę ze swojego konta (tabela wkładu musi zgadzać się z historią repo).

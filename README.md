# rag-eval: does a small RAG pipeline on EU sustainability law tell the truth?

> **Part of a series on RAG over EU sustainability law:** [rag-from-scratch](https://github.com/batutho-dot/rag-from-scratch) (retrieval built by hand) → **rag-eval** (measured: three scores and a Claude judge) → rag-claude-citations (not public yet; API-native: citations and caching)\
> Part 2: it scores the pipeline from rag-from-scratch.

**Question:** when a RAG pipeline answers questions about the EU Taxonomy, the CSRD and the ESRS, how often does it (1) find the passage with the answer, (2) answer right, and (3) say "not in the sources" when the texts cannot answer? And can a second model grade the answers as well as I can?

**Answer (50 questions, one run per config, 2026-11-19):**

| Config | Right passage found | Right answer given | Unanswerable declined | Spend |
|---|---|---|---|---|
| baseline (TF-IDF) | 24 / 36 (at 40 questions: 21 / 29) | 27 / 36 (24 / 29) | 13 / 14 (10 / 11) | 0.069 USD |
| **hybrid (kept)** | **27 / 36** (25 / 29) | **32 / 36** (28 / 29) | **14 / 14** (11 / 11) | 0.068 USD |

The Claude judge agrees with my hand grades on **22 of 24** baseline answers (92 %, Cohen's kappa 0.70). I keep hybrid search: it is better on all three scores at 30, 40 and 50 questions. The test set grew from 30 to 40 and then to 50 questions on the same day. Each time I kept the earlier answers and only answered and judged the new ones (`python run.py add <config>`), so the hand grades still match. The 40-question results are kept in the tables below for comparison.

## Terms

- **Three scores**, so a failure traces to its step:
  1. *Right passage found*: one of the 5 retrieved chunks holds an accepted evidence passage. If not, the retrieval failed.
  2. *Right answer given*: the answer has every required term and does not decline. A miss with the passage found is a generation failure.
  3. *Unanswerable declined*: the model replies `NOT IN SOURCES`.
- **Judge:** a stronger model grades each answer against the reference answer and source passage, through Anthropic's Message Batches API (within 24 hours, usually minutes, half price). It can be wrong, so I check it against hand grades.
- **Agreement:** the share of answers where judge and hand grade match, plus Cohen's kappa, which removes chance agreement (1 = perfect, 0 = no better than chance).

## Run it

Tested on Python 3.13 (Windows 11). From the repo folder:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py download        # the three EU texts from EUR-Lex into data/ (about 6 MB)
python run.py                 # print the scores from the saved runs: free, no key needed
```

New runs cost cents and need an Anthropic API key in `ANTHROPIC_API_KEY`:

```bash
python run.py generate baseline   # retrieve + answer with Claude Haiku 5.5
python run.py judge baseline      # grade with Claude Sonnet 5.5 through the Batches API
python run.py generate hybrid
python run.py judge hybrid
python run.py                     # the three scores, judge scores, agreement, spend
```

`python verify.py` checks the code in about 5 seconds without calling the API (`--full` adds the hybrid search, about 2 minutes). `verify-features/` lists what each check covers.

## The pipeline and the change under test

| Config | Search | Chunks to the model | Answers | Judge |
|---|---|---|---|---|
| `baseline` | TF-IDF, written in numpy (project 2's version A) | top 5 of 1,257 (200 words, 50 overlap) | `claude-haiku-5-5`, effort low | `claude-sonnet-5-5`, batch |
| `hybrid` | TF-IDF and `BAAI/bge-small-en-v1.5` merged by reciprocal rank fusion | same | same | same |

Hybrid search is the one change tested. Project 2 found the two methods miss different questions, so merging their rankings should find more passages. I keep it only if it loses on none of the three scores.

## How it is built

The flow chart (generate, judge, report) and every box with its `file:line` are in [docs/architecture.md](docs/architecture.md).

## The test set

`testset.json`, 50 questions.

- **36 answerable.** q01 to q10 from project 2, q11 to q20 new, q21 to q29 added later the same day (Taxonomy Articles 3, 10(2) and 20; CSRD Article 19a(3) and (7); ESRS 1 paragraphs 45 and 77(a) and Appendix C; ESRS E1-7), q30 to q36 added after that (Taxonomy Articles 26(2) and 27(2); CSRD Articles 5(1) and 19a(2)(b); ESRS S1-6, E1-5 footnote 42 and S1-14 AR 89). Each has a reference answer, required terms and one or more accepted evidence passages, checked to exist word for word in the downloaded text. q07, q12 and q15 accept two passages, which fixes project 2's q07 problem.
- **14 unanswerable.** u01 to u10, plus u11 (an internal carbon price level, which ESRS E1-8 asks companies to report but never sets), plus u12 to u14 (how many members the Platform on Sustainable Finance has; a minimum number of training hours under ESRS S1; a gender pay gap above which a company must act under ESRS S1). None of the three texts gives these values. Several are near misses on purpose: u02 (the ESRS says to use IPCC global warming potentials but gives no value for methane) and u04 (a disclosure requirement on adequate wages but no wage level).

Claude drafted all questions and I approved them. q21 to q29 and u11 were checked before I approved: quotes found word for word, each answer follows from its quote, each article matches the heading above the quote, and I read ESRS E1-8 (paragraphs 62 and 63, AR 65 and 66) in full. All ten passed. q30 to q36 and u12 to u14 were drafted by a Claude Opus worker and checked the same way: every quote found word for word in the downloaded text, every answer containing its required terms, and keyword searches for the unanswerable ones. I approved each of the ten on its own.

## Results

### Right passage found (measured, no model involved, k = 5)

`python run.py retrieval` recomputes this.

| Config | First 30 questions | 40 questions | All 50 questions | Missed (all 50) |
|---|---|---|---|---|
| baseline (TF-IDF) | 15 / 20 | 21 / 29 | **24 / 36** | q01, q04, q10, q14, q17, q21, q23, q27, q31, q32, q34, q35 |
| hybrid | 17 / 20 | 25 / 29 | **27 / 36** | q04, q14, q17, q23, q31, q32, q33, q34, q35 |

Hybrid recovers q21 and q27, which TF-IDF misses. Of the 7 newest answerable questions, both configs miss 4 (q31, q32, q34, q35), and hybrid also misses q33, which TF-IDF finds. Neither finds q23 (which expert body the Taxonomy Regulation sets up: the Platform on Sustainable Finance).

### Answers and refusals (scored by the script)

| Config | Right answer given | Unanswerable declined | Answerable falsely declined |
|---|---|---|---|
| baseline, first 30 | 17 / 20 | 9 / 10 | 2 / 20 (q01, q10) |
| hybrid, first 30 | 20 / 20 | 10 / 10 | 0 / 20 |
| baseline, all 40 | 24 / 29 | 10 / 11 | 3 / 29 (q01, q10, q27) |
| hybrid, all 40 | 28 / 29 | 11 / 11 | 0 / 29 |
| baseline, all 50 | 27 / 36 | 13 / 14 | 7 / 36 (q01, q10, q27, q31, q32, q34, q35) |
| hybrid, all 50 | **32 / 36** | **14 / 14** | 3 / 36 (q31, q32, q35) |

Baseline's false declines are retrieval misses: the model never saw the answer and said so, which is the wanted behaviour. Hybrid found those passages and answered right. Its one unanswerable slip (u09) also went away. Both configs get q23 wrong. On the 10 newest questions both configs decline q31, q32 and q35, whose passages neither search found. All three new unanswerable questions were declined by both.

### The judge, checked against hand grades

`claude-sonnet-5-5` graded every answer through the Batches API. Hand grades exist for 24 baseline answers: all 20 answerable and u01, u02, u04, u09.

| Judge run | Answers right | Unanswerable declined | Agrees with hand grades | Cost (batch) |
|---|---|---|---|---|
| baseline, first 30 | 17 / 20 | 9 / 10 | **22 / 24 (92 %), kappa 0.70** | 0.032 USD |
| hybrid, first 30 | 20 / 20 | 10 / 10 | no hand grades | 0.031 USD |
| baseline, all 40 | 23 / 29 | 10 / 11 | (same 24 hand grades) | 0.044 USD |
| hybrid, all 40 | 28 / 29 | 11 / 11 | no hand grades | 0.043 USD |
| baseline, all 50 | 26 / 36 | 13 / 14 | (same 24 hand grades) | 0.054 USD |
| hybrid, all 50 | 31 / 36 | 14 / 14 | no hand grades | 0.053 USD |

On the 10 newest questions, judge and script disagree once: hybrid q33. Asked for which years absolute GHG reduction targets must at least be set, it answered "2030, and for 2050 if available", from ESRS text instead of the CSRD passage it did not retrieve. The CSRD says at least for 2030 and 2050. The script finds both years and passes it; the judge fails it, and the judge is right.

On the 10 questions before those, judge and script also disagree once: baseline q21. The answer lists three of the four Article 3 conditions and only guesses at the first ("presumably"). The script finds all four terms in the hedged text, the judge marks it incorrect, and here the judge reads better. No hand grades exist for the new questions.

Hand grades: a Claude Opus model drafted a grade and reason for each of the 24 answers, without seeing the judge's verdicts. I decided the three close calls (q04, q07, u09) and accepted the rest. They are in `hand_grades.json`.

### Spend

| Run | Tokens in / out | USD |
|---|---|---|
| baseline answers (Haiku 5.5) | 67,465 / 4,260 | 0.0089 |
| baseline judge (Sonnet 5.5, batch) | 20,268 / 2,353 | 0.0320 |
| hybrid answers (Haiku 5.5) | 68,047 / 3,751 | 0.0087 |
| hybrid judge (Sonnet 5.5, batch) | 20,260 / 2,128 | 0.0309 |
| 10 new questions, baseline answers | 22,226 / 2,266 | 0.0034 |
| 10 new questions, baseline judge (batch) | 7,153 / 868 | 0.0115 |
| 10 new questions, hybrid answers | 22,140 / 2,184 | 0.0033 |
| 10 new questions, hybrid judge (batch) | 7,256 / 885 | 0.0117 |
| 10 newest questions, baseline answers | 23,084 / 1,284 | 0.0029 |
| 10 newest questions, baseline judge (batch) | 6,829 / 704 | 0.0103 |
| 10 newest questions, hybrid answers | 22,973 / 1,024 | 0.0028 |
| 10 newest questions, hybrid judge (batch) | 6,829 / 704 | 0.0103 |
| **total** | | **0.1368** |

Prices from Anthropic's pricing page on 2026-11-19. `python run.py` recomputes them from the saved token counts.

## What failed

- **q23, both configs.** Asked which expert body the Commission sets up under the Taxonomy Regulation, both name the Member State Expert Group on Sustainable Finance instead of the Platform on Sustainable Finance (Article 20). Neither search retrieved Article 20. Both answers add that the passages do not say who set it up. It stays as a recorded failure, though "sets up" may be loose wording.
- **Judge vs hand grades, twice.** q04: too lenient. The answer half-declines and never states the two kinds of information, but the judge counted it as the reference answer in other words. q18: wrong on a fact. The answer said "more than 5 years" (right) and cited ESRS 1 "paragraph 78"; the judge called that "likely inaccurate". It is accurate. The judge saw only one evidence passage. Whether more source would fix this, I did not test.
- **"Right passage found" undercounts.** q14 and q17 are retrieval misses for both configs, yet both answered right from other passages stating the same fact. The test accepts only the listed passages, as with project 2's q07. I left it so the two runs stay comparable. The same holds for hybrid q34 (the country breakdown of headcount).
- **Four of the 7 newest answerable questions are never retrieved** (q31, q32, q34, q35, both configs). q35 sits in an ESRS footnote, which the text keeps far from its paragraph. Most of these end as honest refusals, not wrong answers. These misses are not yet classified in the error analysis, which covers the first 40 questions.
- **u09, baseline.** Asked to which year a later amendment postponed reporting for large non-listed companies, it gave 2025, the original date in the 2022 CSRD. It noted the passages do not call it a postponement but still gave a year, so it counts as not declined. Hybrid declined.
- **q04 is the hardest question.** Hybrid also misses its passage, but the script scores its answer right, from neighbouring CSRD text (not hand-graded).

Every miss, classified: [docs/error-analysis.md](docs/error-analysis.md). Every real failure starts at retrieval: baseline 5 retrieval misses, 1 refusal error (u09), 1 judge error (q18); hybrid 3 retrieval misses, all still answered right. No chunking splits and no wrong answers with the right passage in either run.

## Limits

- 50 questions. One answerable question is about 2.8 percentage points, one unanswerable about 7.
- I knew the texts when I wrote the questions, so they reuse the law's wording, which favours TF-IDF.
- The answer-term check is strict about wording, and I checked the unanswerable questions by keyword search, not by reading every line.
- Texts as first published in the Official Journal, without later amendments.
- Answers vary between runs, so a one-question difference between configs is within noise. I reran hybrid generation once (`runs/hybrid_rerun_2026-11-19.json`, script scores only, no judge). Retrieval was identical and all three scores matched (17/20, 20/20, 10/10): spread 0 on 2 runs. Output tokens moved from 3,751 to 3,855, cost stayed at 0.0087 USD. Two runs cannot rule out a one-question swing on a third.
- Hand grades cover the baseline run only, and a model drafted them before my review.

## Source and credit

EU legal texts: © European Union, https://eur-lex.europa.eu. Unless otherwise specified, EUR-Lex allows re-use for commercial and non-commercial purposes under the Commission's reuse policy (Decision 2011/833/EU): https://eur-lex.europa.eu/content/legal-notice/legal-notice.html. `python run.py download` fetches the texts; they are not stored in this repo. Only the Official Journal texts are authentic. This tool is not legal advice.

Search code in `pipeline/` is copied from rag-from-scratch (MIT, same author). Embedding model: `BAAI/bge-small-en-v1.5` by BAAI, run through [fastembed](https://github.com/qdrant/fastembed). Claude model ids and prices: https://platform.claude.com/docs/en/about-claude/pricing (read 2026-11-19).

Code: MIT licence, see `LICENSE`.

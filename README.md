# rag-eval: does a small RAG pipeline on EU sustainability law tell the truth?

**Question:** when a retrieval-augmented pipeline answers questions about the EU Taxonomy,
the CSRD and the ESRS, how often does it (1) find the passage that holds the answer, (2) give
the right answer, and (3) say "not in the sources" when the texts cannot answer? And can a
second model, used as a judge, grade the answers as well as a person does?

This repo evaluates the pipeline from
[rag-from-scratch](https://github.com/batutho-dot/rag-from-scratch) (same author) and tests
one change to it.

**Status:** code, test set and checks are done. The paid runs (Claude answers and the
Claude judge) have not been made yet, so there are no answer or judge scores below yet. The
retrieval score needs no model and is measured.

## The ideas, in plain words

- **RAG** (retrieval-augmented generation): search the documents for the few passages most
  like the question, then let a language model answer from only those passages.
- **Three scores**, so a failure can be traced to its step:
  1. *Right passage found*: one of the 5 retrieved chunks contains an accepted evidence
     passage. If not, the model never saw the answer: a retrieval failure.
  2. *Right answer given*: the answer contains every required term from the reference
     answer and does not decline. A miss here with the passage found is a generation failure.
  3. *Unanswerable declined*: for questions the texts cannot answer, the model replies
     `NOT IN SOURCES` instead of inventing something.
- **Judge:** a second, stronger model grades each answer against the reference answer and
  the source passage, like a teacher marking against an answer key. Keyword checks miss right
  answers in other words; a judge can catch those, but it can also be wrong, so it is checked
  against hand grades.
- **Batch:** the judge's requests go to Anthropic's Message Batches API in one go. They are
  processed within 24 hours (usually minutes) at half the normal price.
- **Agreement:** the share of answers where the judge and the hand grade say the same thing,
  plus Cohen's kappa, which removes the agreement two graders would reach by chance
  (1 = perfect, 0 = no better than chance).

## Run it

Tested on Python 3.13 (Windows 11). From the repo folder:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py download        # the three EU texts from EUR-Lex into data/ (about 6 MB)
python run.py                 # print the scores from the saved runs: free, no key needed
```

To make new runs (costs cents, needs an Anthropic API key in the environment variable
`ANTHROPIC_API_KEY`):

```bash
python run.py generate baseline   # retrieve + answer with Claude Haiku 5.5
python run.py judge baseline      # grade with Claude Sonnet 5.5 through the Batches API
python run.py generate hybrid
python run.py judge hybrid
python run.py                     # the three scores, judge scores, agreement, spend
```

`python verify.py` checks the code after a change in about 5 seconds without calling the
API (`--full` adds the hybrid search, about 2 minutes). What each check covers is in
`verify-features/`.

## The pipeline and the change under test

| Config | Search | Chunks to the model | Answers | Judge |
|---|---|---|---|---|
| `baseline` | TF-IDF, written in numpy (project 2's version A) | top 5 of 1,257 (200 words, 50 overlap) | `claude-haiku-5-5`, effort low | `claude-sonnet-5-5`, batch |
| `hybrid` | TF-IDF and `BAAI/bge-small-en-v1.5` merged by reciprocal rank fusion | same | same | same |

Hybrid search is the one change tested. Project 2 found the two search methods miss
different questions, so merging their rankings should find more passages. It is kept only if
it does not lose on any of the three scores.

## The test set

`testset.json`, 30 questions:

- **20 answerable**: q01 to q10 from project 2, q11 to q20 new. Each has a reference answer,
  required answer terms, one or more accepted evidence passages (exact words of the law,
  checked to exist in the downloaded text) and where they sit (regulation, article).
  q07 and q12 and q15 accept two passages, which fixes project 2's q07 problem (the recital
  answered it but only the article was accepted).
- **10 unanswerable**: questions the three texts cannot answer. Several are near misses on
  purpose: the topic is in the law but the figure is not (u02: the ESRS says to use IPCC
  global warming potentials but gives no value for methane; u04: there is a disclosure
  requirement on adequate wages but no wage level). Each carries the reason it cannot be
  answered.

All questions were drafted with Claude from the texts and approved by the author.

## Results

### Right passage found (measured, no model involved, k = 5)

| Config | Passage found | Missed |
|---|---|---|
| baseline (TF-IDF) | **15 / 20** | q01, q04, q10, q14, q17 |
| hybrid | **17 / 20** | q04, q14, q17 |

### Answers, refusals, judge and agreement

Not run yet: pending the first paid run.

## What failed

To be filled from the runs.

## Limits

- 30 questions. One answerable question is 5 percentage points; one unanswerable question
  is 10.
- The question author knew the texts, so questions tend to reuse the law's wording, which
  favours TF-IDF search.
- The answer-term check is strict about wording; the judge exists to catch what it misses.
- Unanswerable questions were checked against the texts by keyword search, not by reading
  every line.
- Texts as first published in the Official Journal; later amendments are not included.
- Language model answers vary from run to run; each config is run once.

## Source and credit

EU legal texts: © European Union, https://eur-lex.europa.eu. EUR-Lex's legal notice states
that, unless otherwise specified, its legal documents may be re-used for commercial or
non-commercial purposes, under the Commission's reuse policy (Decision 2011/833/EU):
https://eur-lex.europa.eu/content/legal-notice/legal-notice.html. The texts are downloaded
by `python run.py download` and are not stored in this repo. Only the Official Journal
texts are authentic; this tool is not legal advice.

Search code in `pipeline/` is copied from rag-from-scratch (MIT, same author).
Embedding model: `BAAI/bge-small-en-v1.5` by BAAI, run through
[fastembed](https://github.com/qdrant/fastembed).
Claude model ids and prices: https://platform.claude.com/docs/en/about-claude/pricing
(read 2026-11-19).

Code: MIT licence, see `LICENSE`.

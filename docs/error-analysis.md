# Error analysis of the saved runs

Every miss in `runs/baseline.json` and `runs/hybrid.json` (both 2026-11-19), with the judge verdicts from `runs/*_judge.json` and the hand grades from `hand_grades.json`. Nothing comes from a new run; each line can be checked against those files and the chunk ids.

A question is listed when any score marks it wrong: passage not in the top 5, the script's answer check fails, the judge or the hand grade says incorrect, or an unanswerable question is not declined.

## The classes

| Class | Meaning |
|---|---|
| retrieval miss | No accepted evidence passage is in the 5 retrieved chunks. |
| chunking split | The evidence exists but is cut across two chunks, so no single chunk holds it. |
| answer wrong despite right passage | The passage was retrieved and the answer is still wrong. |
| refusal error | An answerable question declined with the passage present, or an unanswerable question answered. |
| judge error | Right by the script and the hand grade, but the judge disagrees. Not a pipeline miss. |

## Baseline (TF-IDF): 7 questions

| Id | Class | What happened |
|---|---|---|
| q01 | retrieval miss | The evidence (Article 9 of the Taxonomy) is only in `taxonomy#0058`, not retrieved. The top chunk `taxonomy#0101` mentions "the environmental objectives referred to in ... Article 9" without listing them, so the model correctly answered `NOT IN SOURCES`. |
| q04 | retrieval miss | The evidence (new Article 19a(1)) sits in `csrd#0118` and `csrd#0119`, neither retrieved. The answer half-declined and described materiality perspectives from ESRS chunks: incorrect by hand, correct by the judge. |
| q10 | retrieval miss | The E1-6 heading sits in `esrs#0184`, `esrs#0185` and `esrs#0206`, none retrieved. None of the 5 retrieved chunks contains "E1-6" (two, `esrs#0255` and `esrs#0256`, are application requirements on calculating Scope 1 emissions), so the model declined. |
| q14 | retrieval miss | The accepted passage (`csrd#0166`, amended Article 34) was not retrieved, but the answer is right from recital text in `csrd#0085`, which also says the auditor's opinion is "based on a limited assurance engagement". |
| q17 | retrieval miss | The accepted passage (`csrd#0151`, new Article 29b) was not retrieved, but the answer is right from `csrd#0054` and `esrs#0001`, which both name EFRAG's technical advice. |
| q18 | judge error | Passage found, answer "more than 5 years" right by script and hand grade. The judge marked it incorrect for citing "paragraph 78", which it could not see in its single evidence passage. |
| u09 | refusal error | Unanswerable (the corpus has no later amendment), yet the model gave 2025, the original date in the 2022 CSRD, while noting the passages do not call it a postponement. |

## Hybrid (TF-IDF + bge-small, reciprocal rank fusion): 3 questions

| Id | Class | What happened |
|---|---|---|
| q04 | retrieval miss | Article 19a(1) (`csrd#0118`, `csrd#0119`) still not retrieved. The answer is right by script and judge (not hand-graded) from recital `csrd#0038`, which states both materiality perspectives in nearly the reference's words. |
| q14 | retrieval miss | As baseline: `csrd#0166` not retrieved, answer right from `csrd#0085`. |
| q17 | retrieval miss | As baseline: `csrd#0151` not retrieved, answer right from `csrd#0054` and neighbouring CSRD chunks. |

## Counts

| Class | Baseline | Hybrid |
|---|---|---|
| retrieval miss | 5 (q01, q04, q10, q14, q17) | 3 (q04, q14, q17) |
| of which the answer was still right | 2 (q14, q17) | 3 (q04, q14, q17) |
| chunking split | 0 | 0 |
| answer wrong despite right passage | 0 | 0 |
| refusal error | 1 (u09) | 0 |
| judge error | 1 (q18) | 0 |

## What it says

- **Every real failure starts at retrieval.** No answer is wrong with the right passage in front of the model, so the next fix is in the search, not the prompt.
- **No chunking splits, by design.** Every evidence passage is at most 50 words, the chunk overlap, so each lies whole in at least one chunk. `verify.py` checks this on the real corpus. Longer evidence would need the check again.
- **Half the retrieval misses are the test's.** q14, q17 and hybrid's q04 were answered right from a recital or neighbouring paragraph stating the same fact. Listing those as accepted would raise "right passage found". I left it so both configs are scored on the same key.
- **A miss mostly ends in a decline, as wanted.** Baseline q01 and q10 declined rather than guessing. The exceptions are q04 (half-declined, then answered from nearby material) and u09 (answered a question the texts cannot answer).

# Retrieval: baseline (TF-IDF) and hybrid (TF-IDF + bge-small, reciprocal rank fusion)

- **Does:** returns the 5 chunks most similar to the question.
- **Reach:** `pipeline/version_a.py`, `pipeline/hybrid.py`; used by `python run.py generate <config>`.
- **Verify drives it:** `units` checks the fusion arithmetic on two fixed fake rankings.
  `real-run-tfidf` runs TF-IDF on the real corpus (1,257 chunks) and compares the missed
  questions with `verify_known_good.json`; `real-run-hybrid` (`--full`) does the same for hybrid.
- **End state:** TF-IDF misses q01, q04, q10, q14, q17; hybrid misses q04, q14, q17.
- **Gotchas:** hybrid downloads a 65 MB model on first use and takes about 2 minutes on a
  laptop CPU, so it is behind `--full`. A deliberate change to search updates
  `verify_known_good.json` in the same commit.

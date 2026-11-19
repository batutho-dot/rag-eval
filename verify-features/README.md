# Feature map

What this repo does for its user, and how `verify.py` drives each part. `verify.py` never
calls the Claude API, so the two paid features are checked up to the request they send.

| Feature | File | Verify step |
|---|---|---|
| The test set | [testset.md](testset.md) | `testset`, `real-run-tfidf` |
| Retrieval, baseline and hybrid | [retrieval.md](retrieval.md) | `units`, `real-run-tfidf`, `real-run-hybrid` (`--full`) |
| Answer generation with Claude | [generate.md](generate.md) | `units` (prompt layout, prices) |
| Claude as judge, Batches API | [judge.md](judge.md) | `units` (request shape) |
| The report: three scores and agreement | [report.md](report.md) | `units`, `report` |

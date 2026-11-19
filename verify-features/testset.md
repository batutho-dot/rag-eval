# The test set

- **Does:** fixes the 30 questions every run is scored on: 20 with a reference answer,
  required answer terms and accepted evidence passages, 10 the texts cannot answer.
- **Reach:** `testset.json`.
- **Verify drives it:** `testset` checks counts (20 / 10), unique ids, required fields and
  that each evidence passage is at most 50 words (so it fits inside the chunk overlap).
  `real-run-tfidf` checks every evidence passage exists in the downloaded corpus.
- **End state:** both steps ok.
- **Gotchas:** changing a question changes every score; record it in the README. The
  unanswerable questions were checked by keyword search only, not by reading every line.

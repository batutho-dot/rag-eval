# Answer generation with Claude Haiku 5.5

- **Does:** sends the question and the 5 chunks to `claude-haiku-5-5` and stores the answer
  or `NOT IN SOURCES`, with tokens and cost per call, in `runs/<config>.json`.
- **Reach:** `python run.py generate baseline` (needs `ANTHROPIC_API_KEY`, from the
  environment or a git-ignored `.env`).
- **Verify drives it:** `units` checks the prompt layout and the price arithmetic. No API call.
- **End state:** not checked by verify; the saved run file is the evidence.
- **Gotchas:** costs money (a few cents per run). Answers vary between runs: the model is not
  deterministic, so a rerun can move the scores by a question or two.

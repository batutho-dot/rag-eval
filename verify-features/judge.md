# Claude as judge, through the Message Batches API

- **Does:** asks `claude-sonnet-5-5` to grade each saved answer against the reference answer
  and source passage (or, for unanswerable questions, whether it declined), returning JSON
  `{verdict, reason}`. All requests go in one batch at half price.
- **Reach:** `python run.py judge baseline` after `generate`.
- **Verify drives it:** `units` builds the batch requests for two fixed rows and checks the
  ids, the JSON schema and that the prompt states the question type. No API call.
- **End state:** not checked by verify; `runs/<config>_judge.json` is the evidence.
- **Gotchas:** a batch can take minutes; the script polls every 20 seconds. Results arrive
  in any order and are keyed by question id.

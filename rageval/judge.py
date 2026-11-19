"""Step 2 of a run: Claude as judge, through the Message Batches API.

A judge is a second model that grades each answer against the reference answer and the
source passage, the way a teacher marks against an answer key. It catches answers the
keyword check gets wrong (right fact in other words, or the right keyword in a wrong
answer). A batch sends all grading requests at once; Anthropic processes them within
24 hours (usually minutes) at half the normal price.

Writes runs/<config>_judge.json: one verdict per question, tokens, cost, batch id.
"""

import json
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from rageval.config import DECLINE, JUDGE, RUNS, TESTSET, cost_usd, require_key

RUBRIC = (
    "You grade one answer from a question-answering system over EU sustainability law "
    "(EU Taxonomy Regulation, CSRD, ESRS).\n"
    "If the question is ANSWERABLE: the answer is correct only if it states the substance of "
    "the reference answer and says nothing that contradicts the source passage. Extra detail "
    "that is true is fine. Wording does not matter. Declining to answer is incorrect.\n"
    "If the question is UNANSWERABLE (the documents do not contain the answer): the answer is "
    f"correct only if it declines, for example with '{DECLINE}' or by saying the documents do "
    "not give it. Any attempt to give the requested figure or fact is incorrect.\n"
    "Return the verdict and a one-sentence reason."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def grading_prompt(q: dict, answer: str) -> str:
    if q["type"] == "answerable":
        key = (f"Type: ANSWERABLE\nReference answer: {q['answer']}\n"
               f"Source passage ({q['where']}): {' / '.join(q['evidence'])}")
    else:
        key = f"Type: UNANSWERABLE\nWhy the documents cannot answer it: {q['why']}"
    return f"Question: {q['question']}\n{key}\n\nAnswer to grade:\n{answer}"


def build_requests(run: dict, questions: dict) -> list:
    return [
        Request(
            custom_id=row["id"],
            params=MessageCreateParamsNonStreaming(
                model=JUDGE,
                max_tokens=4000,
                system=RUBRIC,
                output_config={"effort": "medium",
                               "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": grading_prompt(questions[row["id"]], row["answer"])}],
            ),
        )
        for row in run["rows"]
    ]


def main(config: str, poll_seconds: int = 20) -> None:
    require_key()
    run = json.loads((RUNS / f"{config}.json").read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in json.loads(TESTSET.read_text(encoding="utf-8"))}
    client = anthropic.Anthropic()

    requests = build_requests(run, questions)
    batch = client.messages.batches.create(requests=requests)
    print(f"judge batch {batch.id}: {len(requests)} requests sent to {JUDGE}")
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  {batch.processing_status}, {batch.request_counts.processing} still processing")
        time.sleep(poll_seconds)

    verdicts, tin, tout = {}, 0, 0
    for res in client.messages.batches.results(batch.id):  # results come in any order: key by id
        if res.result.type != "succeeded":
            verdicts[res.custom_id] = {"verdict": None, "reason": f"batch result {res.result.type}"}
            continue
        msg = res.result.message
        tin += msg.usage.input_tokens
        tout += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            verdicts[res.custom_id] = {"verdict": None, "reason": "judge refused"}
            continue
        text = next(b.text for b in msg.content if b.type == "text")
        verdicts[res.custom_id] = json.loads(text)

    out = {
        "config": config,
        "judge_model": JUDGE,
        "batch_id": batch.id,
        "input_tokens": tin,
        "output_tokens": tout,
        "cost_usd": round(cost_usd(JUDGE, tin, tout, batch=True), 6),
        "verdicts": dict(sorted(verdicts.items())),
    }
    (RUNS / f"{config}_judge.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = sum(v["verdict"] is None for v in verdicts.values())
    print(f"judge {config}: {len(verdicts)} verdicts ({failed} failed), {tin:,} in / {tout:,} out tokens, "
          f"{out['cost_usd']:.4f} USD (batch price) -> runs/{config}_judge.json")

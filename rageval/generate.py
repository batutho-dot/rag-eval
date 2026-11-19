"""Step 1 of a run: for every test question, find the top-k chunks and ask Claude to answer
from those chunks only, or to say NOT IN SOURCES.

Writes runs/<config>.json with the retrieved chunk ids, the answer, the tokens and the cost
of every call. This is the step that spends money (Claude Haiku 5.5, standard API).
"""

import json
import time
from datetime import datetime, timezone

import anthropic

from pipeline.chunk import load_chunks
from rageval.config import CONFIGS, DATA_TEXT, DECLINE, GENERATOR, RUNS, TESTSET, cost_usd, require_key, MAX_RUN_USD

SYSTEM = (
    "You answer questions about EU sustainability law using only the numbered passages in "
    "the user message. Do not use any other knowledge. If the passages do not contain the "
    f"answer, reply with exactly: {DECLINE}\n"
    "Otherwise answer in one to three sentences and name the passage numbers you used in "
    "square brackets, for example [2]."
)


def build_index(search: str, texts: list[str]):
    if search == "tfidf":
        from pipeline.version_a import TfidfIndex

        return TfidfIndex(texts)
    if search == "hybrid":
        from pipeline.hybrid import HybridIndex

        return HybridIndex(texts)
    raise ValueError(search)


def prompt(question: str, passages: list[str]) -> str:
    numbered = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(passages, start=1))
    return f"Passages:\n\n{numbered}\n\nQuestion: {question}"


def answer(client: anthropic.Anthropic, question: str, passages: list[str]) -> dict:
    t = time.perf_counter()
    msg = client.messages.create(
        model=GENERATOR,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt(question, passages)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if msg.stop_reason == "refusal":
        text = f"[refused by the model: {getattr(msg.stop_details, 'category', None)}]"
    u = msg.usage
    return {
        "answer": text,
        "stop_reason": msg.stop_reason,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost_usd": cost_usd(GENERATOR, u.input_tokens, u.output_tokens),
        "seconds": round(time.perf_counter() - t, 2),
    }


def main(config: str) -> None:
    require_key()
    cfg = CONFIGS[config]
    chunks = load_chunks(DATA_TEXT)
    questions = json.loads(TESTSET.read_text(encoding="utf-8"))
    index = build_index(cfg["search"], [c.text for c in chunks])
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    rows = []
    for q in questions:
        hits = index.search(q["question"], cfg["k"])
        passages = [chunks[i].text for i, _ in hits]
        row = {"id": q["id"], "retrieved": [chunks[i].id for i, _ in hits]}
        row.update(answer(client, q["question"], passages))
        rows.append(row)
        if sum(r["cost_usd"] for r in rows) > MAX_RUN_USD:
            raise SystemExit(f"spend passed {MAX_RUN_USD} USD after {len(rows)} questions; stopped, nothing saved")
        print(f"  {q['id']}  {row['output_tokens']:>5} out  {row['answer'][:90]!r}")

    run = {
        "config": config,
        **cfg,
        "model": GENERATOR,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "rows": rows,
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{config}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{config}: {len(rows)} answers, {run['input_tokens']:,} in / {run['output_tokens']:,} out tokens, "
          f"{run['cost_usd']:.4f} USD -> runs/{config}.json")

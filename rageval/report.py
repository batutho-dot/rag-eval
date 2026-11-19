"""Step 3: print the scores from the saved runs. Spends nothing; needs no API key.

Three scores per run, each scored by this script without any model:
  1. Right passage found: the share of the 20 answerable questions where one of the
     retrieved chunks contains an accepted evidence passage.
  2. Right answer given: the share of the 20 answerable questions whose answer contains
     every required term (one of the alternatives in each group of `answer_terms`) and does
     not decline.
  3. Unanswerable declined: the share of the 10 unanswerable questions answered with
     NOT IN SOURCES.

Then the judge (Claude grading each answer) and how often it agrees with hand grades.
Agreement is reported two ways: raw agreement (share of answers where judge and hand grade
match) and Cohen's kappa, which subtracts the agreement two graders would reach by chance
given how often each says "correct" (1 = perfect, 0 = no better than chance).
"""

import json

from pipeline.chunk import load_chunks
from rageval.config import CONFIGS, DATA_TEXT, DECLINE, HAND_GRADES, RUNS, TESTSET


def normalise(s: str) -> str:
    for a, b in {"’": "'", "‘": "'", "–": "-", "—": "-", " ": " "}.items():
        s = s.replace(a, b)
    return " ".join(s.lower().split())


def declined(answer: str) -> bool:
    return DECLINE.lower() in answer.lower()


def terms_ok(q: dict, answer: str) -> bool:
    a = normalise(answer)
    return all(any(normalise(t) in a for t in group) for group in q["answer_terms"])


def script_grades(run: dict, questions: dict, chunk_text: dict) -> dict:
    """Per question: passage_found (answerable only) and correct (the script's own grade)."""
    out = {}
    for row in run["rows"]:
        q = questions[row["id"]]
        if q["type"] == "answerable":
            texts = [normalise(chunk_text[c]) for c in row["retrieved"]]
            found = any(normalise(e) in t for e in q["evidence"] for t in texts)
            correct = (not declined(row["answer"])) and terms_ok(q, row["answer"])
            out[row["id"]] = {"passage_found": found, "correct": correct}
        else:
            out[row["id"]] = {"passage_found": None, "correct": declined(row["answer"])}
    return out


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(judge: dict, hand: dict, ids: list[str]) -> dict:
    ids = [i for i in ids if i in hand and judge.get(i, {}).get("verdict") is not None]
    j = [judge[i]["verdict"] == "correct" for i in ids]
    h = [hand[i] == "correct" for i in ids]
    agree = sum(x == y for x, y in zip(j, h))
    return {"n": len(ids), "agree": agree, "kappa": kappa(j, h) if ids else None,
            "disagree": [i for i, x, y in zip(ids, j, h) if x != y]}


def main() -> None:
    questions = {q["id"]: q for q in json.loads(TESTSET.read_text(encoding="utf-8"))}
    ans_ids = [i for i, q in questions.items() if q["type"] == "answerable"]
    una_ids = [i for i, q in questions.items() if q["type"] == "unanswerable"]
    chunk_text = {c.id: c.text for c in load_chunks(DATA_TEXT)}
    hand_all = json.loads(HAND_GRADES.read_text(encoding="utf-8")) if HAND_GRADES.exists() else {}

    print(f"Test set: {len(ans_ids)} answerable, {len(una_ids)} unanswerable questions\n")
    for config in CONFIGS:
        path = RUNS / f"{config}.json"
        if not path.exists():
            print(f"{config}: no run yet (python run.py generate {config})\n")
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        g = script_grades(run, questions, chunk_text)
        found = sum(g[i]["passage_found"] for i in ans_ids)
        right = sum(g[i]["correct"] for i in ans_ids)
        decl = sum(g[i]["correct"] for i in una_ids)
        print(f"== {config} (search={run['search']}, k={run['k']}, {run['model']}, {run['time_utc']})")
        print(f"  1. right passage found   {found:>2}/{len(ans_ids)}")
        print(f"  2. right answer given    {right:>2}/{len(ans_ids)}")
        print(f"  3. unanswerable declined {decl:>2}/{len(una_ids)}")
        print(f"  answers falsely declined {sum(declined(r['answer']) for r in run['rows'] if r['id'] in ans_ids)}"
              f"/{len(ans_ids)}")
        cost = run["cost_usd"]
        print(f"  generation: {run['input_tokens']:,} in / {run['output_tokens']:,} out tokens, {run['cost_usd']:.4f} USD")

        jpath = RUNS / f"{config}_judge.json"
        if jpath.exists():
            judge = json.loads(jpath.read_text(encoding="utf-8"))
            v = judge["verdicts"]
            jr = sum(v[i]["verdict"] == "correct" for i in ans_ids)
            jd = sum(v[i]["verdict"] == "correct" for i in una_ids)
            print(f"  judge ({judge['judge_model']}, batch): answers right {jr}/{len(ans_ids)}, "
                  f"unanswerable declined {jd}/{len(una_ids)}; {judge['cost_usd']:.4f} USD")
            cost += judge["cost_usd"]
            hand = hand_all.get(config, {})
            if hand:
                ag = agreement(v, hand, list(questions))
                k = "n/a" if ag["kappa"] is None else f"{ag['kappa']:.2f}"
                print(f"  JUDGE vs HAND GRADES: {ag['agree']}/{ag['n']} agree "
                      f"({100 * ag['agree'] / ag['n']:.0f}%), Cohen's kappa {k}; disagree on {ag['disagree'] or 'none'}")
                sg = {i: {"verdict": "correct" if g[i]["correct"] else "incorrect"} for i in g}
                ag2 = agreement(sg, hand, list(questions))
                print(f"  script vs hand grades:  {ag2['agree']}/{ag2['n']} agree; disagree on {ag2['disagree'] or 'none'}")
        print(f"  total measured spend for this run: {cost:.4f} USD\n")

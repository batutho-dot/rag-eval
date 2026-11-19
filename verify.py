"""Check this repository after a change. Prints one line per step and ends with
`verify: PASS` or `verify: FAIL at <step>: <reason>`. Changes nothing in the repo: all
output goes to a temporary folder. Never calls the Claude API and needs no key.

    python verify.py          fast checks plus a real TF-IDF retrieval run (a few seconds)
    python verify.py --full   also the hybrid retrieval run (embedding model, about 2 minutes)

Uses the downloaded HTML in data/raw/ when present; otherwise downloads it (network).
"""

import io
import json
import py_compile
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
KNOWN = json.loads((ROOT / "verify_known_good.json").read_text(encoding="utf-8"))


class Fail(Exception):
    pass


def check(cond, reason):
    if not cond:
        raise Fail(reason)


def step_compile(tmp):
    files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "pipeline").glob("*.py")) + sorted((ROOT / "rageval").glob("*.py"))
    for f in files:
        try:
            py_compile.compile(str(f), cfile=str(tmp / "pyc" / (f.name + "c")), doraise=True)
        except py_compile.PyCompileError as e:
            raise Fail(f"{f.name}: {e.msg.strip().splitlines()[-1]}")
    return f"{len(files)} files compile"


def step_units(tmp):
    from pipeline import hybrid
    from rageval import generate, judge
    from rageval.config import cost_usd
    from rageval.report import declined, kappa, normalise, terms_ok

    check(normalise("It’s – A B") == "it's - a b", "normalise output changed")
    check(declined("NOT IN SOURCES") and declined("Not in sources.") and not declined("E1-6 [2]"), "declined() wrong")
    q = {"answer_terms": [["capex", "capital expenditure"], ["turnover"]]}
    check(terms_ok(q, "Turnover and Capital  expenditure") and not terms_ok(q, "turnover only"), "terms_ok wrong")
    check(kappa([True, False, True, False], [True, False, True, False]) == 1.0, "kappa of identical grades is not 1")
    check(abs(kappa([True, True, False, False], [True, False, True, False])) < 1e-9, "kappa of chance agreement is not 0")
    check(abs(cost_usd("claude-haiku-5-5", 50_000, 1_000_000) - 0.505) < 1e-9, "Haiku price changed")
    check(abs(cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000, batch=True) - 6.00) < 1e-9, "Sonnet batch price changed")
    try:
        cost_usd("claude-haiku-5-5", 100_001, 0)
        raise Fail("cost_usd priced a Haiku prompt above the 100K step")
    except ValueError:
        pass
    p = generate.prompt("Q?", ["alpha", "beta"])
    check("[1] alpha" in p and "[2] beta" in p and p.endswith("Question: Q?"), "generation prompt layout changed")

    class Fake:  # two fixed rankings, to check the fusion arithmetic without the model
        def __init__(self, order):
            self.order = order

        def search(self, query, k):
            return [(i, 0.0) for i in self.order[:k]]

    h = hybrid.HybridIndex.__new__(hybrid.HybridIndex)
    h.a, h.b = Fake([0, 1, 2, 3]), Fake([3, 1, 2, 0])
    check([i for i, _ in h.search("x", 4)][0] == 1, "RRF: the chunk ranked 2nd by both should come first")

    qs = {"q": {"id": "q", "type": "answerable", "question": "Q?", "answer": "A", "where": "W", "evidence": ["E"]},
          "u": {"id": "u", "type": "unanswerable", "question": "U?", "why": "none"}}
    reqs = judge.build_requests({"rows": [{"id": "q", "answer": "a"}, {"id": "u", "answer": "NOT IN SOURCES"}]}, qs)
    check([r["custom_id"] for r in reqs] == ["q", "u"], "judge custom_ids wrong")
    check(reqs[0]["params"]["output_config"]["format"]["type"] == "json_schema", "judge lost its JSON schema")
    check("ANSWERABLE" in reqs[0]["params"]["messages"][0]["content"] and "UNANSWERABLE" in reqs[1]["params"]["messages"][0]["content"],
          "judge prompt does not state the question type")
    return "normalise, decline, terms, kappa, prices, prompt, RRF, judge requests"


def step_testset(tmp):
    qs = json.loads((ROOT / "testset.json").read_text(encoding="utf-8"))
    ids = [q["id"] for q in qs]
    check(len(set(ids)) == len(ids), "duplicate ids")
    ans = [q for q in qs if q["type"] == "answerable"]
    una = [q for q in qs if q["type"] == "unanswerable"]
    check((len(ans), len(una)) == (36, 14), f"{len(ans)} answerable / {len(una)} unanswerable, expected 36 / 14")
    for q in ans:
        check(all(q.get(k) for k in ("question", "doc", "where", "answer", "answer_terms", "evidence")), f"{q['id']} missing a field")
        check(all(len(e.split()) <= 50 for e in q["evidence"]), f"{q['id']} evidence longer than the chunk overlap")
    for q in una:
        check(q.get("question") and q.get("why"), f"{q['id']} missing a field")
    return "36 answerable + 14 unanswerable, all fields, evidence fits in the overlap"


def corpus(tmp):
    from pipeline import download
    from pipeline.chunk import load_chunks

    data = tmp / "data"
    if not data.exists():
        if (ROOT / "data" / "raw").is_dir():
            shutil.copytree(ROOT / "data" / "raw", data / "raw")
        with redirect_stdout(io.StringIO()):
            download.main(data)
    return load_chunks(data / "text")


def missed(index_cls, chunks):
    from rageval.config import K
    from rageval.report import normalise

    qs = [q for q in json.loads((ROOT / "testset.json").read_text(encoding="utf-8")) if q["type"] == "answerable"]
    texts = [c.text for c in chunks]
    norm = [normalise(t) for t in texts]
    for q in qs:  # the test itself must be sound: every evidence passage exists somewhere
        for e in q["evidence"]:
            check(any(normalise(e) in t for t in norm), f"{q['id']} evidence not in the corpus: {e[:40]}")
    index = index_cls(texts)
    out = []
    for q in qs:
        top = [norm[i] for i, _ in index.search(q["question"], K)]
        if not any(normalise(e) in t for e in q["evidence"] for t in top):
            out.append(q["id"])
    return out


def step_real_tfidf(tmp):
    from pipeline.version_a import TfidfIndex

    chunks = corpus(tmp)
    check(len(chunks) == KNOWN["chunks"], f"{len(chunks)} chunks, expected {KNOWN['chunks']}")
    m = missed(TfidfIndex, chunks)
    check(m == KNOWN["tfidf_missed"], f"missed {m}, known-good {KNOWN['tfidf_missed']}")
    return f"TF-IDF on the real corpus: passage found {36 - len(m)}/36, matches verify_known_good.json"


def step_report(tmp):
    """Drive the report on a fixed fake run, so the three scores and the agreement are checked."""
    from rageval import config, report

    qs = json.loads((ROOT / "testset.json").read_text(encoding="utf-8"))
    chunks = corpus(tmp)
    by_text = {c.id: c.text for c in chunks}
    ev_chunk = {}
    for q in qs:
        if q["type"] == "answerable":
            ev_chunk[q["id"]] = next(c.id for c in chunks if report.normalise(q["evidence"][0]) in report.normalise(c.text))
    rows = []
    for q in qs:  # answerable: right passage and reference answer; unanswerable: decline
        if q["type"] == "answerable":
            rows.append({"id": q["id"], "retrieved": [ev_chunk[q["id"]]], "answer": q["answer"]})
        else:
            rows.append({"id": q["id"], "retrieved": [], "answer": "NOT IN SOURCES"})
    rows[0]["answer"] = "NOT IN SOURCES"  # one false decline: expect 35/36 right answers
    runs = tmp / "runs"
    runs.mkdir()
    run = {"search": "tfidf", "k": 5, "model": "fake", "time_utc": "fixed", "input_tokens": 0,
           "output_tokens": 0, "cost_usd": 0.0, "rows": rows}
    (runs / "baseline.json").write_text(json.dumps(run), encoding="utf-8")
    verdicts = {r["id"]: {"verdict": "correct", "reason": ""} for r in rows}
    (runs / "baseline_judge.json").write_text(json.dumps(
        {"judge_model": "fake", "cost_usd": 0.0, "verdicts": verdicts}), encoding="utf-8")
    hand = {"baseline": {r["id"]: "correct" for r in rows}}
    hand["baseline"][rows[0]["id"]] = "incorrect"  # the judge is wrong on that one
    (tmp / "hand.json").write_text(json.dumps(hand), encoding="utf-8")

    old = (report.RUNS, report.HAND_GRADES, report.DATA_TEXT)
    report.RUNS, report.HAND_GRADES, report.DATA_TEXT = runs, tmp / "hand.json", tmp / "data" / "text"
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            report.main()
    finally:
        report.RUNS, report.HAND_GRADES, report.DATA_TEXT = old
    out = buf.getvalue()
    for want in ("right passage found   36/36", "right answer given    35/36", "unanswerable declined 14/14",
                 "JUDGE vs HAND GRADES: 49/50 agree", "disagree on ['q01']"):
        check(want in out, f"report output lacks {want!r}")
    return "fixed fake run: 36/36, 35/36, 14/14, judge agreement 49/50"


def step_real_hybrid(tmp):
    from pipeline.hybrid import HybridIndex

    m = missed(HybridIndex, corpus(tmp))
    check(m == KNOWN["hybrid_missed"], f"missed {m}, known-good {KNOWN['hybrid_missed']}")
    return f"hybrid on the real corpus: passage found {36 - len(m)}/36, matches verify_known_good.json"


def main():
    full = "--full" in sys.argv
    print(f"verify-repo: {ROOT}")
    steps = [("compile", step_compile), ("units", step_units), ("testset", step_testset),
             ("real-run-tfidf", step_real_tfidf), ("report", step_report)]
    if full:
        steps.append(("real-run-hybrid", step_real_hybrid))
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, fn in steps:
            try:
                print(f"{name}: ok ({fn(tmp)})")
            except Fail as e:
                print(f"verify: FAIL at {name}: {e}")
                return 1
            except Exception as e:  # a crash is a failure of that step, never a pass
                print(f"verify: FAIL at {name}: {type(e).__name__}: {e}")
                return 1
    if not full:
        print("real-run-hybrid: skipped (run with --full)")
    print("verify: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

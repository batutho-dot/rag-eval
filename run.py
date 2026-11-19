"""rag-eval: score a small RAG pipeline on EU sustainability law.

    python run.py                    print the scores from the saved runs (free, no API key)
    python run.py retrieval          right passage found on the whole test set, no API call
    python run.py download           fetch the three EU texts from EUR-Lex into data/
    python run.py generate <config>  retrieve + answer with Claude Haiku 5.5 (costs cents)
    python run.py judge <config>     grade the answers with Claude through the Batches API
    python run.py all                download, then generate and judge every config, then report

<config> is baseline or hybrid (see rageval/config.py).
"""

import sys

from rageval import report
from rageval.config import CONFIGS

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report.main()
    elif cmd == "retrieval":
        report.retrieval(sys.argv[2:] or None)
    elif cmd == "download":
        from pipeline import download

        download.main()
    elif cmd in ("generate", "judge"):
        config = sys.argv[2]
        if config not in CONFIGS:
            sys.exit(f"unknown config {config!r}; choose from {list(CONFIGS)}")
        from rageval import generate, judge

        (generate if cmd == "generate" else judge).main(config)
    elif cmd == "all":
        from pipeline import download
        from rageval import generate, judge

        download.main()
        for config in CONFIGS:
            generate.main(config)
            judge.main(config)
        report.main()
    else:
        sys.exit(__doc__)

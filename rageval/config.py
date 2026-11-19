"""Everything a run depends on, in one place: models, prices, paths, the two pipeline configs.

Model ids and prices were read from Anthropic's documentation on 2026-11-19:
https://platform.claude.com/docs/en/about-claude/pricing
Prices are USD per million tokens. The Batches API is billed at half the normal price.
Claude Haiku 5.5 costs more for prompts over 100,000 tokens; every prompt here is far below
that, and `cost_usd` refuses to price one that is not.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_TEXT = ROOT / "data" / "text"
TESTSET = ROOT / "testset.json"
RUNS = ROOT / "runs"
HAND_GRADES = ROOT / "hand_grades.json"

GENERATOR = "claude-haiku-5-5"
JUDGE = "claude-sonnet-5-5"

PRICES = {  # (input, output) USD per million tokens, standard (not batch) rates
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
BATCH_DISCOUNT = 0.5
HAIKU_PRICE_STEP = 100_000  # prompt tokens above which Haiku 5.5 has a higher price

K = 5  # chunks handed to the model
CONFIGS = {
    "baseline": {"search": "tfidf", "k": K},
    "hybrid": {"search": "hybrid", "k": K},
}

DECLINE = "NOT IN SOURCES"
MAX_RUN_USD = 5.0  # a run stops and asks before spending more than this


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment, without overriding a variable
    that is already set. .env is git-ignored; the key is never printed or logged."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def require_key() -> None:
    """Stop before any paid call if no key is set. Checks only that it exists."""
    import os
    import sys

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (environment variable or .env); no API call made.")


def cost_usd(model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
    if model == "claude-haiku-5-5" and input_tokens > HAIKU_PRICE_STEP:
        raise ValueError("prompt above the Haiku 5.5 price step; price it by hand")
    p_in, p_out = PRICES[model]
    usd = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
    return usd * (BATCH_DISCOUNT if batch else 1.0)

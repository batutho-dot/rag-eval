"""The pipeline change under test: hybrid search.

Runs both of project 2's searches (TF-IDF in numpy, and the bge-small embedding model) and
merges their two ranked lists with reciprocal rank fusion (RRF): every chunk gets
1 / (RRF_K + rank) from each list it appears in, and the chunks are re-sorted on the sum.
A chunk that both methods rank fairly high beats one that only one method ranks first.
RRF_K damps the advantage of rank 1 over rank 2; 60 was picked once and not tuned here.
"""

from pipeline.version_a import TfidfIndex
from pipeline.version_b import EmbeddingIndex

RRF_K = 60
DEPTH = 50  # how far down each list is read before fusing


class HybridIndex:
    def __init__(self, texts: list[str]):
        self.a = TfidfIndex(texts)
        self.b = EmbeddingIndex(texts)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for index in (self.a, self.b):
            for rank, (i, _) in enumerate(index.search(query, DEPTH), start=1):
                scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(i, s) for i, s in top]

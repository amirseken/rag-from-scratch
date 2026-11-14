"""Two-stage retrieval: fast vector search finds candidates, a cross-encoder reranks them.

Stage 1 (bi-encoder, store.py): question and chunks are embedded SEPARATELY, so chunk
        vectors are precomputed and searching 100,000 chunks is quick. But the model
        never sees the question and a chunk together, so it can miss fine detail.
Stage 2 (cross-encoder, here): the model reads question + chunk TOGETHER and outputs one
        relevance score. Much more accurate, but too slow to run on every chunk, so we
        only run it on the top CANDIDATES from stage 1.
"""
from fastembed.rerank.cross_encoder import TextCrossEncoder

import config
from store import embed


def load_reranker():
    """Load the cross-encoder, or return None when reranking is switched off in config.py."""
    if not config.RERANK:
        return None
    return TextCrossEncoder(config.RERANK_MODEL, cache_dir=config.EMBED_CACHE_DIR)


def retrieve(store, embed_model, reranker, question: str, k: int = None):
    """Return the k best (chunk, score) pairs for a question.

    With a reranker the score is the cross-encoder's relevance score (any real number,
    higher = better, NOT a cosine similarity). Without one it is cosine similarity.
    """
    k = k or config.TOP_K
    q_vec = embed(embed_model, [question])[0]
    if reranker is None:
        return store.search(q_vec, k)

    candidates = store.search(q_vec, max(config.CANDIDATES, k))
    if not candidates:
        return []
    scores = list(reranker.rerank(question, [chunk["text"] for chunk, _ in candidates]))
    ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])
    return [(chunk, float(score)) for (chunk, _), score in ranked[:k]]

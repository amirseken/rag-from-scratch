"""Measure retrieval quality: is each answer's key phrase found in the top-k chunks, and at what rank?"""

import config
from fastembed import TextEmbedding
from store import VectorStore
from rerank import load_reranker, retrieve

# (question, short phrase that must appear in a chunk that answers it)
TESTS = [
    ("How many copies of Batman comics have been sold worldwide?", "460 million"),
    ("Which fictional character was Batman inspired from?", "Shadow"),
    ("Who voiced Batman in animated media and games?", "Kevin Conroy"),
    ("What caused DC to begin crediting Finger's contribution as one of the creators of Batman in 2015??", "legal action by Finger's family"),
    ("At what issue was Robin introduced as Batman's sidekick?", "issue #38"),
    ("Who murdered Jason Todd?", "Joker"),
    ("What is the newest Batman series?", "Absolute Batman"),
    ("What does Gotham do to call Batman?", "Bat-Signal"),
    ("Batman's first love interest?", "Julie Madison"),
    ("Who created Batman?", "Bill Finger")
]

store = VectorStore(config.INDEX_DIR)
model = TextEmbedding(config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)
reranker = load_reranker()


found, rank_sum = 0, 0.0
for question, phrase in TESTS:
    results = retrieve(store, model, reranker, question)
    rank = next((i for i, (c, _) in enumerate(results, 1) if phrase.lower() in c["text"].lower()), None)
    if rank:
        found += 1
        rank_sum += 1 / rank
    print(f"{'rank ' + str(rank) if rank else 'MISS':8} | {question}")

n = len(TESTS)
print(f"\nFound in top {config.TOP_K}: {found}/{n}   |   mean reciprocal rank: {rank_sum / n:.2f}")
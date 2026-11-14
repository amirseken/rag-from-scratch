import sys
import config
from fastembed import TextEmbedding
from store import VectorStore, embed

store = VectorStore(config.INDEX_DIR)
model = TextEmbedding(config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)
question, phrase = sys.argv[1], sys.argv[2]

ranked = store.search(embed(model, [question])[0], len(store))  # rank ALL chunks
for rank, (chunk, score) in enumerate(ranked, 1):
    if phrase.lower() in chunk["text"].lower():
        print(f"phrase first found at rank {rank} of {len(store)} (score {score:.2f}), chunk {chunk['chunk']}")
        break
else:
    print("phrase is not in any chunk: fix the phrase, or the page wasn't crawled/split as expected")
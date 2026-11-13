import sys
import config
from fastembed import TextEmbedding
from store import VectorStore, embed

store = VectorStore(config.INDEX_DIR)
model = TextEmbedding(config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)
question = " ".join(sys.argv[1:])

for chunk, score in store.search(embed(model, [question])[0], config.TOP_K):
    print(f"--- score {score:.2f} | {chunk['url']} | chunk {chunk['chunk']}")
    print(chunk["text"], "\n")
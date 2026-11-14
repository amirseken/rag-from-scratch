# RAG from scratch: ask questions about any website

A small retrieval-augmented generation (RAG) system built in plain Python, with no LangChain and no vector database. It crawls a website, indexes the text locally, retrieves with vector search plus a cross-encoder reranker, and answers questions in the terminal with citations to the source pages.

```
INDEXING   ingest.py
  crawl pages -> extract article text -> chunk (200 words, 40 overlap)
  -> drop reference-list chunks -> embed locally (fastembed) -> index/ (NumPy + JSON)

ANSWERING  ask.py  (retrieval in rerank.py)
  question -> embed -> stage 1: top-20 chunks by dot product
           -> stage 2: cross-encoder rescored, keep top-5
           -> Gemini answers using only those chunks, with [1] [2] citations
```

## Results

I wrote 10 questions about one Wikipedia article, each with a short phrase that must appear in a retrieved chunk for the question to be answerable (`eval.py`).

| Retrieval | Answer found in top 5 | Correct chunk ranked 1st |
|---|---|---|
| Vector search only | 8/10 | not all |
| Vector search + cross-encoder reranking | 10/10 | 10/10 |

Caveats: this is a small sanity check, not a benchmark. It covers one article and ten questions, so one question moves the result by 10 points. One question ("What is the newest Batman series?") is ambiguous, because "newest" depends on the date. I also corrected one test phrase that did not match the page's actual wording; the numbers above use the corrected test for both rows.

## Design decisions

- **No framework.** Every stage is a short, readable function, so you can see what RAG actually does.
- **No vector database.** Embeddings are normalised to length 1, so cosine similarity is a dot product. Search is one matrix-vector multiplication in `store.py` (about 10 lines). This is fast enough up to tens of thousands of chunks; beyond that, use an approximate index such as FAISS or a database.
- **Local embeddings.** `fastembed` runs `all-MiniLM-L6-v2` on CPU through ONNX Runtime: free, no API key, and a much smaller install than PyTorch.
- **Two-stage retrieval.** The embedding model reads the question and each chunk separately, which makes search fast but loses detail. A cross-encoder (`ms-marco-MiniLM-L-6-v2`, about 80 MB) reads the question and a chunk together and scores relevance more accurately, but is too slow for the whole index. So vector search picks 20 candidates and the cross-encoder reorders only those.
- **Gemini for generation**, called through plain HTTP (`requests`), so the free tier is enough. The system prompt tells the model to answer only from the retrieved passages and to say "I don't know" otherwise.

## What I learned building it

- **Retrieval quality is capped by what is in the index.** On my first test, the correct answer ranked 2nd. The 1st result was a chunk of Wikipedia reference citations ("Retrieved April 4, 2018. Archived from the original..."). Printing the retrieved chunks, not just the final answer, exposed it, and a small filter in `ingest.py` that drops citation-heavy chunks fixed the ranking.
- **Check the test before blaming the system.** One "miss" turned out to be a test phrase that was not in the page. A diagnostic script that finds where the phrase ranks across the whole index separated "retrieval failed" from "the test was wrong".
- **Raising top-k widens the net but does not improve ranking.** Going from top 5 to top 10 found one more answer (at rank 8), but the ordering barely changed.
- **A synonym gap beat the embeddings, and the reranker fixed it.** The question "What does Gotham do to call Batman?" ranked its answer 13th by vector search, because the page says "Bat-Signal". The cross-encoder moved it to 1st.
- **Similarity scores are not percentages.** With a small embedding model, a short question against a 200-word chunk scores around 0.2 to 0.3 even for a correct match. What matters is rank and the gap to the next result.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --no-cache-dir -r requirements.txt
export GEMINI_API_KEY=your-key       # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

Get a free key from Google AI Studio. Model names change over time: run `python list_models.py` to see what your key can use, and set `LLM_MODEL` in `config.py`. The first run downloads the embedding model (about 90 MB) and the reranker (about 80 MB).

## Usage

```bash
python ingest.py https://en.wikipedia.org/wiki/Batman --max-pages 10
python ask.py
```

Crawl only sites you are allowed to crawl; the crawler stays on one domain and waits 0.5 s between requests. The index is stored in `index/` and is not included in this repository.

To measure retrieval on your own data, edit the `TESTS` list in `eval.py` (a question plus a phrase that must appear in a relevant chunk) and run `python eval.py`. Set `RERANK = False` in `config.py` to compare against plain vector search.

## Project structure

| File | Purpose |
|---|---|
| `ingest.py` | Crawl, extract, chunk, filter, embed, save |
| `store.py` | The vector store: NumPy matrix + JSON, cosine search |
| `rerank.py` | Two-stage retrieval: vector search, then cross-encoder reranking |
| `ask.py` | Retrieve chunks and generate a cited answer |
| `eval.py` | Retrieval test set: hit rate and rank per question |
| `config.py` | Chunk size, top-k, models, reranking switch |
| `list_models.py` | List Gemini models available to your key |

## Limitations

- Fixed-size word chunks can cut sentences in half.
- Pages that need JavaScript to render are not supported (the crawler reads raw HTML).
- No conversation memory: each question is answered independently, so follow-up questions without a subject retrieve poorly.
- The test set is small and covers a single article. The reranker can only reorder the 20 candidates that vector search returns.

## Roadmap

- [ ] Heading-aware chunking
- [ ] A larger, harder test set (multi-chunk answers, questions the page cannot answer)
- [ ] Follow-up questions via query rewriting
- [ ] Hybrid search (BM25 + vectors)
- [ ] Simple web UI

## Note on data

Indexed content belongs to its original authors (for example, Wikipedia text is available under CC BY-SA). This repository contains no crawled data.

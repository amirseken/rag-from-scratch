# RAG from scratch: ask questions about any website

A small retrieval-augmented generation (RAG) system built in plain Python, with no LangChain and no vector database. It crawls a website, indexes the text locally, and answers questions in the terminal with citations to the source pages.

```
INDEXING   ingest.py
  crawl pages -> extract article text -> chunk (200 words, 40 overlap)
  -> drop reference-list chunks -> embed locally (fastembed) -> index/ (NumPy + JSON)

ANSWERING  ask.py
  question -> embed -> top-5 chunks (dot product) -> Gemini answers using only those chunks, with [1] [2] citations
```

## Design decisions

- **No framework.** Every stage is a short, readable function, so you can see what RAG actually does.
- **No vector database.** Embeddings are normalised to length 1, so cosine similarity is a dot product. Search is one matrix-vector multiplication in `store.py` (about 10 lines). This is fast enough up to tens of thousands of chunks; beyond that, use an approximate index such as FAISS or a database.
- **Local embeddings.** `fastembed` runs `all-MiniLM-L6-v2` on CPU through ONNX Runtime: free, no API key, and a much smaller install than PyTorch.
- **Gemini for generation.** Called through plain HTTP (`requests`), so the free tier is enough to run the project. The system prompt tells the model to answer only from the retrieved passages and to say "I don't know" otherwise.

## What I learned building it

- **Retrieval quality is capped by what is in the index.** On my first test, the correct answer was only ranked 2nd. The 1st result was a chunk of Wikipedia reference citations ("Retrieved April 4, 2018. Archived from the original..."). Printing the retrieved chunks (not just the final answer) exposed it, and a small filter in `ingest.py` that drops citation-heavy chunks fixed the ranking.
- **Similarity scores are not percentages.** With a small embedding model, a short question against a 200-word chunk scores around 0.2 to 0.3 even for a correct match. What matters is the rank and the gap to the next result.
- **Specific questions retrieve better.** Adding the subject ("of Batman comics") to a question improves matching, which is why follow-up questions without a subject are a weak spot.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --no-cache-dir -r requirements.txt
export GEMINI_API_KEY=your-key       # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

Get a free key from Google AI Studio. Model names change over time: run `python list_models.py` to see what your key can use, and set `LLM_MODEL` in `config.py`.

## Usage

```bash
python ingest.py https://en.wikipedia.org/wiki/Batman --max-pages 10
python ask.py
```

Crawl only sites you are allowed to crawl; the crawler stays on one domain and waits 0.5 s between requests. The index is stored in `index/` and is not included in this repository.

## Project structure

| File | Purpose |
|---|---|
| `ingest.py` | Crawl, extract, chunk, filter, embed, save |
| `store.py` | The vector store: NumPy matrix + JSON, cosine search |
| `ask.py` | Retrieve top-k chunks and generate a cited answer |
| `config.py` | Chunk size, overlap, top-k, models |
| `list_models.py` | List Gemini models available to your key |

## Limitations

- Fixed-size word chunks can cut sentences in half.
- Pages that need JavaScript to render are not supported (the crawler reads raw HTML).
- No conversation memory: each question is answered independently.
- No automated evaluation of retrieval quality.

## Roadmap

- [ ] Heading-aware chunking
- [ ] Follow-up questions via query rewriting
- [ ] A small test set to measure retrieval changes (hit rate, MRR)
- [ ] Hybrid search (BM25 + vectors) and reranking
- [ ] Simple web UI

## Note on data

Indexed content belongs to its original authors (for example, Wikipedia text is available under CC BY-SA). This repository contains no crawled data.

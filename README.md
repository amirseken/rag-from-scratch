# RAG pet project (lightweight)

Ask questions about a website's docs. Plain Python, no framework, small install (~300 MB).

```
ingest.py:  crawl -> extract text -> chunk -> embed (fastembed, local) -> index/ (NumPy + JSON)
ask.py:     question -> embed -> top-k chunks (dot product) -> Claude answers with citations
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --no-cache-dir -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...  # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

```bash
python ingest.py https://docs.example.com --max-pages 50
python ask.py
```

Only crawl sites you're allowed to, and keep the delay polite.

## Things to experiment with

1. **Chunk size / overlap** in `config.py`: re-ingest and see how answers change.
2. **TOP_K**: too low misses context, too high adds noise.
3. **Embedding model**: try `BAAI/bge-small-en-v1.5` (also supported by fastembed).
4. **Prompt**: loosen or tighten the "only use the context" rule.

## Roadmap

- [ ] Heading-aware chunking (split on headings instead of word windows)
- [ ] Conversation memory + query rewriting for follow-up questions
- [ ] Hybrid search (BM25 + vectors) and a reranker
- [ ] Small eval set: 20 questions with known answers, measure hit rate
- [ ] Wrap in FastAPI or a Streamlit UI

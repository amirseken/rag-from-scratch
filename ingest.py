"""Crawl a website, chunk the text, embed it, and save it to the local index.
 
Usage:
    python ingest.py https://docs.example.com --max-pages 50
"""
import argparse
import time
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse
 
import requests
import trafilatura
from bs4 import BeautifulSoup
from fastembed import TextEmbedding
 
import config
from store import VectorStore, embed
 
HEADERS = {"User-Agent": "rag-pet-project/0.1 (learning project)"}
 
REFERENCE_MARKERS = ["↑", "Retrieved", "Archived from the original", "ISBN", "doi:", "hdl:"]


def looks_like_references(chunk: str) -> bool:
    """True for chunks that are mostly citations (reference lists), which are noise for retrieval."""
    return sum(chunk.count(m) for m in REFERENCE_MARKERS) >= 5
 
def crawl(start_url: str, max_pages: int, delay: float):
    """Breadth-first crawl limited to the start URL's domain. Yields (url, html)."""
    domain = urlparse(start_url).netloc
    queue, seen = deque([start_url]), {start_url}
    fetched = 0
 
    while queue and fetched < max_pages:
        url = queue.popleft()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                continue
        except requests.RequestException as e:
            print(f"  skip {url}: {e}")
            continue
 
        fetched += 1
        yield url, resp.text
 
        for a in BeautifulSoup(resp.text, "html.parser").find_all("a", href=True):
            link, _ = urldefrag(urljoin(url, a["href"]))
            if urlparse(link).netloc == domain and link not in seen:
                seen.add(link)
                queue.append(link)
        time.sleep(delay)  # be polite to the server
 
 
def chunk_text(text: str, size: int, overlap: int):
    """Split text into overlapping windows of `size` words."""
    words = text.split()
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= 20:  # drop tiny trailing fragments
            yield " ".join(piece)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Start URL to crawl")
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    args = parser.parse_args()
 
    print(f"Loading embedding model {config.EMBED_MODEL} (first run downloads ~90 MB) ...")
    model = TextEmbedding(config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)
 
    store = VectorStore(config.INDEX_DIR)
 
    total_chunks = 0
    for url, html in crawl(args.url, args.max_pages, args.delay):
        text = trafilatura.extract(html)  # strips nav, footers, ads
        if not text:
            continue
        chunks = list(chunk_text(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP))
        chunks = [c for c in chunks if not looks_like_references(c)]
        if not chunks:
            continue
 
        store.replace_url(url, chunks, embed(model, chunks))
        store.save()  # save after every page so an interruption loses nothing
        total_chunks += len(chunks)
        print(f"  {url} -> {len(chunks)} chunks")
 
    print(f"Done. Added {total_chunks} chunks. Index size: {len(store)}")
 
 
if __name__ == "__main__":
    main()
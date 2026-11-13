"""Ask questions about the ingested docs from the terminal (answers by Google Gemini).
 
Usage:
    export GEMINI_API_KEY=your-key       (Windows PowerShell: $env:GEMINI_API_KEY="your-key")
    python ask.py
"""
import os
import sys
 
import requests
from fastembed import TextEmbedding
 
import config
from store import VectorStore, embed
 
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
 
SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
- Cite the passages you used like [1], [2].
- If the context does not contain the answer, say you don't know based on the indexed docs.
- Do not invent details that are not in the context."""
 
 
def retrieve(store, model, question: str):
    """Embed the question and fetch the most similar chunks."""
    q_vec = embed(model, [question])[0]
    return store.search(q_vec, config.TOP_K)
 
 
def build_prompt(question: str, hits) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {chunk['url']})\n{chunk['text']}" for i, (chunk, _) in enumerate(hits, 1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"
 
 
def generate(api_key: str, prompt: str) -> str:
    """Call the Gemini REST API and return the answer text."""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": config.MAX_TOKENS},
    }
    resp = requests.post(
        API_URL.format(model=config.LLM_MODEL),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        hint = ""
        if resp.status_code == 404:
            hint = "  (model name not found: run `python list_models.py` and update LLM_MODEL in config.py)"
        elif resp.status_code == 429:
            hint = "  (rate limit hit: wait a minute and try again)"
        elif resp.status_code in (400, 403):
            hint = "  (check that GEMINI_API_KEY is correct)"
        return f"[API error {resp.status_code}]{hint}\n{resp.text[:300]}"
 
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return f"[No answer returned. Response: {str(data)[:300]}]"
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    return text or "[Empty answer: try raising MAX_TOKENS in config.py]"
 
 
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=your-key")
 
    store = VectorStore(config.INDEX_DIR)
    if len(store) == 0:
        sys.exit("The index is empty. Run ingest.py first.")
 
    model = TextEmbedding(config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR)
 
    print(f"Index has {len(store)} chunks. Type a question (or 'exit').\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
 
        hits = retrieve(store, model, question)
        answer = generate(api_key, build_prompt(question, hits))
 
        print(f"\nassistant> {answer}\n\nSources:")
        for i, (chunk, score) in enumerate(hits, 1):
            print(f"  [{i}] {chunk['url']}  (similarity {score:.2f})")
        print()
 
 
if __name__ == "__main__":
    main()
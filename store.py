"""A tiny vector store: one NumPy matrix + one JSON file. No database needed.

Because all vectors are normalised to length 1, cosine similarity is just a
dot product, so searching = one matrix-vector multiplication.
"""
import json
from pathlib import Path

import numpy as np


def embed(model, texts):
    """Embed texts with a fastembed model and normalise to unit length."""
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


class VectorStore:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.vec_path = self.dir / "vectors.npy"
        self.meta_path = self.dir / "chunks.json"
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.chunks = []  # list of {"url", "chunk", "text"}, same order as vectors
        if self.vec_path.exists() and self.meta_path.exists():
            self.vectors = np.load(self.vec_path)
            self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def __len__(self):
        return len(self.chunks)

    def replace_url(self, url, texts, vectors):
        """Drop any old chunks of this page and add the new ones."""
        keep = [i for i, c in enumerate(self.chunks) if c["url"] != url]
        new_vecs = np.asarray(vectors, dtype=np.float32)
        new_chunks = [{"url": url, "chunk": i, "text": t} for i, t in enumerate(texts)]
        if keep:
            self.vectors = np.vstack([self.vectors[keep], new_vecs])
        else:
            self.vectors = new_vecs
        self.chunks = [self.chunks[i] for i in keep] + new_chunks

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self.vectors)
        self.meta_path.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    def search(self, query_vec, k):
        """Return the k most similar chunks as (chunk_dict, similarity) pairs."""
        if not self.chunks:
            return []
        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

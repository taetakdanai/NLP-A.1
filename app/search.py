import json
import re
import numpy as np
from typing import List, Dict, Tuple

TOKEN_RE = re.compile(r"[a-z0-9']+")

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def load_resources(corpus_path: str, embeddings_path: str, word2idx_path: str):
    # corpus contexts: 1 context per line (you can store paragraphs/lines)
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        contexts = [line.strip() for line in f if line.strip()]

    with open(word2idx_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    embeddings = np.load(embeddings_path).astype(np.float32)  # (V,D)

    # normalize embeddings for cosine-like dot (optional but usually better)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / np.maximum(norms, 1e-12)

    return contexts, word2idx, emb_norm

def text_to_vec(text: str, word2idx: Dict[str, int], emb_norm: np.ndarray) -> np.ndarray:
    """
    Build a vector for a query/context by averaging word vectors.
    Returns a normalized vector (unit length) for stable dot products.
    """
    toks = simple_tokenize(text)
    idxs = [word2idx[t] for t in toks if t in word2idx]
    if not idxs:
        return np.zeros((emb_norm.shape[1],), dtype=np.float32)

    v = emb_norm[idxs].mean(axis=0)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)

def build_corpus_matrix(contexts: List[str], word2idx: Dict[str,int], emb_norm: np.ndarray) -> np.ndarray:
    """
    Precompute one vector per context for fast search.
    Returns matrix C of shape (N_contexts, D), each row unit-normalized.
    """
    D = emb_norm.shape[1]
    C = np.zeros((len(contexts), D), dtype=np.float32)
    for i, ctx in enumerate(contexts):
        C[i] = text_to_vec(ctx, word2idx, emb_norm)
    return C

def topk_dot(query: str, contexts: List[str], C: np.ndarray,
             word2idx: Dict[str,int], emb_norm: np.ndarray, k: int = 10) -> List[Tuple[int, float, str]]:
    """
    Compute dot product between query vector and each context vector, return top-k.
    """
    q = text_to_vec(query, word2idx, emb_norm)
    if np.allclose(q, 0):
        return []

    scores = C @ q  # dot product (cosine-like since vectors are normalized)
    if len(scores) == 0:
        return []

    k = min(k, len(scores))
    # fast top-k
    top_idx = np.argpartition(-scores, k-1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [(int(i), float(scores[i]), contexts[i]) for i in top_idx]

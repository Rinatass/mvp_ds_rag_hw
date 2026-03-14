import os
import pickle
from pathlib import Path

import faiss
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import chunk_pdf


# ─── Промпты ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Ты — ассистент, отвечающий на вопросы строго по предоставленному контексту.
Используй ТОЛЬКО информацию из контекста. Если информации недостаточно — скажи об этом.
Указывай номера страниц. Отвечай кратко и по делу."""

USER_PROMPT = """\
Контекст:
{context}

Вопрос: {query}

Ответ:"""

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _is_e5():
    return "e5" in EMBEDDING_MODEL.lower()


def embed_documents(texts: list[str]) -> np.ndarray:
    if _is_e5():
        texts = [f"passage: {t}" for t in texts]
    return _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=True)


def embed_query(query: str) -> np.ndarray:
    if _is_e5():
        query = f"query: {query}"
    return _get_model().encode([query], normalize_embeddings=True)[0]


def build_index(chunks: list[dict], index_dir: str = "rag_index") -> faiss.Index:
    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{index_dir}/index.faiss")
    with open(f"{index_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Индекс сохранён: {len(chunks)} чанков, dim={embeddings.shape[1]}")
    return index


def load_index(index_dir: str = "rag_index"):
    index = faiss.read_index(f"{index_dir}/index.faiss")
    with open(f"{index_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(query: str, index: faiss.Index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    qvec = embed_query(query).astype(np.float32).reshape(1, -1)
    scores, indices = index.search(qvec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append({**chunks[idx], "score": float(score)})
    return results


LLM_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_llm(query: str, context: str) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]

    resp = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(context=context, query=query)},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def format_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"[Страница {r['page']}, score {r['score']:.2f}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def ask(query: str, index: faiss.Index, chunks: list[dict], top_k: int = 5) -> str:
    results = retrieve(query, index, chunks, top_k)
    context = format_context(results)
    return call_llm(query, context)


if __name__ == "__main__":
    PDF_PATH = "document.pdf"
    INDEX_DIR = "rag_index"
    if not Path(INDEX_DIR).exists():
        chunks = chunk_pdf(PDF_PATH)
        build_index(chunks, INDEX_DIR)

    index, chunks = load_index(INDEX_DIR)

    answer = ask("О чём этот документ?", index, chunks)
    print(answer)
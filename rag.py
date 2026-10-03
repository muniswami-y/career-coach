"""RAG layer: text extraction, chunking, embedding + retrieval via ChromaDB.

Chroma's default embedding model (all-MiniLM-L6-v2, ONNX) runs locally,
so no second API key is needed for embeddings.
"""
import io
import uuid

import chromadb
from pypdf import PdfReader


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


class DocumentStore:
    """One in-memory vector store per user session."""

    def __init__(self):
        self._client = chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection(f"docs_{uuid.uuid4().hex[:10]}")
        self.sources: set[str] = set()

    def add(self, source: str, text: str) -> int:
        chunks = chunk_text(text)
        if not chunks:
            return 0
        self._col.add(
            ids=[f"{source}-{uuid.uuid4().hex[:8]}" for _ in chunks],
            documents=chunks,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        self.sources.add(source)
        return len(chunks)

    def query(self, question: str, k: int = 4, source: str | None = None) -> list[dict]:
        if not self.sources:
            return []
        where = {"source": source} if source else None
        res = self._col.query(query_texts=[question], n_results=k, where=where)
        return [
            {"source": m["source"], "text": d}
            for d, m in zip(res["documents"][0], res["metadatas"][0])
        ]

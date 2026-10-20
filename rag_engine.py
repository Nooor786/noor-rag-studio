from __future__ import annotations

import html
import io
import re
from typing import Any

import numpy as np
import requests
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import cos_sim

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model; cached by Streamlit in app.py."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_reranker() -> CrossEncoder:
    """Load the optional second-stage cross-encoder reranker."""
    return CrossEncoder(RERANKER_MODEL_NAME)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w+#.]+\b", text.lower())


def normalize_scores(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(high, low):
        return np.zeros_like(values)
    return (values - low) / (high - low)


def split_into_chunks(text: str, chunk_size: int = 140, overlap: int = 35) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


class DocumentIndex:
    """In-memory hybrid index with page-aware PDF chunks."""

    def __init__(self, chunks: list[dict[str, Any]], model: SentenceTransformer,
                 reranker: CrossEncoder | None = None):
        if not chunks:
            raise ValueError("No readable text was found in the selected PDF files.")
        self.chunks = chunks
        self.model = model
        self.reranker = reranker
        self.embedding_dimension = 384
        self.tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.embeddings = np.asarray(
            model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True,
                         show_progress_bar=False), dtype=np.float32
        )
        if self.embeddings.ndim == 2 and self.embeddings.shape[1] > 0:
            self.embedding_dimension = int(self.embeddings.shape[1])
        self.document_summary = self._make_document_summary()

    @classmethod
    def from_pdfs(cls, documents: list[tuple[str, bytes]], embedding_model: SentenceTransformer,
                  chunk_size: int = 140, overlap: int = 35,
                  reranker: CrossEncoder | None = None) -> "DocumentIndex":
        all_chunks: list[dict[str, Any]] = []
        doc_stats: list[dict[str, Any]] = []
        for filename, pdf_bytes in documents:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
            except Exception as exc:
                raise ValueError(f"Could not open {filename}: {exc}") from exc
            if getattr(reader, "is_encrypted", False):
                try:
                    if reader.decrypt("") == 0:
                        raise ValueError(f"{filename} is password-protected. Upload an unlocked PDF.")
                except Exception as exc:
                    raise ValueError(f"Could not read encrypted PDF {filename}: {exc}") from exc
            page_count = len(reader.pages)
            doc_chunk_count = 0
            word_count = 0
            empty_pages = 0
            for page_index, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    empty_pages += 1
                    continue
                word_count += len(page_text.split())
                for chunk_number, text in enumerate(split_into_chunks(page_text, chunk_size, overlap), start=1):
                    source_id = f"S{len(all_chunks) + 1}"
                    all_chunks.append({
                        "source_id": source_id,
                        "filename": filename,
                        "page": page_index,
                        "chunk_number": chunk_number,
                        "text": text,
                    })
                    doc_chunk_count += 1
            warning = ""
            if empty_pages == page_count:
                warning = "No selectable text was found. This may be a scanned PDF and needs OCR."
            elif empty_pages:
                warning = f"{empty_pages} page(s) had no extractable text."
            doc_stats.append({
                "filename": filename,
                "pages": page_count,
                "chunks": doc_chunk_count,
                "words": word_count,
                "warning": warning,
            })
        obj = cls(all_chunks, embedding_model, reranker)
        obj.document_summary = doc_stats
        return obj

    def _make_document_summary(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in self.chunks:
            name = chunk["filename"]
            if name not in grouped:
                grouped[name] = {"filename": name, "pages": 0, "chunks": 0, "words": 0, "warning": ""}
            grouped[name]["pages"] = max(grouped[name]["pages"], int(chunk["page"]))
            grouped[name]["chunks"] += 1
            grouped[name]["words"] += len(chunk["text"].split())
        return list(grouped.values())

    def search(self, question: str, top_k: int = 4, use_reranker: bool = True) -> list[dict[str, Any]]:
        question = question.strip()
        if not question:
            return []
        if not self.chunks:
            return []
        # Lexical relevance (BM25)
        bm25_raw = np.asarray(self.bm25.get_scores(tokenize(question)), dtype=float)
        bm25_norm = normalize_scores(bm25_raw.tolist())
        # Semantic relevance (cosine similarity; embeddings are normalized)
        query_embedding = self.model.encode(question, normalize_embeddings=True, show_progress_bar=False)
        semantic_raw = np.asarray(cos_sim(query_embedding, self.embeddings)[0].cpu().tolist(), dtype=float)
        semantic_norm = normalize_scores(semantic_raw.tolist())
        # Hybrid score gives a balanced contribution to keyword and semantic retrieval.
        hybrid = (0.45 * bm25_norm) + (0.55 * semantic_norm)
        candidate_count = min(len(self.chunks), max(top_k * 3, 10))
        candidate_indices = np.argsort(hybrid)[::-1][:candidate_count].tolist()

        reranker_scores: dict[int, float] = {}
        if use_reranker and self.reranker is not None and candidate_indices:
            pairs = [(question, self.chunks[i]["text"]) for i in candidate_indices]
            raw_rerank = self.reranker.predict(pairs, show_progress_bar=False)
            reranker_scores = {i: float(score) for i, score in zip(candidate_indices, raw_rerank)}
            ranked = sorted(candidate_indices, key=lambda i: reranker_scores[i], reverse=True)
        else:
            ranked = sorted(candidate_indices, key=lambda i: float(hybrid[i]), reverse=True)

        results: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for idx in ranked:
            chunk = self.chunks[idx]
            normalized_text = re.sub(r"\s+", " ", chunk["text"]).strip().lower()
            if normalized_text in seen_text:
                continue
            seen_text.add(normalized_text)
            item = dict(chunk)
            item["score"] = float(hybrid[idx])
            item["bm25_score"] = float(bm25_norm[idx])
            item["semantic_score"] = float(semantic_norm[idx])
            item["reranker_score"] = reranker_scores.get(idx)
            results.append(item)
            if len(results) >= min(top_k, len(self.chunks)):
                break
        return results

def answer_with_ollama(
    question: str,
    results: list[dict[str, Any]],
    ollama_url: str = "http://localhost:11434",
    model_name: str = "llama3.2:3b",
) -> tuple[str, list[dict[str, Any]]]:

    import os

    if not results:
        return (
            "I couldn't find relevant passages in the selected documents. "
            "Try rephrasing your question or adding another PDF.",
            [],
        )

    context_parts = []
    sources = []

    for result in results:
        source_id = f"[{result['source_id']}]"

        context_parts.append(
            f"{source_id} File: {result['filename']} | "
            f"Page: {result['page']}\n{result['text']}"
        )

        sources.append({
            **result,
            "source_id": source_id,
            "text_html": html.escape(result["text"]).replace("\n", "<br>"),
        })

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a careful assistant answering questions about uploaded documents.
Use only the CONTEXT below. Do not invent facts.
If the context does not answer the question, say that the documents do not contain enough information.
Cite factual claims using source IDs such as [S1] or [S2].
Do not invent source IDs.
Be concise and clear.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with source citations):"""

    # Select cloud mode when configured; otherwise use local Ollama.
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        api_url = "https://api.groq.com/openai/v1/chat/completions"

        payload = {
            "model": os.getenv(
                "GROQ_MODEL",
                "llama-3.3-70b-versatile",
            ),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied document context. "
                        "Cite claims using the provided source IDs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        }

        service_name = "Groq"

    else:
        api_url = f"{ollama_url.rstrip('/')}/api/generate"

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        headers = {}
        service_name = "Ollama"

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError as exc:
        if service_name == "Groq":
            raise RuntimeError(
                "Cannot connect to Groq. Check your internet connection."
            ) from exc

        raise RuntimeError(
            "Cannot connect to Ollama. Check that Ollama is running."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"{service_name} took too long to respond. Please try again."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        detail = response.text[:500]

        if response.status_code in (401, 403) and service_name == "Groq":
            raise RuntimeError(
                "Groq authentication failed. Check GROQ_API_KEY."
            ) from exc

        raise RuntimeError(
            f"{service_name} returned HTTP {response.status_code}: {detail}"
        ) from exc

    response_data = response.json()

    if service_name == "Groq":
        answer = (
            response_data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    else:
        answer = response_data.get("response", "")

    answer = str(answer).strip()

    if not answer:
        answer = (
            "The model returned an empty answer. Please try again."
        )

    return answer, sources

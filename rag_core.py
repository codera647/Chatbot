from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from providers import ChatSettings, chat, env_value

load_dotenv()


RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented generation assistant. Answer only from the retrieved context. "
    "Cite sources using the provided [source | chunk] labels. If the answer is not in the context, say so."
)


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str
    strategy: str

    def label(self) -> str:
        return f"{self.source} | chunk {self.chunk_id}"


@dataclass
class TfidfModel:
    vocabulary: dict[str, int]
    idf: list[float]


@dataclass
class RagIndex:
    index_id: str
    chunks: list[Chunk]
    embedding_method: str
    embedding_model: str
    vector_store: str
    vectors: list[list[float]]
    tfidf_model: TfidfModel | None = None
    qdrant_client: Any | None = None
    qdrant_collection: str | None = None


RAG_INDEXES: dict[str, RagIndex] = {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def read_uploaded_file(path: str) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            return f"[Could not read PDF. pypdf is missing or broken: {exc}]"
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(errors="ignore")


def sentence_split(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def fixed_word_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def fixed_character_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(clean), step):
        chunk = clean[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(clean):
            break
    return chunks


def paragraph_chunks(text: str, max_words: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        count = len(paragraph.split())
        if current and current_words + count > max_words:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += count
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def sentence_window_chunks(text: str, window_size: int, overlap: int) -> list[str]:
    sentences = sentence_split(text)
    if not sentences:
        return []
    step = max(1, window_size - overlap)
    chunks = []
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + window_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + window_size >= len(sentences):
            break
    return chunks


def recursive_separator_chunks(text: str, max_words: int, overlap: int) -> list[str]:
    paragraphs = paragraph_chunks(text, max_words)
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph.split()) <= max_words:
            chunks.append(paragraph)
        else:
            chunks.extend(fixed_word_chunks(paragraph, max_words, overlap))
    return chunks


def chunk_document(text: str, strategy: str, chunk_size: int, overlap: int) -> list[str]:
    if strategy == "Fixed words":
        return fixed_word_chunks(text, chunk_size, overlap)
    if strategy == "Fixed characters":
        return fixed_character_chunks(text, chunk_size * 6, overlap * 6)
    if strategy == "Paragraph packing":
        return paragraph_chunks(text, chunk_size)
    if strategy == "Sentence window":
        return sentence_window_chunks(text, max(1, chunk_size // 35), max(0, overlap // 35))
    if strategy == "Recursive separators":
        return recursive_separator_chunks(text, chunk_size, overlap)
    return fixed_word_chunks(text, chunk_size, overlap)


def load_and_chunk_files(files, strategy: str, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file_obj in files or []:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        source = Path(path).name
        text = read_uploaded_file(path)
        for chunk_id, chunk_text in enumerate(chunk_document(text, strategy, chunk_size, overlap), start=1):
            if chunk_text.strip():
                chunks.append(Chunk(source=source, chunk_id=chunk_id, text=chunk_text, strategy=strategy))
    return chunks


def fit_tfidf(texts: list[str]) -> tuple[list[list[float]], TfidfModel]:
    documents = [tokenize(text) for text in texts]
    vocabulary = {token: idx for idx, token in enumerate(sorted({token for doc in documents for token in doc}))}
    doc_count = len(documents)
    df = Counter(token for doc in documents for token in set(doc))
    idf = [0.0] * len(vocabulary)
    for token, idx in vocabulary.items():
        idf[idx] = math.log((1 + doc_count) / (1 + df[token])) + 1

    vectors = [tfidf_vector_from_tokens(doc, vocabulary, idf) for doc in documents]
    return vectors, TfidfModel(vocabulary=vocabulary, idf=idf)


def tfidf_vector_from_tokens(tokens: list[str], vocabulary: dict[str, int], idf: list[float]) -> list[float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    vector = [0.0] * len(vocabulary)
    for token, count in counts.items():
        idx = vocabulary.get(token)
        if idx is not None:
            vector[idx] = (count / total) * idf[idx]
    return normalize_vector(vector)


def openai_embeddings(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    from openai import OpenAI

    key = api_key.strip() or env_value("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing.")
    client = OpenAI(api_key=key)
    response = client.embeddings.create(model=model, input=texts)
    return [normalize_vector(item.embedding) for item in response.data]


def ollama_embeddings(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    url = (base_url or "http://localhost:11434").rstrip("/")
    response = requests.post(f"{url}/api/embed", json={"model": model, "input": texts}, timeout=120)
    if response.ok:
        data = response.json()
        vectors = data.get("embeddings")
        if vectors:
            return [normalize_vector([float(value) for value in vector]) for vector in vectors]

    vectors = []
    for text in texts:
        single = requests.post(f"{url}/api/embeddings", json={"model": model, "prompt": text}, timeout=120)
        single.raise_for_status()
        vectors.append(normalize_vector([float(value) for value in single.json()["embedding"]]))
    return vectors


def huggingface_embeddings(texts: list[str], model: str, token: str) -> list[list[float]]:
    from huggingface_hub import InferenceClient

    key = token.strip() or env_value("HF_TOKEN")
    if not key:
        raise ValueError("HF_TOKEN is missing.")
    client = InferenceClient(model=model, token=key)
    vectors = []
    for text in texts:
        output = client.feature_extraction(text)
        if output and isinstance(output[0], list):
            vector = [sum(values) / len(values) for values in zip(*output)]
        else:
            vector = output
        vectors.append(normalize_vector([float(value) for value in vector]))
    return vectors


def embed_corpus(
    texts: list[str],
    method: str,
    model: str,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
) -> tuple[list[list[float]], TfidfModel | None]:
    if method == "TF-IDF keyword vectors":
        vectors, tfidf_model = fit_tfidf(texts)
        return vectors, tfidf_model
    if method == "OpenAI embeddings":
        return openai_embeddings(texts, model, openai_key), None
    if method == "Ollama local embeddings":
        return ollama_embeddings(texts, model, ollama_base_url), None
    if method == "Hugging Face Inference embeddings":
        return huggingface_embeddings(texts, model, hf_token), None
    raise ValueError(f"Unknown embedding method: {method}")


def embed_query(
    query: str,
    index: RagIndex,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
) -> list[float]:
    if index.embedding_method == "TF-IDF keyword vectors":
        if not index.tfidf_model:
            raise ValueError("TF-IDF model is missing from the index.")
        return tfidf_vector_from_tokens(tokenize(query), index.tfidf_model.vocabulary, index.tfidf_model.idf)
    if index.embedding_method == "OpenAI embeddings":
        return openai_embeddings([query], index.embedding_model, openai_key)[0]
    if index.embedding_method == "Ollama local embeddings":
        return ollama_embeddings([query], index.embedding_model, ollama_base_url)[0]
    if index.embedding_method == "Hugging Face Inference embeddings":
        return huggingface_embeddings([query], index.embedding_model, hf_token)[0]
    raise ValueError(f"Unknown embedding method: {index.embedding_method}")


def qdrant_client_for(vector_store: str, qdrant_url: str, qdrant_api_key: str):
    from qdrant_client import QdrantClient

    if vector_store == "Qdrant local in-memory":
        return QdrantClient(":memory:")
    if vector_store == "Qdrant Cloud":
        if not qdrant_url.strip():
            raise ValueError("Qdrant Cloud URL is required.")
        return QdrantClient(url=qdrant_url.strip(), api_key=qdrant_api_key.strip() or None)
    return None


def store_in_qdrant(index: RagIndex, qdrant_url: str, qdrant_api_key: str) -> RagIndex:
    from qdrant_client import models

    if not index.vectors:
        return index

    client = qdrant_client_for(index.vector_store, qdrant_url, qdrant_api_key)
    collection_name = f"rag_lab_{index.index_id.replace('-', '_')}"
    vector_size = len(index.vectors[0])
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    except AttributeError:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    points = []
    for point_id, (chunk, vector) in enumerate(zip(index.chunks, index.vectors), start=1):
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "strategy": chunk.strategy,
                    "text": chunk.text,
                },
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    index.qdrant_client = client
    index.qdrant_collection = collection_name
    return index


def build_index(
    files,
    chunk_strategy: str,
    chunk_size: int,
    overlap: int,
    embedding_method: str,
    embedding_model: str,
    vector_store: str,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
    qdrant_url: str,
    qdrant_api_key: str,
) -> tuple[str, str, str]:
    if not files:
        return "", "Upload documents before building the index.", ""

    chunks = load_and_chunk_files(files, chunk_strategy, int(chunk_size), int(overlap))
    if not chunks:
        return "", "No readable text chunks were created.", ""

    texts = [chunk.text for chunk in chunks]
    vectors, tfidf_model = embed_corpus(texts, embedding_method, embedding_model, openai_key, hf_token, ollama_base_url)
    index = RagIndex(
        index_id=str(uuid.uuid4()),
        chunks=chunks,
        embedding_method=embedding_method,
        embedding_model=embedding_model,
        vector_store=vector_store,
        vectors=vectors,
        tfidf_model=tfidf_model,
    )

    if vector_store in {"Qdrant local in-memory", "Qdrant Cloud"}:
        index = store_in_qdrant(index, qdrant_url, qdrant_api_key)

    RAG_INDEXES[index.index_id] = index
    sources = sorted({chunk.source for chunk in chunks})
    status = (
        f"Built index {index.index_id[:8]} with {len(chunks)} chunks from {len(sources)} file(s).\n"
        f"Chunking: {chunk_strategy}. Embeddings: {embedding_method} ({embedding_model}). Store: {vector_store}."
    )
    return index.index_id, status, "\n".join(f"- {source}" for source in sources)


def qdrant_search(index: RagIndex, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    client = index.qdrant_client
    collection = index.qdrant_collection
    if not client or not collection:
        raise ValueError("Qdrant index is not initialized.")
    try:
        response = client.query_points(collection_name=collection, query=query_vector, limit=top_k)
        points = response.points
    except Exception:
        points = client.search(collection_name=collection, query_vector=query_vector, limit=top_k)

    results = []
    for point in points:
        payload = point.payload or {}
        results.append(
            {
                "score": float(point.score),
                "source": payload.get("source", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "text": payload.get("text", ""),
            }
        )
    return results


def memory_search(index: RagIndex, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    scored = []
    for chunk, vector in zip(index.chunks, index.vectors):
        scored.append(
            {
                "score": cosine_similarity(query_vector, vector),
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def retrieve(
    index_id: str,
    query: str,
    top_k: int,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
) -> list[dict[str, Any]]:
    index = RAG_INDEXES.get(index_id)
    if not index:
        raise ValueError("Build or select an index first.")
    query_vector = embed_query(query, index, openai_key, hf_token, ollama_base_url)
    if index.vector_store in {"Qdrant local in-memory", "Qdrant Cloud"}:
        return qdrant_search(index, query_vector, int(top_k))
    return memory_search(index, query_vector, int(top_k))


def format_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No chunks retrieved."
    blocks = []
    for idx, result in enumerate(results, start=1):
        blocks.append(
            f"[{idx}] {result['source']} | chunk {result['chunk_id']} | score {result['score']:.3f}\n"
            f"{result['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(question: str, results: list[dict[str, Any]]) -> str:
    return (
        "Retrieved context:\n"
        f"{format_context(results)}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations to the retrieved chunks."
    )


def answer_with_rag(
    index_id: str,
    question: str,
    history: list[dict[str, str]] | None,
    top_k: int,
    generator_provider: str,
    generator_model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
) -> tuple[str, list[dict[str, str]], str, str]:
    history = history or []
    results = retrieve(index_id, question, top_k, openai_key, hf_token, ollama_base_url)
    rag_prompt = build_rag_prompt(question, results)
    settings = ChatSettings(
        provider=generator_provider,
        model=generator_model,
        system_prompt=system_prompt.strip() or RAG_SYSTEM_PROMPT,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        openai_api_key=openai_key.strip(),
        hf_token=hf_token.strip(),
        ollama_base_url=ollama_base_url.strip() or "http://localhost:11434",
    )
    answer = chat(settings, rag_prompt, history)
    next_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    payload = {
        "index_id": index_id,
        "retrieval_top_k": int(top_k),
        "generator_provider": generator_provider,
        "generator_model": generator_model,
        "messages": [
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": rag_prompt},
        ],
    }
    return answer, next_history, format_context(results), json.dumps(payload, indent=2)


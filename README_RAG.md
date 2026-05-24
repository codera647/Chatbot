# RAG Lab

Separate classroom app for Retrieval-Augmented Generation.

## Run

```powershell
cd D:\Chatbot
.\.venv\Scripts\Activate.ps1
python rag_app.py
```

The RAG app starts on the first free port from `7870` to `7899`.

## What It Demonstrates

- Chunking techniques:
  - Fixed words
  - Fixed characters
  - Paragraph packing
  - Sentence window
  - Recursive separators

- Embedding techniques:
  - Built-in TF-IDF keyword vectors
  - OpenAI embeddings
  - Ollama local embeddings
  - Hugging Face Inference embeddings

- Vector stores:
  - Python memory
  - Qdrant local in-memory
  - Qdrant Cloud

## Suggested Demo

1. Upload a short `.txt`, `.md`, or `.pdf`.
2. Build an index with `Fixed words`, `TF-IDF keyword vectors`, and `Python memory`.
3. Ask a question and click `Retrieve only`.
4. Change chunking to `Sentence window`, rebuild, and compare retrieved chunks.
5. Change embeddings to `OpenAI embeddings`, rebuild, and compare.
6. Change vector store to `Qdrant local in-memory`, rebuild, and explain vector DB search.
7. Use `Qdrant Cloud` only when you have a URL and API key.

## Ollama Embeddings

For local embeddings, pull an embedding model first:

```powershell
ollama pull nomic-embed-text
```

Then choose:

```text
Embedding technique: Ollama local embeddings
Embedding model: nomic-embed-text
```

## Qdrant Cloud

Use:

```text
Vector store: Qdrant Cloud
Qdrant Cloud URL: https://your-cluster-url
Qdrant API key: your-key
```

The app creates a temporary collection for each index.


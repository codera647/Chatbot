from __future__ import annotations

import json
import math
import os
import re
import socket
from collections import Counter
from pathlib import Path

import gradio as gr

from providers import (
    DEFAULT_SYSTEM_PROMPT,
    ChatSettings,
    chat,
    env_value,
    health_report,
    ollama_status,
    payload_preview,
)


OPENAI_DEFAULT_MODEL = env_value("OPENAI_MODEL", "gpt-5-mini")
HF_DEFAULT_MODEL = env_value("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
OLLAMA_DEFAULT_URL = env_value("OLLAMA_BASE_URL", "http://localhost:11434")


def get_default_ollama_model() -> str:
    configured = env_value("OLLAMA_MODEL")
    if configured:
        return configured
    status = ollama_status(OLLAMA_DEFAULT_URL)
    models = status.get("models") if status.get("running") else None
    if models:
        return str(models[0])
    return "llama3.2"


OLLAMA_DEFAULT_MODEL = get_default_ollama_model()


CSS = """
.app-shell {max-width: 1180px; margin: 0 auto;}
.compact-code textarea {font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;}
.status-box textarea {font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;}
.small-note {font-size: 0.92rem; opacity: 0.84;}
"""


def find_free_port(start: int = 7860, end: int = 7899) -> int:
    requested = os.getenv("GRADIO_SERVER_PORT")
    if requested:
        return int(requested)

    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found from {start} to {end}.")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def read_text_file(path: str) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            return "[PDF support needs pypdf. Run: python -m pip install pypdf]"
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(errors="ignore")


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def build_rag_index(files, chunk_words: int, overlap_words: int):
    if not files:
        return [], "Upload one or more documents first.", ""

    index = []
    total_words = 0
    for file_obj in files:
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        source = Path(path).name
        text = read_text_file(path)
        if not text.strip():
            continue
        chunks = chunk_text(text, int(chunk_words), int(overlap_words))
        total_words += len(text.split())
        for chunk_id, chunk in enumerate(chunks, start=1):
            tokens = tokenize(chunk)
            index.append(
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "tokens": dict(Counter(tokens)),
                    "token_count": len(tokens),
                }
            )

    if not index:
        return [], "No readable text was found in the uploaded files.", ""

    sources = sorted({item["source"] for item in index})
    status = (
        f"Indexed {len(index)} chunks from {len(sources)} file(s). "
        f"Approximate source words: {total_words}."
    )
    preview = "\n".join(f"- {source}" for source in sources)
    return index, status, preview


def score_chunk(query_counts: Counter, chunk: dict) -> float:
    if not query_counts or not chunk.get("tokens"):
        return 0.0

    chunk_counts = Counter(chunk["tokens"])
    dot = sum(query_counts[token] * chunk_counts.get(token, 0) for token in query_counts)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    chunk_norm = math.sqrt(sum(value * value for value in chunk_counts.values()))
    if query_norm == 0 or chunk_norm == 0:
        return 0.0
    return dot / (query_norm * chunk_norm)


def retrieve_chunks(query: str, index: list[dict], top_k: int) -> list[dict]:
    query_counts = Counter(tokenize(query))
    scored = []
    for chunk in index or []:
        score = score_chunk(query_counts, chunk)
        if score > 0:
            scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: int(top_k)]


def format_retrieved_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No matching chunks found. Try a question that uses words from the uploaded document."

    parts = []
    for position, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{position}] {chunk['source']} | chunk {chunk['chunk_id']} | score {chunk['score']:.3f}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context = format_retrieved_chunks(chunks)
    return (
        "Use the retrieved context to answer the question. "
        "If the context does not contain the answer, say that the uploaded documents do not provide enough information.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Question: {question}"
    )


def append_turn(
    message: str,
    history: list[dict[str, str]] | None,
    provider: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
):
    history = history or []
    settings = ChatSettings(
        provider=provider,
        model=model.strip(),
        system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        openai_api_key=openai_key.strip(),
        hf_token=hf_token.strip(),
        ollama_base_url=ollama_base_url.strip() or OLLAMA_DEFAULT_URL,
    )
    answer = chat(settings, message, history)
    next_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    preview = payload_preview(settings, history, message)
    return "", next_history, next_history, preview


def clear_chat():
    return [], [], ""


def default_model_for_provider(provider: str):
    defaults = {
        "OpenAI via LangChain": OPENAI_DEFAULT_MODEL,
        "Hugging Face Inference API": HF_DEFAULT_MODEL,
        "Ollama Local": OLLAMA_DEFAULT_MODEL,
    }
    return defaults.get(provider, OPENAI_DEFAULT_MODEL)


def preview_payload(
    provider: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
    message: str,
    history: list[dict[str, str]] | None,
):
    settings = ChatSettings(
        provider=provider,
        model=model.strip(),
        system_prompt=system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        openai_api_key=openai_key.strip(),
        hf_token=hf_token.strip(),
        ollama_base_url=ollama_base_url.strip() or OLLAMA_DEFAULT_URL,
    )
    return payload_preview(settings, history, message)


def refresh_ollama_models(base_url: str):
    status = ollama_status(base_url.strip() or OLLAMA_DEFAULT_URL)
    if not status.get("running"):
        return gr.update(choices=[OLLAMA_DEFAULT_MODEL], value=OLLAMA_DEFAULT_MODEL), str(status)
    models = status.get("models") or [OLLAMA_DEFAULT_MODEL]
    value = models[0] if models else OLLAMA_DEFAULT_MODEL
    return gr.update(choices=models, value=value), str(status)


def rag_append_turn(
    message: str,
    history: list[dict[str, str]] | None,
    index: list[dict] | None,
    provider: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
):
    history = history or []
    if not index:
        return "", history, history, "Build a RAG index before asking a question.", ""

    chunks = retrieve_chunks(message, index, int(top_k))
    retrieved_text = format_retrieved_chunks(chunks)
    rag_message = build_rag_prompt(message, chunks)
    rag_system_prompt = (
        system_prompt.strip()
        or "You are a RAG assistant. Answer only from retrieved context and cite source chunk numbers."
    )

    settings = ChatSettings(
        provider=provider,
        model=model.strip(),
        system_prompt=rag_system_prompt,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        openai_api_key=openai_key.strip(),
        hf_token=hf_token.strip(),
        ollama_base_url=ollama_base_url.strip() or OLLAMA_DEFAULT_URL,
    )
    answer = chat(settings, rag_message, history)
    next_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    preview = payload_preview(settings, history, rag_message)
    return "", next_history, next_history, retrieved_text, preview


def rag_preview_payload(
    provider: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    openai_key: str,
    hf_token: str,
    ollama_base_url: str,
    message: str,
    history: list[dict[str, str]] | None,
    index: list[dict] | None,
):
    chunks = retrieve_chunks(message, index or [], int(top_k))
    rag_message = build_rag_prompt(message or "<student question>", chunks)
    settings = ChatSettings(
        provider=provider,
        model=model.strip(),
        system_prompt=system_prompt.strip()
        or "You are a RAG assistant. Answer only from retrieved context and cite source chunk numbers.",
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        openai_api_key=openai_key.strip(),
        hf_token=hf_token.strip(),
        ollama_base_url=ollama_base_url.strip() or OLLAMA_DEFAULT_URL,
    )
    return format_retrieved_chunks(chunks), payload_preview(settings, history, rag_message)


def build_chat_lab(
    provider_choices: list[str],
    default_provider: str,
    default_model: str,
    default_base_url: str = OLLAMA_DEFAULT_URL,
):
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="Chat",
                height=460,
                buttons=["copy", "copy_all"],
                placeholder="Start the lab conversation here.",
            )
            message = gr.Textbox(
                label="Message",
                placeholder="Ask the model a question...",
                lines=3,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
                preview = gr.Button("Preview payload")

        with gr.Column(scale=5):
            provider = gr.Dropdown(
                provider_choices,
                value=default_provider,
                label="Provider",
                interactive=len(provider_choices) > 1,
            )
            if provider_choices == ["Ollama Local"]:
                model = gr.Dropdown(
                    choices=[default_model],
                    value=default_model,
                    label="Model",
                    allow_custom_value=True,
                )
            else:
                model = gr.Dropdown(
                    choices=[
                        OPENAI_DEFAULT_MODEL,
                        "gpt-5.1",
                        "gpt-4o-mini",
                        HF_DEFAULT_MODEL,
                        "mistralai/Mistral-7B-Instruct-v0.3",
                        "HuggingFaceH4/zephyr-7b-beta",
                    ],
                    value=default_model,
                    label="Model",
                    allow_custom_value=True,
                )
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                label="System prompt",
                lines=4,
            )
            with gr.Row():
                temperature = gr.Slider(0, 1.5, value=0.3, step=0.1, label="Temperature")
                max_tokens = gr.Slider(32, 2048, value=512, step=32, label="Max tokens")
            openai_key = gr.Textbox(
                label="OpenAI API key",
                type="password",
                placeholder="Uses OPENAI_API_KEY from .env when empty",
                visible="OpenAI via LangChain" in provider_choices,
            )
            hf_token = gr.Textbox(
                label="Hugging Face token",
                type="password",
                placeholder="Uses HF_TOKEN from .env when empty",
                visible="Hugging Face Inference API" in provider_choices,
            )
            ollama_base_url = gr.Textbox(
                value=default_base_url,
                label="Ollama base URL",
                visible="Ollama Local" in provider_choices,
            )
            payload = gr.Code(
                label="Under the hood",
                language="json",
                lines=14,
                elem_classes=["compact-code"],
            )

    inputs = [
        message,
        state,
        provider,
        model,
        system_prompt,
        temperature,
        max_tokens,
        openai_key,
        hf_token,
        ollama_base_url,
    ]
    outputs = [message, chatbot, state, payload]

    send.click(append_turn, inputs=inputs, outputs=outputs)
    message.submit(append_turn, inputs=inputs, outputs=outputs)
    clear.click(clear_chat, outputs=[chatbot, state, payload])
    provider.change(default_model_for_provider, inputs=provider, outputs=model)
    preview.click(
        preview_payload,
        inputs=[
            provider,
            model,
            system_prompt,
            temperature,
            max_tokens,
            openai_key,
            hf_token,
            ollama_base_url,
            message,
            state,
        ],
        outputs=payload,
    )

    return {
        "state": state,
        "chatbot": chatbot,
        "provider": provider,
        "model": model,
        "ollama_base_url": ollama_base_url,
    }


def build_rag_lab():
    history_state = gr.State([])
    index_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=5):
            files = gr.File(
                label="Documents",
                file_count="multiple",
                file_types=[".txt", ".md", ".csv", ".json", ".py", ".pdf"],
            )
            with gr.Row():
                chunk_words = gr.Slider(80, 500, value=180, step=20, label="Chunk words")
                overlap_words = gr.Slider(0, 120, value=40, step=10, label="Overlap words")
            build_index = gr.Button("Build RAG index", variant="primary")
            index_status = gr.Textbox(label="Index status", lines=2)
            sources = gr.Textbox(label="Sources", lines=5)
            retrieved = gr.Textbox(label="Retrieved chunks", lines=16)

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="RAG Chat",
                height=430,
                buttons=["copy", "copy_all"],
                placeholder="Ask questions about the uploaded documents.",
            )
            question = gr.Textbox(label="Question", placeholder="Ask about the documents...", lines=3)
            with gr.Row():
                send = gr.Button("Ask with RAG", variant="primary")
                clear = gr.Button("Clear chat")
                preview = gr.Button("Preview RAG payload")

            with gr.Row():
                provider = gr.Dropdown(
                    ["OpenAI via LangChain", "Hugging Face Inference API", "Ollama Local"],
                    value="OpenAI via LangChain",
                    label="Generator",
                )
                model = gr.Dropdown(
                    [
                        OPENAI_DEFAULT_MODEL,
                        "gpt-4o-mini",
                        HF_DEFAULT_MODEL,
                        "mistralai/Mistral-7B-Instruct-v0.3",
                        OLLAMA_DEFAULT_MODEL,
                    ],
                    value=OPENAI_DEFAULT_MODEL,
                    label="Model",
                    allow_custom_value=True,
                )
                top_k = gr.Slider(1, 6, value=3, step=1, label="Top K")
            system_prompt = gr.Textbox(
                value="You are a RAG assistant. Answer only from retrieved context and cite source chunk numbers.",
                label="System prompt",
                lines=3,
            )
            with gr.Row():
                temperature = gr.Slider(0, 1.5, value=0.2, step=0.1, label="Temperature")
                max_tokens = gr.Slider(64, 2048, value=700, step=32, label="Max tokens")
            with gr.Row():
                openai_key = gr.Textbox(
                    label="OpenAI API key",
                    type="password",
                    placeholder="Uses OPENAI_API_KEY from .env when empty",
                )
                hf_token = gr.Textbox(
                    label="Hugging Face token",
                    type="password",
                    placeholder="Uses HF_TOKEN from .env when empty",
                )
            ollama_base_url = gr.Textbox(value=OLLAMA_DEFAULT_URL, label="Ollama base URL")
            payload = gr.Code(label="RAG payload", language="json", lines=14)

    build_index.click(
        build_rag_index,
        inputs=[files, chunk_words, overlap_words],
        outputs=[index_state, index_status, sources],
    )
    clear.click(clear_chat, outputs=[chatbot, history_state, payload]).then(lambda: "", outputs=retrieved)
    provider.change(default_model_for_provider, inputs=provider, outputs=model)

    inputs = [
        question,
        history_state,
        index_state,
        provider,
        model,
        system_prompt,
        temperature,
        max_tokens,
        top_k,
        openai_key,
        hf_token,
        ollama_base_url,
    ]
    outputs = [question, chatbot, history_state, retrieved, payload]
    send.click(rag_append_turn, inputs=inputs, outputs=outputs)
    question.submit(rag_append_turn, inputs=inputs, outputs=outputs)
    preview.click(
        rag_preview_payload,
        inputs=[
            provider,
            model,
            system_prompt,
            temperature,
            max_tokens,
            top_k,
            openai_key,
            hf_token,
            ollama_base_url,
            question,
            history_state,
            index_state,
        ],
        outputs=[retrieved, payload],
    )


with gr.Blocks(
    title="LLM Chatbot Labs",
) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.Markdown(
            "# LLM Chatbot Labs\n"
            "OpenAI + LangChain, Hugging Face, and Ollama in one classroom app."
        )

        with gr.Tabs():
            with gr.Tab("Lab 1: Cloud LLMs"):
                cloud = build_chat_lab(
                    ["OpenAI via LangChain", "Hugging Face Inference API"],
                    "OpenAI via LangChain",
                    OPENAI_DEFAULT_MODEL,
                )
                with gr.Accordion("Demo prompts", open=False):
                    gr.Markdown(
                        "- Explain tokenization using a two-sentence example.\n"
                        "- Act as a chatbot for a university admissions office.\n"
                        "- Compare system, user, and assistant messages.\n"
                        "- Make the same answer more formal, then more concise."
                    )

            with gr.Tab("Lab 2: Local LLM with Ollama"):
                local = build_chat_lab(
                    ["Ollama Local"],
                    "Ollama Local",
                    OLLAMA_DEFAULT_MODEL,
                    OLLAMA_DEFAULT_URL,
                )
                with gr.Row():
                    refresh_models = gr.Button("Refresh Ollama models")
                    ollama_report = gr.Textbox(label="Ollama status", lines=4)
                refresh_models.click(
                    refresh_ollama_models,
                    inputs=local["ollama_base_url"],
                    outputs=[local["model"], ollama_report],
                )
                with gr.Accordion("Local model commands", open=False):
                    gr.Code(
                        "ollama pull llama3.2\n"
                        "ollama serve\n"
                        "ollama list",
                        language="shell",
                        label="Terminal",
                    )

            with gr.Tab("Lab 3: RAG"):
                build_rag_lab()
                with gr.Accordion("RAG teaching script", open=False):
                    gr.Markdown(
                        "1. Upload a small document.\n"
                        "2. Build the index to split it into overlapping chunks.\n"
                        "3. Ask a question and inspect the retrieved chunks.\n"
                        "4. Show that the LLM answer is generated from retrieved context, not from the model's memory."
                    )

            with gr.Tab("Setup Check"):
                status = gr.Code(
                    value="Click Refresh status to check packages, API keys, and Ollama.",
                    language="json",
                    label="Runtime status",
                    lines=24,
                    elem_classes=["status-box"],
                )
                refresh_status = gr.Button("Refresh status")
                refresh_status.click(health_report, outputs=status)


if __name__ == "__main__":
    server_port = find_free_port()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=server_port,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        ssr_mode=False,
    )

from __future__ import annotations

import os
import socket

import gradio as gr

from providers import env_value
from rag_core import RAG_SYSTEM_PROMPT, answer_with_rag, build_index, retrieve, format_context


OPENAI_DEFAULT_MODEL = env_value("OPENAI_MODEL", "gpt-4o-mini")
HF_DEFAULT_MODEL = env_value("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
OLLAMA_DEFAULT_URL = env_value("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = env_value("OLLAMA_MODEL", "llama3.2")


CSS = """
.app-shell {max-width: 1240px; margin: 0 auto;}
.mono textarea {font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;}
"""


def find_free_port(start: int = 7870, end: int = 7899) -> int:
    requested = os.getenv("GRADIO_SERVER_PORT")
    if requested:
        return int(requested)
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found from {start} to {end}.")


def default_embedding_model(method: str) -> str:
    defaults = {
        "TF-IDF keyword vectors": "built-in-tfidf",
        "OpenAI embeddings": "text-embedding-3-small",
        "Ollama local embeddings": "nomic-embed-text",
        "Hugging Face Inference embeddings": "sentence-transformers/all-MiniLM-L6-v2",
    }
    return defaults.get(method, "built-in-tfidf")


def default_generator_model(provider: str) -> str:
    defaults = {
        "OpenAI via LangChain": OPENAI_DEFAULT_MODEL,
        "Hugging Face Inference API": HF_DEFAULT_MODEL,
        "Ollama Local": OLLAMA_CHAT_MODEL,
    }
    return defaults.get(provider, OPENAI_DEFAULT_MODEL)


def build_index_ui(
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
):
    try:
        return build_index(
            files,
            chunk_strategy,
            int(chunk_size),
            int(overlap),
            embedding_method,
            embedding_model,
            vector_store,
            openai_key,
            hf_token,
            ollama_base_url,
            qdrant_url,
            qdrant_api_key,
        )
    except Exception as exc:
        return "", f"Index build failed: {exc}", ""


def retrieve_only_ui(index_id: str, question: str, top_k: int, openai_key: str, hf_token: str, ollama_base_url: str):
    try:
        results = retrieve(index_id, question, int(top_k), openai_key, hf_token, ollama_base_url)
        return format_context(results)
    except Exception as exc:
        return f"Retrieval failed: {exc}"


def ask_ui(
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
):
    if not question.strip():
        return "", history or [], history or [], "Type a question first.", ""
    try:
        _, next_history, context, payload = answer_with_rag(
            index_id,
            question,
            history,
            int(top_k),
            generator_provider,
            generator_model,
            system_prompt,
            float(temperature),
            int(max_tokens),
            openai_key,
            hf_token,
            ollama_base_url,
        )
        return "", next_history, next_history, context, payload
    except Exception as exc:
        current = history or []
        next_history = current + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"RAG failed: {exc}"},
        ]
        return "", next_history, next_history, "", ""


def clear_chat():
    return [], [], "", ""


with gr.Blocks(title="RAG Lab") as demo:
    chat_state = gr.State([])
    index_state = gr.State("")

    with gr.Column(elem_classes=["app-shell"]):
        gr.Markdown("# RAG Lab\nCompare chunking, embeddings, vector stores, retrieval, and generation.")

        with gr.Row():
            with gr.Column(scale=5):
                files = gr.File(
                    label="Documents",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".csv", ".json", ".py", ".pdf"],
                )
                chunk_strategy = gr.Dropdown(
                    ["Fixed words", "Fixed characters", "Paragraph packing", "Sentence window", "Recursive separators"],
                    value="Fixed words",
                    label="Chunking technique",
                )
                with gr.Row():
                    chunk_size = gr.Slider(40, 700, value=180, step=20, label="Chunk size")
                    overlap = gr.Slider(0, 200, value=40, step=10, label="Overlap")

                embedding_method = gr.Dropdown(
                    [
                        "TF-IDF keyword vectors",
                        "OpenAI embeddings",
                        "Ollama local embeddings",
                        "Hugging Face Inference embeddings",
                    ],
                    value="TF-IDF keyword vectors",
                    label="Embedding technique",
                )
                embedding_model = gr.Textbox(value="built-in-tfidf", label="Embedding model")

                vector_store = gr.Dropdown(
                    ["Python memory", "Qdrant local in-memory", "Qdrant Cloud"],
                    value="Python memory",
                    label="Vector store",
                )
                qdrant_url = gr.Textbox(label="Qdrant Cloud URL", placeholder="https://xxxxx.region.cloud.qdrant.io")
                qdrant_api_key = gr.Textbox(label="Qdrant API key", type="password")

                with gr.Row():
                    openai_key = gr.Textbox(label="OpenAI key", type="password", placeholder="Uses OPENAI_API_KEY when empty")
                    hf_token = gr.Textbox(label="HF token", type="password", placeholder="Uses HF_TOKEN when empty")
                ollama_base_url = gr.Textbox(value=OLLAMA_DEFAULT_URL, label="Ollama base URL")

                build = gr.Button("Build index", variant="primary")
                index_id = gr.Textbox(label="Index ID")
                index_status = gr.Textbox(label="Index status", lines=4)
                sources = gr.Textbox(label="Sources", lines=5)

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="RAG Chat", height=420, buttons=["copy", "copy_all"])
                question = gr.Textbox(label="Question", placeholder="Ask a question about the uploaded documents...", lines=3)
                with gr.Row():
                    top_k = gr.Slider(1, 8, value=3, step=1, label="Top K retrieved chunks")
                    retrieve_button = gr.Button("Retrieve only")
                    ask = gr.Button("Ask with RAG", variant="primary")
                    clear = gr.Button("Clear")

                with gr.Row():
                    generator_provider = gr.Dropdown(
                        ["OpenAI via LangChain", "Hugging Face Inference API", "Ollama Local"],
                        value="OpenAI via LangChain",
                        label="Generator",
                    )
                    generator_model = gr.Textbox(value=OPENAI_DEFAULT_MODEL, label="Generator model")
                with gr.Row():
                    temperature = gr.Slider(0, 1.5, value=0.2, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(64, 2048, value=700, step=32, label="Max tokens")
                system_prompt = gr.Textbox(value=RAG_SYSTEM_PROMPT, label="System prompt", lines=3)

                retrieved_context = gr.Textbox(label="Retrieved chunks", lines=14)
                payload = gr.Code(label="RAG payload", language="json", lines=14, elem_classes=["mono"])

        with gr.Accordion("Classroom demo sequence", open=False):
            gr.Markdown(
                "1. Build with TF-IDF + Python memory to show the simplest possible retriever.\n"
                "2. Change chunking from fixed words to sentence windows and compare retrieved chunks.\n"
                "3. Switch to OpenAI, Ollama, or Hugging Face embeddings and rebuild the index.\n"
                "4. Switch vector store from Python memory to Qdrant local in-memory.\n"
                "5. Use Qdrant Cloud when you want to show a managed open-source vector DB."
            )

    embedding_method.change(default_embedding_model, inputs=embedding_method, outputs=embedding_model)
    generator_provider.change(default_generator_model, inputs=generator_provider, outputs=generator_model)
    build.click(
        build_index_ui,
        inputs=[
            files,
            chunk_strategy,
            chunk_size,
            overlap,
            embedding_method,
            embedding_model,
            vector_store,
            openai_key,
            hf_token,
            ollama_base_url,
            qdrant_url,
            qdrant_api_key,
        ],
        outputs=[index_state, index_status, sources],
    ).then(lambda value: value, inputs=index_state, outputs=index_id)

    retrieve_button.click(
        retrieve_only_ui,
        inputs=[index_state, question, top_k, openai_key, hf_token, ollama_base_url],
        outputs=retrieved_context,
    )

    ask_inputs = [
        index_state,
        question,
        chat_state,
        top_k,
        generator_provider,
        generator_model,
        system_prompt,
        temperature,
        max_tokens,
        openai_key,
        hf_token,
        ollama_base_url,
    ]
    ask_outputs = [question, chatbot, chat_state, retrieved_context, payload]
    ask.click(ask_ui, inputs=ask_inputs, outputs=ask_outputs)
    question.submit(ask_ui, inputs=ask_inputs, outputs=ask_outputs)
    clear.click(clear_chat, outputs=[chatbot, chat_state, retrieved_context, payload])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=find_free_port(),
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
        css=CSS,
        ssr_mode=False,
    )


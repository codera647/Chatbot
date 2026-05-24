from __future__ import annotations

import os
import socket

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

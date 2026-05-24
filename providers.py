from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


def clear_dead_local_proxy() -> None:
    proxy_names = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for name in proxy_names:
        value = os.getenv(name, "")
        if "127.0.0.1:9" in value or "localhost:9" in value:
            os.environ.pop(name, None)


clear_dead_local_proxy()


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful teaching assistant. Explain clearly, use examples, "
    "and keep answers suitable for a master's level AI class."
)


@dataclass(frozen=True)
class ChatSettings:
    provider: str
    model: str
    system_prompt: str
    temperature: float
    max_tokens: int
    openai_api_key: str = ""
    hf_token: str = ""
    ollama_base_url: str = "http://localhost:11434"


def env_value(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def package_status() -> dict[str, str]:
    packages = [
        "gradio",
        "langchain",
        "langchain-openai",
        "langchain-huggingface",
        "langchain-ollama",
        "openai",
        "huggingface-hub",
        "python-dotenv",
        "requests",
    ]
    status: dict[str, str] = {}
    for package in packages:
        try:
            status[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            status[package] = "missing"
    return status


def health_report() -> str:
    report = {
        "environment": {
            "OPENAI_API_KEY": "set" if env_value("OPENAI_API_KEY") else "missing",
            "HF_TOKEN": "set" if env_value("HF_TOKEN") else "missing",
            "OLLAMA_BASE_URL": env_value("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
        "packages": package_status(),
        "ollama": ollama_status(env_value("OLLAMA_BASE_URL", "http://localhost:11434")),
    }
    return json.dumps(report, indent=2)


def ollama_status(base_url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        response.raise_for_status()
        models = [model.get("name", "") for model in response.json().get("models", [])]
        return {"running": True, "models": models}
    except Exception as exc:
        return {"running": False, "error": str(exc)}


def normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    cleaned: list[dict[str, str]] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned


def build_message_dicts(
    system_prompt: str,
    history: list[dict[str, str]] | None,
    user_message: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": user_message})
    return messages


def payload_preview(settings: ChatSettings, history: list[dict[str, str]] | None, user_message: str) -> str:
    payload = {
        "provider": settings.provider,
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "messages": build_message_dicts(settings.system_prompt, history, user_message or "<student message>"),
    }
    if settings.provider == "Ollama Local":
        payload["base_url"] = settings.ollama_base_url
    return json.dumps(payload, indent=2)


def chat(settings: ChatSettings, user_message: str, history: list[dict[str, str]] | None) -> str:
    if not user_message.strip():
        return "Please type a message first."

    if settings.provider == "OpenAI via LangChain":
        return openai_chat(settings, user_message, history)
    if settings.provider == "Hugging Face Inference API":
        return huggingface_chat(settings, user_message, history)
    if settings.provider == "Ollama Local":
        return ollama_chat(settings, user_message, history)
    raise ValueError(f"Unknown provider: {settings.provider}")


def openai_chat(settings: ChatSettings, user_message: str, history: list[dict[str, str]] | None) -> str:
    api_key = settings.openai_api_key.strip() or env_value("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY is missing. Add it in .env or paste it in the OpenAI key field."

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        return f"OpenAI/LangChain dependencies are not installed correctly: {exc}"

    messages: list[Any] = [SystemMessage(content=settings.system_prompt or DEFAULT_SYSTEM_PROMPT)]
    for item in normalize_history(history):
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=user_message))

    try:
        llm = ChatOpenAI(
            model=settings.model,
            api_key=api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as exc:
        return f"OpenAI request failed: {exc}"


def huggingface_chat(settings: ChatSettings, user_message: str, history: list[dict[str, str]] | None) -> str:
    token = settings.hf_token.strip() or env_value("HF_TOKEN")
    if not token:
        return "HF_TOKEN is missing. Add it in .env or paste it in the Hugging Face token field."

    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        return f"huggingface_hub is not installed correctly: {exc}"

    messages = build_message_dicts(settings.system_prompt, history, user_message)

    try:
        client = InferenceClient(model=settings.model, token=token)
        response = client.chat.completions.create(
            messages=messages,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"Hugging Face request failed: {exc}"


def ollama_chat(settings: ChatSettings, user_message: str, history: list[dict[str, str]] | None) -> str:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama
    except Exception as exc:
        return f"Ollama/LangChain dependencies are not installed correctly: {exc}"

    messages: list[Any] = [SystemMessage(content=settings.system_prompt or DEFAULT_SYSTEM_PROMPT)]
    for item in normalize_history(history):
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=user_message))

    try:
        llm = ChatOllama(
            model=settings.model,
            base_url=settings.ollama_base_url.rstrip("/"),
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
        )
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as exc:
        return f"Ollama request failed: {exc}"

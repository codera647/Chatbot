# LLM Chatbot Labs

One GUI for a master's-level lecture demo:

- Lab 1: OpenAI API through LangChain, plus Hugging Face Inference API.
- Lab 2: Local LLM chatbot through Ollama.
- Setup Check: package, key, and Ollama diagnostics.

## 1. Setup

```powershell
cd D:\Chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add only the keys you want to demo.

```env
OPENAI_API_KEY=sk-your-openai-key
HF_TOKEN=hf_your_huggingface_token
```

If `python -m venv .venv` fails on Windows because `ensurepip` cannot write to the system temp folder, use the existing `.venv` if it is present. Otherwise create a writable temp folder and retry from an elevated terminal.

## 2. Run the GUI

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## 3. Ollama Lab

Install or start Ollama, then pull a model:

```powershell
ollama pull llama3.2
ollama serve
```

If `ollama serve` says the address is already in use, Ollama is already running.
The app defaults to an installed Ollama model when it can detect one. Use `Refresh Ollama models` in the local lab if you pull a new model while the app is open.

## 4. Lecture Flow

1. Start in `Setup Check` to show that keys, packages, and Ollama are separate moving parts.
2. Open `Lab 1: Cloud LLMs`, choose `OpenAI via LangChain`, and ask a simple question.
3. Click `Preview payload` to show message roles, history, temperature, max tokens, and model name.
4. Switch to `Hugging Face Inference API` and run the same prompt to compare providers.
5. Open `Lab 2: Local LLM with Ollama`, refresh models, and ask the same prompt again.
6. Discuss the tradeoff: cloud models are easy and powerful; local models improve privacy and offline control but require local resources.

## 5. Student Concepts To Highlight

- A chatbot is not just one prompt; it sends conversation history each turn.
- The system prompt sets behavior, the user message asks the task, and assistant messages preserve context.
- Temperature changes response variation.
- Max tokens caps the output length.
- LangChain standardizes model calls but each provider still has its own setup, auth, and runtime behavior.
- Ollama exposes a local HTTP API, so the app talks to a model running on the same machine.

## 6. Troubleshooting

- `OPENAI_API_KEY is missing`: add it to `.env` or paste it into the key field.
- `HF_TOKEN is missing`: create a Hugging Face access token and add it to `.env`.
- `Ollama request failed`: run `ollama pull llama3.2`, then confirm Ollama is running.
- Package errors: activate `.venv`, then run `python -m pip install -r requirements.txt`.

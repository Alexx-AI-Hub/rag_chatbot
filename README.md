# AI Chatbot(RAG)

A grounded AI assistant that can answer questions in seconds using your own documents or the web.

## What it does

- Chat with your own documents
- Search the web when local files are not enough
- Show sources to make answers easier to trust
- Keep a persistent knowledge base
- Handle temporary session uploads separately
- Let useful session files become part of the long-term knowledge base

## Why this project is interesting

Practical AI Chat-Bot that combines:
- Local document knowledge
- Live web context
- Source-based answers
- Smoother workflow for adding and reusing information
- Run locally(Free)


## Built with

- Python
- Chainlit(App/UI)
- LlamaIndex(Hybrid-RAG)
- DuckDuckGo Search(Web-RAG)
- Ollama / OpenAI-Response-Client


Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
chainlit run src/ragbot/app.py
```


## This repository is meant to demonstrate:
- practical AI application building
- grounded/transparent answers using citation
- RAG-capabilities using llama-index
- Settings and model abstraction for more user-control
- User Friendly UI

## Run

1. Install deps: `pip install -r requirements.txt` (or `pip install -e .` if you use pyproject)
2. Start: `chainlit run src/ragbot/app.py`

# AGENTS.md — rag-chatbot (Codex Guidelines)

## Repository exploration rules
- Read the minimum number of files needed to solve the task.
- Prefer targeted search and targeted reads over broad file dumps.
- Avoid recursive scans of excluded, generated, or large folders unless clearly needed.

## Output limits
- Prefer constrained search output, for example:
- Avoid broad root-level recursive listings.

## Expensive operations to avoid by default
- broad `Get-Content` across many files
- recursive reads across the whole repo
- scanning `.venv`, `.git`, `__pycache__`, `.pytest_cache`, `node_modules`, `dist`, `build`
- recursive scans of `data/` unless the task is about indexing, persistence, file storage or needed

## Purpose
This project is a simple RAG chatbot built with Python, Chainlit, and LlamaIndex.
Keep the codebase as small, readable, and understandable as possible.
The Code Should be Scalable and Testable using SOLID Principles at a reasonable level.

## Current project scope
- Local RAG chatbot (default path)
- Simple online answer mode (DuckDuckGo + LLM)
- Persistent base docs and base index
- Temporary session docs and session index
- Cleanup temp data on app start and on `@cl.on_chat_end`
- Promote temp files to base using copy (not move) when triggered by user/UI event

## File ownership
- `src/ragbot/app.py`: Chainlit UI/events, event routing, and UI rendering. All chainlit framework code
- `src/ragbot/rag.py`: RAG logic(LlamaIndex) --> indexing, retrieval, citations/source extraction, and similar
- `src/ragbot/duckrag.py`: Online search retrieval and LLM Generation/synthesis flow (DuckDuckGo + prompt_template + llm.py)
- `src/ragbot/llm.py`: OpenAI Client helpers using local models
- `src/ragbot/file_manager.py`:All logic and handling related to Read/Write/Validate/Move/Copy files and similar.
- `src/ragbot/Schemas.py`: All shared API/Schemas
- `src/ragbot/config.py`: Constants/prompts/Global-Configurations

## Design principles (minimal)
- Follow Clean Code: consistent and well describing names, small functions
- Prefer simplicity to excessive separation
- Avoid duplication (DRY), but do not over-abstract early (KISS/YAGNI). Ask for clarification if you are unsure!

## Rules
- Prefer small code but with clear names
- Keep Chainlit-specific code in `app.py`
- Do not add complex architecture layers unless explicitly requested
- Add new files only when they clearly reduce complexity and keep current contracts stable.**
- Suggest improvements that increase testability, scalability, and maintainability.

## Future plans and scalability
- Planned future extensions may include a dedicated `workflow.py`/`router.py` (possibly LangGraph/LangChain) as routing complexity grows.
- Do not hallucinate answers when retrieval is empty or weak

## Testing
- Prefer targeted tests over full-suite runs.
Validate at least:
 - To be implemented!
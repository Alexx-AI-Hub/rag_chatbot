# AGENTS.md

## Repo rules
- Inspect only files relevant to the task.
- Prefer targeted searches and focused reads over broad scans.
- Do not scan `.venv`, `.git`, `__pycache__`, `.pytest_cache`, `node_modules`, `dist`, `build`, or unrelated `data/` paths unless needed.

## Change policy
- Keep changes minimal and localized.
- Do not refactor unrelated code.
- Add new files only when they clearly reduce complexity.
- Keep existing contracts and API shapes stable unless the task requires changes.

## Project structure
- `src/ragbot/app.py`: Chainlit-specific code, UI, and event handling.
- `src/ragbot/router.py`: LangGraph routing between local and web retrieval.
- `src/ragbot/local_rag.py`: Local document indexing, loading, and retrieval.
- `src/ragbot/web_rag.py`: Web search, page loading, chunking, and ranking.
- `src/ragbot/rag.py`: Shared retrieval, chunking, and normalization helpers.
- `src/ragbot/llm.py`: LLM client helpers.
- `src/ragbot/file_manager.py`: File handling and validation.
- `src/ragbot/schemas.py`: Shared schemas.
- `src/ragbot/config.py`: Config and prompts.

## Style
- Prefer simple, readable, and testable code.
- Use small functions and clear names.
- Do not add complex architecture layers unless explicitly requested.

## Validation
- Use the smallest relevant test or validation command.
- Prefer targeted tests over full-suite runs.

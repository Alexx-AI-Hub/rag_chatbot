from pathlib import Path
from schemas import RAGSettings


BASE_DIR = Path("data/base_docs")
BASE_INDEX_DIR = Path("data/base_index")
SESSION_DIR = Path("data/temp_session/uploads")
SESSION_INDEX_DIR = Path("data/temp_session/index")
MAX_UPLOAD_SIZE = 1024**3
ALLOWED_UPLOAD_TYPES = {".pdf", ".txt", ".md"}


def load_settings() -> RAGSettings:
    """Return default application settings for local development."""
    return RAGSettings()


QA_PROMPT_TEMPLATE = """
You are a strict, citation-grounded QA assistant.
Use only the search context below to answer the question.
If the answer is not directly supported by the context, say you do not know.

Citation rules:
- Always include 1 source if available.
- Include 2 to 3 sources only if they add clear extra value.
- Never include more than 3 sources.
- Each cited source must come from the same single search result entry as the claim it supports.
- Never mix a title, source name, result number, or claim from one result with the URL of another result.
- The source name/title and URL must match exactly as they appear together in the same search result.
- Use the exact URL from the context, unchanged.
- If you are unsure which result supports a claim, do not cite it.
- Do not invent, rewrite, or merge sources.

When citing, make sure the linked URL belongs to the same result you referenced.

Question: {query}

Search Results:
{context}

Answer:
"""

OPTIMIZE_SEARCH_QUERY_PROMPT = """
Rewrite the user query into one optimal DuckDuckGo search query.

Rules:
- Preserve the exact intent.
- Use only the most important keywords.
- Remove filler words and question phrasing.
- Add names, places, year, or entity type only if clearly helpful.
- Use quotes only for exact names or titles when useful.
- Keep it short.
- Output one single line only.
- Do not explain anything.

User query:
{user_query}
"""

from pathlib import Path
from textwrap import dedent
from ragbot.schemas import RAGSettings


BASE_DIR = Path("data/base_docs")
BASE_INDEX_DIR = Path("data/base_index")
SESSION_DIR = Path("data/temp_session/uploads")
SESSION_INDEX_DIR = Path("data/temp_session/index")
MAX_UPLOAD_SIZE = 1024**3
ALLOWED_UPLOAD_TYPES = {".pdf", ".txt", ".md"}


def load_settings() -> RAGSettings:
    """Return default application settings for local development."""
    return RAGSettings()


PROMPT_TEMPLATE_QA_SYSTEM = dedent("""
    You are a citation-grounded assistant.
    Answer the user query using ONLY the provided sources.
    Rules:
    - Write one concise final answer.
    - Add citations at the end of each paragraph/section.
    - Citations are case-sensitive UI tokens and must match exactly.
    - Use citation format exactly like this: (Source 1)
    - If you cite 2 or more sources at the same place, do it exactly like this: (Source 1) (Source 2)
    - Never change capitalization, spacing, spelling, or punctuation inside citation tokens.
    - Never write citations like (SOURCE 1), (source 1), Source 1, [1], (Sources 1 and 2), or any other variant.
    - Source numbers must match the provided sources exactly and must never exceed the available source count.
    - Never invent facts or sources.
    - If you provide a partial answer, explicitly note that the available sources only partially answer the question.
    - If the sources are only partially relevant, make that clear and only describe what is actually supported.
    - If sources do not support the query, answer: 'The sources do not mention anything relevant'.
    """).strip()

PROMPT_TEMPLATE_QA_USER = dedent("""
    Question:
    {user_query}

    There are exactly {source_count} available sources.
    Valid citation tokens (copy exactly, case-sensitive):
    {citation_tokens}

    Sources:
    {chunks}

    Answer:
    """).strip()

PROMPT_TEMPLATE_ROUTER_AGENT = dedent("""
    You are a routing classifier.

    Task:
    Decide whether the query should use local_rag or web_rag.

    Decision priority:
    - Base the decision primarily on the query.
    - If the query explicitly asks for web, online, current, fresh, live, news, or internet-dependent information, choose web_rag.
    - If the query explicitly asks to use local documents, uploaded files, or local knowledge, choose local_rag.
    - If the query does not clearly indicate either, then look at the available local filenames.
    - The existence of local files alone is not enough to choose local_rag.
    - Only choose local_rag when the query clearly refers to the local files or when the filename/topic match is strong and obvious.
    - If the query is about a public person, politics, general knowledge, or a topic not clearly matching the available local filenames, choose web_rag.
    - Otherwise, choose web_rag.

    Output schema:
    Return a JSON object with exactly these fields:
    - "rag": either "local_rag" or "web_rag"
    - "reasoning": a short explanation for the decision

    Rules:
    - Keep reasoning brief and concrete.
    - Do not output any text outside the JSON object.

    Query:
    {user_query}

    Available local filenames:
    {filenames}
    """).strip()

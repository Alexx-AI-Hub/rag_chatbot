from openai import OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel
import ollama
import httpx
import logging as log
from textwrap import dedent

from ragbot.schemas import Chunk

PROMPT_TEMPLATE_QA_SYSTEM = dedent("""
    You are a citation-grounded assistant.
    Answer the user query using ONLY the provided sources.
    Rules:
    - Write one concise final answer.
    - Add citations at the end of each paragraph/section.
    - Use citation format exactly: (Source 1), (Source 2).
    - It is extremely important that you do not forget parenthesis around Source () !.
    - If you want to cite 2 or more sources at the same place do it like this (Source 1) (Source 2).
    - Source numbers must match provided sources exactly.
    - Never invent facts or sources.
    - If you provide a partial answer, explicitly note that the available sources only partially answer the question.
    - If the sources are only partially relevant, make that clear and only describe what is actually supported.
    - If sources do not support the query, answer: 'The sources do not mention anything relevant'.
    """).strip()

PROMPT_TEMPLATE_QA_USER = dedent("""
    Question:
    {user_query}

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
    - If the filenames suggest relevant local documents exist, choose local_rag.
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



def get_openai_client(api_key:str="KEY", base_url:str="http://localhost:11434/v1", timeout:int=30 ):
    """Build an OpenAI-compatible client for the configured backend."""
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def chat_response(client:OpenAI, prompts:ResponseInputParam, system_prompt: str,
                  model: str = "llama3.2:3b", temp:float=0.2, top_p:float=0.95, stream:bool=False):
    """Send a chat request through the Responses API."""
    return client.responses.create(
        model=model,
        instructions=system_prompt,
        input=prompts,
        temperature=temp,
        top_p=top_p,
        stream=stream,
    )


def gen_response(client:OpenAI, prompt:str, model:str, temp:float=0.2, top_p:float=0.95):
    """Return plain output text for a single prompt."""
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temp,
        top_p=top_p,
    )
    return response.output_text


def gen_response_strict_output(prompt:str | ResponseInputParam, resp_format:type[BaseModel],
                               client:OpenAI, model:str, temp:float=0.2, top_p:float=0.95):
    """Parse a structured response into the supplied Pydantic model."""

    strict_response = client.responses.parse(
        model=model,
        input=prompt,
        temperature=temp,
        top_p=top_p,
        text_format=resp_format,
    )
    return strict_response.output_parsed


def str_format_chunks(chunks:list[Chunk]) -> str:
    """Format retrieved chunks into a prompt-friendly text block."""
    return "\n\n".join(
        dedent(f"""
         Source {i}
         Url: {chunk.url or 'N/A'}
         Content: {chunk.content or "N/A"}
         """).strip()
        for i, chunk in enumerate(chunks, 1)
    )



def synthesize_response_with_chunks(client:OpenAI, query:str, chunks:list[Chunk],
                        model:str="llama3.2:3b", temp:float=0.2, top_p:float=0.95, stream:bool=False) -> str:
    """Generate an answer grounded only in the provided chunks."""

    if not query.strip():
        return "SYSTEM: Try asking something or uploading a file at least !"
    if not chunks:
        return "SYSTEM: GOT NO CHUNKS !!!"

    structured_chunks = str_format_chunks(chunks=chunks)
    user_prompt = PROMPT_TEMPLATE_QA_USER.format(user_query=query, chunks=structured_chunks)

    response = chat_response(
        client=client,
        prompts=user_prompt,
        system_prompt=PROMPT_TEMPLATE_QA_SYSTEM,
        model=model,
        temp=temp,
        top_p=top_p,
        stream=stream,
    )
    return response.output_text if not stream else response


def get_model_list(client:OpenAI):
    """Return available model ids or an empty list on error."""
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        log.error(e)
        return []


def __get_ollama_list():
    """Return available local Ollama model names or raise a runtime error."""
    try:
        return [m["model"] for m in ollama.list()["models"]]
    except httpx.ConnectError as e:
        error_msg = f"No response from Ollama server --> Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)
    except ollama.ResponseError as e:
        error_msg = f"There was an error in the response from Ollama. Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)
    except (KeyError, TypeError) as e:
        error_msg = f"Failed to parse models (this can occur because of an change in their API). Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred. Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)

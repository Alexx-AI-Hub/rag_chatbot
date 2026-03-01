from __future__ import annotations

import asyncio
from typing import Any, Coroutine

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

import file_manager, rag
from llm import get_openai_list
from duckrag import chat_resp_with_duck_search

from config import BASE_DIR, BASE_INDEX_DIR, SESSION_INDEX_DIR, SESSION_DIR, QA_PROMPT_TEMPLATE, log
from schemas import RAGSettings


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize settings, ensure folders, cleanup leftovers, and show chat settings."""
    file_manager.ensure_data_dirs()             # Ensure all 4 path/dir exist or create them
    file_manager.clean_session_dirs()           # Cleans session dirs if not empty at start

    rag.set_handler(cl.LlamaIndexCallbackHandler()) #Sets LlamaIndexCallbackHandler
    models = get_openai_list()

    settings = RAGSettings()
    chat_settings = await _build_chat_settings(settings, models) # get default RAG-Settings and Ollama models list
    cl.user_session.set("chat_settings",chat_settings)

    await cl.Message(
        content="Save/Persist session files",
        actions=[
            cl.Action(name="persist_session_to_base", label="Click to Save", payload={})
        ],
    ).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    """Clear session files and session index when the chat ends."""
    await asyncio.to_thread(file_manager.reset_directory, SESSION_DIR)
    await asyncio.to_thread(file_manager.reset_directory, SESSION_INDEX_DIR)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle uploads and a user query from the same message event."""
    chat_settings = cl.user_session.get("chat_settings")

    if message.elements:
        for element in message.elements:
            path = file_manager.get_valid_path(element.path)
            file_name = element.name
            if path:
                saved = await asyncio.to_thread(file_manager.copy_if_allowed, file_path=path, dest_dir_path=SESSION_DIR, file_name=file_name)
                if saved:
                    indexed = await asyncio.to_thread(rag.build_index, SESSION_DIR, SESSION_INDEX_DIR, chat_settings)
                    if not indexed:
                        await cl.Message("Failed to Save and Index Uploaded file !").send()
                        return

    if message.content:
        msg = str(message.content or "").strip()

        if not chat_settings["online"]:

            response = await cl.make_async(rag.query)(msg, chat_settings)
            answer_message = cl.Message(
                content="",
                elements=_build_sources_element(response)
            )
            if response:
                try:
                    for token in response.response_gen:
                        await answer_message.stream_token(token)
                    await answer_message.update()
                except AttributeError:                             # <-- streaming=False
                    await cl.Message(
                        content=response.response,
                        elements=_build_sources_element(response)
                    ).send()

        else:
            response = chat_resp_with_duck_search(
                query=msg,
                prompt_template=QA_PROMPT_TEMPLATE,
                model=chat_settings["llm_model"],
                temp=0.2,
                top_p=0.9
            )
            await cl.Message(content=response).send()



@cl.on_settings_update
async def settings_update(chat_settings: dict[str, Any]) -> None:
    """Apply updated Chainlit settings to the current app settings object."""
    cl.user_session.set("chat_settings", chat_settings)
    log.info("New Settings: %s",chat_settings)


def _build_chat_settings(settings: RAGSettings, models: list[str]) -> Coroutine[Any, Any, Any]:
    """Build the Chainlit chat settings UI using current settings and available models."""
    return cl.ChatSettings(
        [
            Select(
                id="llm_model",
                label="LLM Model - Ollama",
                values=models,
                initial_value=settings.llm_model,
            ),
            Select(
                id="embed_model",
                label="Embedding Model - Ollama",
                values=models,
                initial_value=settings.embed_model,
            ),
            Slider(
                id="temperature",
                label="LLM - Temperature",
                initial=float(settings.temperature),
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="top_k",
                label="Top K - Retrieved Chunks",
                initial=int(settings.top_k),
                min=1,
                max=20,
                step=1,
            ),
            Switch(
                id="streaming",
                label="Stream Response - ON/OFF",
                initial=settings.streaming,
            ),
            Select(
                id="metadata_mode",
                label="Provide Metadata - LLM/Embedding Model",
                values=["all", "llm", "embed", "none"],
                initial_value=settings.metadata_mode,
            ),
            Switch(
                id="online",
                label="Online Search - Web/Local RAG",
                initial=settings.online,
            ),
        ]
    ).send()


@cl.action_callback("persist_session_to_base")
async def on_promote():
    file_manager.copy_files_from_dir(SESSION_DIR, BASE_DIR)
    rag.build_index(BASE_DIR, BASE_INDEX_DIR, cl.user_session.get("chat_settings"))
    await cl.Message(content="Session files are now persisted for future sessions :).").send()


def _build_sources_element(response: object) -> list[cl.Text | cl.Pdf] | None:
    """Create a sidebar source element from the query response."""
    elements = []
    if response:
        sources = rag.resp_source(response)
        for i, src in enumerate(sources, 1):
            if src["src_type"] == "application/pdf":
                element = cl.Pdf(
                    name=f"Source {i} Score{src['src_score']}",
                    display="side",
                    path=src["file_path"],
                    page=int(src["src_page"]),
                )
            else:
                element = cl.Text(
                    name=f"Source {i}",
                    content=src["source_info"],
                    display="side",

                )
            log.info("")
            elements.append(element)
    return elements
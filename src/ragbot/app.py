from __future__ import annotations
import logging as log
import time
from pathlib import Path
import sys

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s() - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
from dataclasses import asdict
import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
import asyncio
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ragbot.file_manager as file_manager
import ragbot.local_rag as local_rag
import ragbot.router as router
from ragbot.llm import get_model_list, get_openai_client
from ragbot.config import BASE_DIR, BASE_INDEX_DIR, SESSION_INDEX_DIR, SESSION_DIR
from ragbot.schemas import RAGSettings


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize session state, settings, and indexes."""

    file_manager.ensure_data_dirs()
    file_manager.clean_session_dirs()

    client = get_openai_client()
    models = get_model_list(client)
    cl.user_session.set("client", client)

    chat_settings = await _build_chat_settings(RAGSettings(), models)
    settings = _to_rag_settings(chat_settings)
    cl.user_session.set("chat_settings",settings)

    base_index = local_rag.load_index(BASE_INDEX_DIR, settings)
    cl.user_session.set("base_index", base_index)

    cl.user_session.set("router_graph", router.build_graph())

    await cl.Message(
        content="Save/Persist session files",
        actions=[
            cl.Action(name="persist_session_to_base", label="Click to Save", payload={})
        ]
    ).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    """Clear session files and session index when the chat ends."""
    await asyncio.to_thread(file_manager.reset_directory, SESSION_DIR)
    await asyncio.to_thread(file_manager.reset_directory, SESSION_INDEX_DIR)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handles both file uploads and user query message send event from UI."""
    msg_text = str(message.content or "").strip()
    saved = 0
    start_time = time.perf_counter()


    chat_settings = cl.user_session.get("chat_settings")
    if chat_settings:
        settings = _to_rag_settings(chat_settings)
    else:
        raise RuntimeError("chat_settings missing in user_session")
    session_settings_time = time.perf_counter()


    if message.elements:
        for element in message.elements:
            path = file_manager.get_valid_path(element.path)
            file_name = element.name

            if await asyncio.to_thread(
                file_manager.copy_if_allowed,
                file_path=path,
                dest_dir_path=SESSION_DIR,
                file_name=file_name
            ):
                saved += 1
            else:
                await cl.Message(f"Failed to Copy File: {file_name}").send()
                log.warning("Failed to Copy File: %s",file_name)

        if saved:
            session_index = await asyncio.to_thread(local_rag.build_index, SESSION_DIR, SESSION_INDEX_DIR, settings)
            if not session_index:
                await cl.Message("Failed to Re-Index Uploaded files !").send()
                return
            cl.user_session.set("session_index", session_index)
    element_time = time.perf_counter()


    if msg_text:
        both_indexes = [idx for idx in (cl.user_session.get("base_index"), cl.user_session.get("session_index")) if idx]
        graph = cl.user_session.get("router_graph") or router.build_graph()
        state = router.State(
            query=msg_text,
            settings=settings,
            local_indexes=both_indexes,
            client=cl.user_session.get("client"),
        )
        workflow_state = await cl.make_async(graph.invoke)(state)

        if workflow_output := (workflow_state["output"] or None):

            if error:= workflow_output.error:
                await cl.Message(content=f"Workflow-Error: {error}").send()

            elif answer:= workflow_output.answer:
                sources = workflow_output.metadata.chunks

                if not settings.streaming:
                    await cl.Message(content=answer, elements=_build_citation_element(sources)).send()
                else:
                    msg = cl.Message(content="", elements=_build_citation_element(sources))
                    await msg.send()
                    for event in answer:
                        if event.type == "response.output_text.delta":
                            await msg.stream_token(event.delta)
                    await msg.update()
    msg_end = time.perf_counter()

    log.info(f"\nLatency: {(msg_end - start_time):.2f} seconds.")
    log.info(f"    Upload: {(element_time - start_time):.2f} seconds.")
    log.info(f"    Query: {(msg_end - element_time):.2f} seconds.")


@cl.on_settings_update
async def settings_update(chat_settings: dict[str, Any]) -> None:
    """Apply updated Chainlit settings to the current app settings object."""

    settings = _to_rag_settings(chat_settings)
    cl.user_session.set("chat_settings", settings)
    log.info("New Settings: %s",chat_settings)


async def _build_chat_settings(settings: RAGSettings, models: list[str]) -> dict[str, Any]:
    """Build the Chainlit chat settings UI using current settings and available models."""
    return await cl.ChatSettings(
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
            Select(
                id="rag_mode",
                label="RAG Mode",
                values=["auto", "web", "local"],
                initial_value=settings.rag_mode,
            ),
        ]
    ).send()


def _to_rag_settings(chat_settings: dict[str, str|int|float|bool] | RAGSettings) -> RAGSettings:
    """Normalize UI settings into a RAGSettings instance."""
    return chat_settings if isinstance(chat_settings, RAGSettings) else RAGSettings(**chat_settings)


@cl.action_callback("persist_session_to_base")
async def on_persist():
    """Persist uploaded session files into the base document store."""
    success = True
    if not await asyncio.to_thread(file_manager.copy_files_from_dir,
        src_dir_path=SESSION_DIR,
        dest_dir_path=BASE_DIR
    ):
        success = False
    base_index = await asyncio.to_thread(local_rag.build_index,
                            src_path=BASE_DIR,
                            dest_path=BASE_INDEX_DIR,
                            settings=_to_rag_settings(cl.user_session.get("chat_settings") or asdict(RAGSettings()))
    )
    cl.user_session.set("base_index", base_index)

    if not success:
        await cl.Message(content="Some session files were persisted, but one or more files could not be copied.").send()
    else:
        await cl.Message(content="Session files are now persisted for future sessions :).").send()


def _build_citation_element(chunks:list[object])-> list[cl.Text] | None:
    """Build Chainlit side-panel elements for retrieved chunks."""
    if chunks is None:
        return None

    elements = []
    for i, chunk in enumerate(chunks, 1):
        element = cl.Text(
            content=f"Content:\n{chunk.content}\n\nUrl: {chunk.url}",
            name=f"(Source {i})",
            display="side"
        )
        elements.append(element)
    return elements

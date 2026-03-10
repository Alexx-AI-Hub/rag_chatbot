from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from openai import OpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from llama_index.core.indices.base import BaseIndex

import ragbot.config as config
import ragbot.file_manager as file_manager
import ragbot.llm as llm
import ragbot.local_rag as local_rag
import ragbot.rag as rag
import ragbot.web_rag as web_rag
from ragbot.schemas import RAGSettings, Chunk, WorkflowOutput

import logging as log

class RouterResponse(BaseModel):
    rag: Literal["local_rag", "web_rag"]
    reasoning: str = Field(description="Explain why you choose web or local for rag")

Node = Literal["local_rag", "web_rag", "chat_output"]
@dataclass
class State:
    """Global Workflow State/Memory"""

    query: str
    settings: RAGSettings
    client: OpenAI | None = None

    local_indexes: list[BaseIndex] = field(default_factory=list)
    local_filenames: list[str] = field(default_factory=list)

    node: Node = None
    reasoning :str = None
    output: WorkflowOutput = field(default_factory=WorkflowOutput)


def router(state: State) -> Node:
    """Select next node based on explicit rag_mode (local, web, auto).
       auto lets the agent decide between local and web retrieval."""

    if not (state.query or '').strip() or not state.settings or not state.client:
        state.output.error = (
            f"Initial state not valid!"
            f"\nQuery: {state.query!r}"
            f"\nSettings: {'OK' if state.settings  else 'Missing!'}"
            f"\nClient: {'OK' if state.client else 'Missing!'}"
        )
        state.output.events.append({"node": "router", "error": state.output.error})
        return "chat_output"

    if state.settings.rag_mode == "local":
        state.node = "local_rag"
        return state.node
    elif state.settings.rag_mode == "web":
        state.node = "web_rag"
        return state.node
    elif state.settings.rag_mode == "auto":
        state.node = llm_router_node(state)
        return state.node

    else:
        state.output.error = f"Rag-Mode value is invalid: {state.settings.rag_mode}"
        state.output.events.append({"node": "router", "error": state.output.error})
        return "chat_output"


def llm_router_node(state: State) -> Node:
    """Use LLM-based Agent to choose between local_rag and web_rag from the user query."""
    list_of_filenames = file_manager.get_filenames_from_dir(
        paths=([config.BASE_DIR, config.SESSION_DIR])
    )
    router_prompt = llm.PROMPT_TEMPLATE_ROUTER_AGENT.format(
        user_query=state.query,
        filenames=("\n".join(list_of_filenames) or "N/A")
    )

    try:
        node_resp = llm.gen_response_strict_output(
            client=state.client,
            prompt=router_prompt,
            model=state.settings.llm_model,
            temp=0.0,
            top_p=1.0,
            resp_format=RouterResponse,
        )
        log.info("Router prompt:\n%s", router_prompt)
        log.info("Router choice: %s", node_resp.rag)
        log.info("Router reasoning: %s", node_resp.reasoning)
        return node_resp.rag

    except Exception as e:
        state.output.error = str(e)
        state.output.events.append({"node": "router", "status": state.output.error})
        return "chat_output"


def local_rag_node(state: State) -> State:
    """Populate state with locally retrieved chunks."""
    embed_model = rag.build_openai_compatible_embedding(model_name=state.settings.embed_model)
    llm_model = rag.build_openai_compatible_llm(model_name=state.settings.llm_model)
    state.output.metadata.rag_used = "local"

    chunks:list[Chunk] = local_rag.get_top_k_chunks(
        state.query,
        state.local_indexes,
        state.settings,
        llm_model=llm_model
    )
    if chunks:
        state.output.metadata.chunks.extend(chunks)
        state.output.events.append({"node": "local_rag", "status": "OK"})
    else:
        state.output.error = "Got No Chunks!"
        state.output.events.append({"node": "local_rag", "status": state.output.error })

    return state


def web_rag_node(state: State) -> State:
    """Populate state with web-retrieved chunks."""
    embed_model = rag.build_openai_compatible_embedding(model_name=state.settings.embed_model)
    llm_model = rag.build_openai_compatible_llm(model_name=state.settings.llm_model)
    state.output.metadata.rag_used = "web"

    chunks: list[Chunk] = web_rag.get_top_k_chunks(
        state.query,
        state.settings,
        embed_model,
        llm_model
    )
    if chunks:
        state.output.metadata.chunks.extend(chunks)
        state.output.events.append({"node": "web_rag", "status": "OK"})
    else:
        state.output.error = "Got No Chunks!"
        state.output.events.append({"node": "web_rag", "status": state.output.error})

    return state


def chat_output_node(state: State) -> State:
    """Finalize result payload so output is self-contained for UI/logging."""
    if state.output.metadata.chunks and state.query:
        try:
            response = llm.synthesize_response_with_chunks(
                client=state.client,
                query=state.query,
                chunks=state.output.metadata.chunks,
                model=state.settings.llm_model,
                temp=state.settings.temperature,
                top_p=state.settings.top_p,
                stream=state.settings.streaming
            )
            state.output.answer = response
            state.output.events.append({"node": "chat_output_node", "status": "OK"})
        except Exception as e:
            state.output.error = str(e)
            state.output.events.append({"node": "chat_output_node", "status": state.output.error})

    state.output.query = state.query if state.query else None
    if state.settings:
        state.output.metadata.temperature = state.settings.temperature
        state.output.metadata.top_p = state.settings.top_p
        state.output.metadata.top_k = state.settings.top_k
        state.output.metadata.llm_model = state.settings.llm_model
        state.output.metadata.embed_model = state.settings.embed_model

        md = state.output.metadata
        log.info(
            "Metadata:\nrag_used=%r\nllm_model=%r\ntop_p=%r\ntemperature=%r\nembed_model=%r\ntop_k=%r\nchunks=%d",
            md.rag_used, md.llm_model, md.top_p, md.temperature, md.embed_model, md.top_k, len(md.chunks),
        )
    log.info("Chunk contents:\n%s", "\n\n".join(chunk.content or "" for chunk in state.output.metadata.chunks))

    return state


def build_graph() -> Any:
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(State)
    graph.add_node("local_rag", local_rag_node)
    graph.add_node("web_rag", web_rag_node)
    graph.add_node("chat_output", chat_output_node)

    graph.add_conditional_edges(
        START,
        router,
        {
            "local_rag": "local_rag",
            "web_rag": "web_rag",
            "chat_output": "chat_output",
        },
    )
    graph.add_edge("local_rag", "chat_output")
    graph.add_edge("web_rag", "chat_output")
    graph.add_edge("chat_output", END)
    return graph.compile()

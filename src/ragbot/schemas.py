from dataclasses import dataclass
from typing import Literal, Any
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

@dataclass
class RAGSettings:
    """Retrieval, chunking, and model/generation settings."""
    client:OpenAI | None = None

    chunk_size: int = 512
    chunk_overlap: int = 100

    llm_model: str = "llama3.1:8b"
    top_p:float = 1
    temperature: float = 0.1

    embed_model: str = "nomic-embed-text:latest"
    top_k: int = 2

    streaming:bool = True
    rag_mode: Literal["auto", "web", "local"] = "auto"


@dataclass(frozen=True)
class TextItem:
    url: str
    title: str | None
    content: str


@dataclass(frozen=True)
class SearchHit(TextItem):
    score: float | None = None


@dataclass(frozen=True)
class Chunk(TextItem):
    score: float


class MetaData(BaseModel):
    rag_used: Literal["local", "web"] | None = None
    chunks: list["Chunk"] = Field(default_factory=list)

    llm_model: str | None = None
    top_p: float | None = None
    temperature: float | None = None

    embed_model: str | None = None
    top_k: int | None = None


class WorkflowOutput(BaseModel):
    """Unified model output for both local RAG and web RAG flows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str | None = None
    answer: str | None = None
    reasoning: str | None = None
    metadata: MetaData = Field(default_factory=MetaData)
    events: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

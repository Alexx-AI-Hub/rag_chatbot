from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from ragbot.schemas import RAGSettings, Chunk

BASE_URL = "http://localhost:11434/v1"
API_KEY = "placeholder"


def build_openai_compatible_embedding(model_name: str, base_url: str=BASE_URL, api_key: str=API_KEY, batch_size: int=64) -> OpenAILikeEmbedding:
    """Build an OpenAI-compatible embedding client."""
    return OpenAILikeEmbedding(
        model_name=model_name,
        api_base=base_url,
        api_key=api_key,
        embed_batch_size=batch_size
    )


def build_openai_compatible_llm(model_name: str, base_url: str=BASE_URL, api_key: str=API_KEY) -> OpenAILike:
    """Build an OpenAI-compatible LLM client."""
    return OpenAILike(
        model=model_name,
        api_base=base_url,
        api_key=api_key,
    )


def build_sentence_splitter(settings: RAGSettings) -> SentenceSplitter:
    """Build a sentence splitter from chunk settings."""
    return SentenceSplitter(
        chunk_size=max(50, int(settings.chunk_size)),
        chunk_overlap=max(0, int(settings.chunk_overlap)),
    )



def retrieve_fusion(
        query:str, nodes:list[TextNode], embed_model:BaseEmbedding, llm:LLM, top_k:int)->list[NodeWithScore]:
    """Fuse BM25 and vector retrieval over in-memory nodes."""

    if nodes and query:

        bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
        vector = VectorStoreIndex(nodes=nodes, embed_model=embed_model).as_retriever(similarity_top_k=top_k)

        fusion = QueryFusionRetriever(
            retrievers=[bm25, vector],
            similarity_top_k=top_k,
            num_queries=1,
            mode="relative_score",
            use_async=False,
            llm=llm
        )
        if fusion_nodes := fusion.retrieve(query):
            cutoff_nodes =  [fusion_nodes[0]]
            for i, f_node in enumerate(fusion_nodes[1:], 1):
                if float(f_node.score) < float(fusion_nodes[i - 1].score * 0.5):
                    break
                cutoff_nodes.append(f_node)
            return cutoff_nodes
    return []


def retrieve_fusion_from_indexes(
        query: str, indexes: list[BaseIndex], llm: LLM, top_k: int) -> list[NodeWithScore]:
    """Fuse retrievers built directly from loaded indexes."""

    if indexes and query:
        retrievers = [
            index.as_retriever(similarity_top_k=top_k)
            for index in indexes
        ]

        fusion = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=top_k,
            num_queries=1,
            mode="relative_score",
            use_async=False,
            llm=llm
        )
        if fusion_nodes := fusion.retrieve(query):
            cutoff_nodes = [fusion_nodes[0]]
            for i, f_node in enumerate(fusion_nodes[1:], 1):
                if float(f_node.score) < float(fusion_nodes[i - 1].score * 0.5):
                    break
                cutoff_nodes.append(f_node)
            return cutoff_nodes
    return []


def retrieve_bm25(query:str, nodes:list[TextNode], top_k:int)->list[NodeWithScore]:
    """Run BM25 retrieval over the provided nodes."""
    if not nodes or not query:
        return []

    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    bm25_nodes = bm25.retrieve(query)

    return bm25_nodes


def norm_node_to_chunk(nodes: list[NodeWithScore]) -> list[Chunk]:
    """Normalize retrieved nodes into sorted Chunk objects."""
    chunks: list[Chunk] = []

    for node in nodes:
        metadata = node.metadata or {}
        content = (node.get_content() or "").strip()

        url = (
            metadata.get("url")
            or metadata.get("file_path")
            or metadata.get("file_name")
            or "local://unknown-source"
        )
        title = metadata.get("title") or metadata.get("file_name") or str(url)

        chunks.append(
            Chunk(
                url=str(url).strip(),
                title=str(title).strip(),
                content=content,
                score=float(node.score or 0.0),
            )
        )

    chunks = [chunk for chunk in chunks if chunk.content]
    chunks.sort(key=lambda chunk: (-chunk.score, chunk.url, chunk.title, chunk.content))
    return chunks

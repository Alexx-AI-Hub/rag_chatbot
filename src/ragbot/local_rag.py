from __future__ import annotations
import logging
from pathlib import Path
import ragbot.file_manager as file_manager
import ragbot.rag as rag
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode
from llama_index.core.indices.base import BaseIndex
from ragbot.schemas import RAGSettings, Chunk

log = logging.getLogger(__name__)


def build_index(src_path: str | Path, dest_path: str | Path, settings: RAGSettings) -> VectorStoreIndex | None:
    """Build and persist a local vector index from a document directory."""
    src_path, dest_path = Path(src_path), Path(dest_path)

    if src_path.is_dir() and any(src_path.iterdir()):
        try:
            embed_model = rag.build_openai_compatible_embedding(model_name=settings.embed_model)
            text_splitter = rag.build_sentence_splitter(settings)
            docs = SimpleDirectoryReader(input_dir=str(src_path)).load_data(show_progress=True)

            if docs:
                index = VectorStoreIndex.from_documents(
                    docs,
                    embed_model=embed_model,
                    transformations=[text_splitter],
                )
                file_manager.reset_directory(dest_path)
                index.storage_context.persist(persist_dir=str(dest_path))
                log.debug("Index successfully built from %s -> %s", src_path, dest_path)
                return index

            log.debug("Docs Empty/None. Failed to build index from %s -> %s: ", src_path, dest_path)

        except Exception as e:
            log.error("Failed to build index from %s -> %s: %s", src_path, dest_path, e)

    return None


def load_index(index_dir: Path, settings: RAGSettings):
    """Load a persisted local vector index from disk."""
    embed_model = rag.build_openai_compatible_embedding(model_name=settings.embed_model)

    if index_dir.exists():

        if any(index_dir.iterdir()):
            try:
                return load_index_from_storage(StorageContext.from_defaults(
                    persist_dir=str(index_dir)),
                    embed_model=embed_model)

            except Exception:
                log.exception("Failed to load index from: %s", index_dir)

        else:
            log.debug("Index directory is empty: %s", index_dir)
    else:
        log.error("Index directory does not exist: %s", index_dir)

    return None


def _collect_text_nodes(indexes: list[BaseIndex]) -> list[TextNode]:
    """Collect stored text nodes from loaded indexes."""
    text_nodes: list[TextNode] = []

    for index in indexes:
        for node in index.docstore.docs.values():
            text = (node.get_content() or "").strip()

            if text:
                metadata = node.metadata or {}
                text_nodes.append(TextNode(text=text, metadata=metadata))

    return text_nodes


def get_top_k_chunks(msg: str, index_both: list[BaseIndex], settings: RAGSettings, llm_model: LLM | None = None) -> list[Chunk]:
    """Run local retrieval and return top_k chunks."""
    msg = msg.strip()

    if msg:
        indexes = [index for index in index_both if index]

        if indexes:
            llm_model = llm_model or rag.build_openai_compatible_llm(model_name=settings.llm_model)
            top_k = max(1, int(settings.top_k))
            fused_nodes = rag.retrieve_fusion_from_indexes(
                query=msg,
                indexes=indexes,
                top_k=top_k,
                llm=llm_model
            )
            if fused_nodes:
                return rag.norm_node_to_chunk(fused_nodes)
    return []


def get_chunk(chunk_id, index_both: list[BaseIndex]):
    """Return the stored chunk text for a node id, if present."""
    indexes = [index for index in index_both if index]

    for index in indexes:
        try:
            chunk_content = index.docstore.get_node(chunk_id).text
            return chunk_content
        except(ValueError, KeyError):
            continue

    return None


def __resp_source(query_resp, index_both: list[BaseIndex]) -> list[dict[str, str]]:
    """Map query response sources into serializable metadata dictionaries."""
    sources = []
    for src in query_resp.source_nodes:
        src_name = src.node.metadata.get("file_name", "Unknown File Name")
        src_score = round(src.score, 2)
        src_page = src.node.metadata.get("page_label")
        src_type = src.node.metadata.get("file_type")
        src_text = get_chunk(src.node.node_id, index_both)
        file_path = src.node.metadata.get("file_path")

        source_info = f"""Name: {src_name}\nType: {src_type}\nDocument: \n {src_text}"""
        log.debug("Source Info:\n%s", source_info)
        source_metadata = {"source_info": source_info, "src_type": src_type, "file_path": file_path,
                           "src_page": src_page, "src_score": src_score}

        sources.append(source_metadata)

    return sources

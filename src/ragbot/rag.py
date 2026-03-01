from __future__ import annotations
import logging
from pathlib import Path

import file_manager
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.callbacks import CallbackManager
from config import BASE_INDEX_DIR, SESSION_INDEX_DIR
from schemas import RAGSettings
log = logging.getLogger(__name__)


def set_handler(cb_handler)->None:
    Settings.callback_manager = CallbackManager([cb_handler])

def build_index(src_path: str | Path, dest_path: str | Path, settings: RAGSettings | dict)-> VectorStoreIndex | None:
    """Build and persist a vector index from a documents in directory, or return None on failure/empty directory."""
    src_path, dest_path = Path(src_path), Path(dest_path)
    if not src_path.is_dir() or not any(src_path.iterdir()):
        return None
    try:
        settings = _normalize_settings_if_dict(settings)
        _set_llama_settings(settings)
        docs = SimpleDirectoryReader(input_dir=str(src_path)).load_data(show_progress=True)
        if docs:
            file_manager.reset_directory(dest_path)
            index = VectorStoreIndex.from_documents(docs)
            index.storage_context.persist(persist_dir=str(dest_path))
            log.debug("Index successfully built from %s -> %s", src_path, dest_path)
            return index
        log.debug("Docs Empty/None. Failed to build index from %s -> %s: ", src_path, dest_path)
        return None
    except Exception as e:
        log.error("Failed to build index from %s -> %s: %s", src_path, dest_path, e)
        return None


def _normalize_settings_if_dict(settings: RAGSettings | dict) -> RAGSettings:
    """Normalizes settings if it's a dict. Important: Key values must match attribute names in RAGSettings."""
    if isinstance(settings, dict):
        settings_dict = settings
        settings = RAGSettings()
        for key, value in settings_dict.items():
            if key in settings.__dict__:
                setattr(settings, key, value)
    return settings


def _set_llama_settings(settings: RAGSettings) -> None:
    Settings.llm = Ollama(model=settings.llm_model, temperature=settings.temperature)
    Settings.embed_model = OllamaEmbedding(model_name=settings.embed_model)
    Settings.text_splitter = SentenceSplitter(
        chunk_size=max(50, int(settings.chunk_size)),
        chunk_overlap=max(0, int(settings.chunk_overlap)),
    )


def _load_index(index_dir: Path):
    if not index_dir.exists():
        log.error("Index directory does not exist: %s", index_dir)
        return None

    if not any(index_dir.iterdir()):
        log.info("Index directory is empty: %s", index_dir)
        return None

    try:
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(index_dir))
        )
    except Exception:
        log.exception("Failed to load index from: %s", index_dir)
        return None


def query(msg: str, settings: RAGSettings | dict):
    """Run a RAG query against available persisted indexes and return raw LlamaIndex response."""
    msg = msg.strip()
    if msg:
        settings = _normalize_settings_if_dict(settings)
        _set_llama_settings(settings)
        indexes = [idx for idx in (_load_index(BASE_INDEX_DIR), _load_index(SESSION_INDEX_DIR)) if idx is not None]
        if indexes:
            top_k = max(1, int(settings.top_k))
            retrievers = [
                r for idx in indexes for r in (
                    VectorIndexRetriever(index=idx, similarity_top_k=top_k),
                    BM25Retriever.from_defaults(docstore=idx.docstore, similarity_top_k=top_k),
                )
            ]
            fused_retriever = QueryFusionRetriever(
                retrievers=retrievers,
                similarity_top_k=top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=True,
                #callback_manager=Settings.callback_manager
            )
            query_engine = CitationQueryEngine.from_args(
                indexes[0],     # only required anchor for engine
                retriever=fused_retriever,
                citation_chunk_size=512,
                citation_chunk_overlap=50,
                top_k=top_k,
                streaming=settings.streaming,
                #callback_manager=Settings.callback_manager,
                metadata_mode=settings.metadata_mode, # "all", "llm", "embed" "none"
                #citation_qa_template=
            )
            return query_engine.query(msg)
    return None


def get_chunk(chunk_id):
    indexes = [idx for idx in (_load_index(BASE_INDEX_DIR), _load_index(SESSION_INDEX_DIR)) if idx is not None]
    for index in indexes:
        try:
            chunk_content = index.docstore.get_node(chunk_id).text
            return chunk_content
        except(ValueError, KeyError):
            continue
    return None


def resp_source(query_resp)->list[dict[str, str]]:
    sources = []
    # response.source_nodes[0].node.metadata.keys() = (['page_label', 'file_name', 'file_path', 'file_type', 'file_size', 'creation_date', 'last_modified_date'])
    for src in query_resp.source_nodes:
        src_name = src.node.metadata.get("file_name", "Unknown File Name")
        src_score = round(src.score, 2)
        src_page = src.node.metadata.get("page_label")
        src_type = src.node.metadata.get("file_type")
        src_text = get_chunk(src.node.node_id)                             #get_content(metadata_mode=MetadataMode.NONE)
        file_path = src.node.metadata.get("file_path")

        source_info = f"""Name: {src_name}\nScore:({src_score})\nType: {src_type}\nDocument: \n {src_text}"""

        source_metadata = {"source_info":source_info, "src_type":src_type, "file_path":file_path, "src_page":src_page, "src_score":src_score}

        sources.append(source_metadata)
    return sources
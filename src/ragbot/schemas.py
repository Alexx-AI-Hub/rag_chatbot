from dataclasses import dataclass, field


@dataclass
class RAGSettings:
    """Retrieval, chunking, and model/generation settings."""
    # Retrieval
    top_k: int = 2
    min_relevance_score: float = 0.05

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 150

    # Models / Generation
    llm_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text:latest"
    temperature: float = 0.1
    context_window: int = 4096
    max_tokens: int = 512
    streaming:bool = True
    metadata_mode:str = "all"
    online:bool = True


# ChatMessage


# ChatResponse
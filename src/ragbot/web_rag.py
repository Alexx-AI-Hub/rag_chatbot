import time, logging as log
from dataclasses import asdict
from urllib.parse import urlparse
from ddgs import DDGS
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.readers.web import TrafilaturaWebReader


from ragbot import rag
from ragbot.schemas import Chunk, SearchHit, RAGSettings

MAX_SEARCH_RESULTS = 50
SNIPPET_TOP_K = 10


BLOCKED_DOMAINS = {
    "ratsit",
    "eniro",
    "birthday",
    "instagram",
    "facebook",
    "youtube",
    "hitta",
}


def skip_url(url: str) -> bool:
    """Return True when a URL belongs to a blocked domain."""
    domain = urlparse(url).netloc.lower()
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def doc_to_textnode(doc: Document) -> TextNode | None:
    """Convert a fetched web document into a single TextNode."""
    content = (doc.get_content() or "").strip()
    if not content:
        return None

    return TextNode(
        text=content,
        metadata={
            "url": ((doc.metadata or {}).get("url") or "").strip(),
            "title": ((doc.metadata or {}).get("title") or "").strip(),
        },
    )


def _split_web_documents(web_docs: list[Document], settings: RAGSettings) -> list[TextNode]:
    """Split fetched web documents into chunk-sized TextNodes."""
    splitter: SentenceSplitter = rag.build_sentence_splitter(settings)
    chunk_nodes: list[TextNode] = []

    for doc in web_docs:
        base_node = doc_to_textnode(doc)
        if not base_node:
            continue
        chunks = splitter.split_text_metadata_aware(base_node.text, "")
        chunk_nodes.extend(
            TextNode(text=chunk, metadata=base_node.metadata)
            for chunk in chunks
            if chunk.strip()
        )

    return chunk_nodes


def fast_web_search(query: str) -> list[SearchHit]:
    """Search DuckDuckGo and return unique `SearchHit` items.

    The function normalizes raw search rows into `SearchHit(url, title, content)`,
    removes empty URLs, and deduplicates by URL while preserving first-seen order.
    Raises `RuntimeError` if the upstream web search call fails.
    """
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(
                query,
                region="se-sv",
                timelimit=None,
                max_results=MAX_SEARCH_RESULTS,
                safesearch="on",
            )
            sources = [
                SearchHit(
                    url=(row.get("href") or "").strip(),
                    title=(row.get("title") or "").strip(),
                    content=(row.get("body") or "").replace("\n", " ").strip(),
                )
                for row in raw_results
            ]
            seen: set[str] = set()
            sources = [
                src for src in sources
                if src.url and not skip_url(src.url) and not (src.url in seen or seen.add(src.url))
                ]
            log.debug(
                "Fast search urls after filtering:\n%s",
                "\n".join(f"- {src.url}" for src in sources) or "<no urls>",
            )

    except Exception as e:
        log.exception("For Query %s fast-search Exception: %s", query, e)
        return []


    return sources


def filter_fast_sources(query: str, sources: list[SearchHit], embed_model: BaseEmbedding, llm_model: LLM) -> list[SearchHit]:
    """Rank snippet-level sources with BM25 and return the top hits."""
    txt_nodes = [
        TextNode(
            text=f"{source.title}.{source.content}".strip(),
            metadata={"url": source.url, "title": source.title, "content": source.content},
        )
        for source in sources if source.url and source.content
    ]
    if not txt_nodes:
        return []

    scored_nodes = rag.retrieve_bm25(
        query=query,
        nodes=txt_nodes,
        top_k=SNIPPET_TOP_K,
    )
    filtered_sources: list[SearchHit] = [
        SearchHit(
            url=((node.metadata or {}).get("url") or "").strip(),
            title=((node.metadata or {}).get("title") or "").strip(),
            content=((node.metadata or {}).get("content") or "").strip(),
            score=node.score or None
        )
        for node in scored_nodes
    ]
    log.debug(
        "BM25 selected urls:\n%s",
        "\n".join(f"- {src.url}" for src in filtered_sources if src.url) or "<no urls>",
    )

    return [src for src in filtered_sources if src.url and src.content]


def _retrieve_full_web_sources(sources: list[SearchHit]) -> list[Document]:
    """Fetch full-page text for selected source URLs.

    Uses `TrafilaturaWebReader` to load page content and keeps only minimal
    metadata needed downstream (`url`, `title`). Documents with missing text
    or unresolved URL are skipped.
    """
    urls = [source.url for source in sources if source.url]
    if not urls:
        return []
    try:
        web_documents = TrafilaturaWebReader().load_data(
            urls=urls,
            include_comments=False,
            include_links=False,
            include_formatting=False,
            include_tables=True,
            favor_precision=True,
            deduplicate=True,
            show_progress=True,
        )
        valid_web_documents = [
            doc for doc in web_documents
            if ((doc.metadata or {}).get("url") or "").strip()
               and (doc.get_content() or "").strip()
        ]
        log.debug(
            "Readable urls:\n%s",
            "\n".join(
                f"- {((doc.metadata or {}).get('url') or '').strip()}"
                for doc in valid_web_documents
            ) or "<no urls>",
        )
        return valid_web_documents
    except Exception as e:
        log.exception("For Url's: %s full-Web-Search Exception: %s", urls, e)
        return []


def get_top_k_chunks(query: str, settings: RAGSettings, embed_model: BaseEmbedding, llm_model: LLM) -> list[Chunk]:
    """Run the web retrieval pipeline and return the top chunks."""
    log.info(f"Getting top_k Web-Chunks for Query: {query}")
    if sources := fast_web_search(query):

        if selected_sources := filter_fast_sources(query, sources, embed_model, llm_model):

            if web_docs := _retrieve_full_web_sources(selected_sources):

                chunk_nodes = _split_web_documents(web_docs, settings)
                if chunk_nodes:
                    log.debug(
                        "Chunks before fusion:\n%s",
                        "\n".join(
                            f"- title={((node.metadata or {}).get('title') or '').strip()} "
                            f"url={((node.metadata or {}).get('url') or '').strip()} "
                            for node in chunk_nodes
                        ) or "<no chunks>",
                    )
                    try:
                        ranked_nodes = rag.retrieve_fusion(
                            query=query,
                            nodes=chunk_nodes,
                            embed_model=embed_model,
                            top_k=max(1, int(settings.top_k)),
                            llm=llm_model,
                        )
                        log.debug(
                            "Chunks after fusion:\n%s",
                            "\n".join(
                                f"- title={((node.metadata or {}).get('title') or '').strip()} "
                                f"url={((node.metadata or {}).get('url') or '').strip()} "
                                f"score={node.score}"
                                for node in ranked_nodes
                            ) or "<no chunks>",
                        )
                        chunks = rag.norm_node_to_chunk(ranked_nodes)

                        log.info("Got Web-Chunks from:\n%s",
                                 "\n".join(f"- {chunk.url}" for chunk in chunks if chunk.url)
                        )
                        return chunks

                    except Exception as e:
                        log.exception("Fusion-Retriever(bm25+V) Exception: %s", e)
    return []


def rag_web_search(user_query: str, settings: RAGSettings, embed_model: BaseEmbedding, llm_model: LLM) -> list[Chunk]:
    """Call get_top_k_chunks() for backward compatibility."""
    return get_top_k_chunks(user_query, settings, embed_model, llm_model)


if __name__ == "__main__":
    query = "What can you tell me about the latest improvement in AI?"

    start = time.perf_counter()
    test_chunks = rag_web_search(
        user_query=query,
        settings=RAGSettings(),
        embed_model=rag.build_openai_compatible_embedding(model_name="nomic-embed-text"),
        llm_model= rag.build_openai_compatible_llm(model_name="llama3.2:3b"))
    end = time.perf_counter()

    print(f"======QUERY: {query} =======")
    print("======WEB SEARCH RESULTING CHUNKS=======")
    for test_chunk in test_chunks:
        print(f'Title: --> {asdict(test_chunk)["title"]}\nScore({asdict(test_chunk)["score"]})\nUrl: --> {asdict(test_chunk)["url"]}\n')
        print(f'Content: \n{asdict(test_chunk)["content"]}\n')
    print(f"\n\nLATENCY: {(end - start)}Seconds")

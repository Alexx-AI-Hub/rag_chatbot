from ddgs import DDGS
from duckduckgo_search.exceptions import RatelimitException
import logging as log
import llm
from config import QA_PROMPT_TEMPLATE, OPTIMIZE_SEARCH_QUERY_PROMPT
# DuckDuckGo Search Engine
ddg_search = DDGS()


def _duckduckgo_search(query:str, model:str )->list[dict[str, str]] | list[str]:
    # DuckDuckGo Rate-Limit = 1req/s OR 20req/30mins
    search_query_prompt = OPTIMIZE_SEARCH_QUERY_PROMPT.format(user_query=query)
    search_query = llm.gen_response(prompt=search_query_prompt, model=model, temp=0.0, top_p=1.0)
    log.info("Search Query to DuckDuckGo: %s",search_query)
    try:
        results = ddg_search.text(
            query=search_query,
            max_results=50,
            safesearch="none",
            region="se-sv",
            timelimit=None,
        )
    except RatelimitException:
        log.warning("DuckDuckGo RatelimitException")
        return [{
            "title": "Rate limited",
            "href": "",
            "Body": "DuckDuckGo rate limit reached.",
            "structured_result": "Web search unavailable right now due to rate limit.",
        }]


    results_list = []
    for i, res in enumerate(results, start=1):
        struct_res =f"""Result {i}
        Title: {res.get('title')}
        Link: {res.get('href')}
        Body: {res.get('body')}"""

        web_results ={
            "title":res.get('title'),
            "href":res.get('href'),
            "Body": res.get('body'),
            "structured_result":struct_res,
        }
        results_list.append(web_results)
    return results_list


def chat_resp_with_duck_search(client, query: str, model: str, temp:float, top_p:float) -> str:
    # Use user's primary search function and support its fallback return types.
    res_list = _duckduckgo_search(query, model)

    struct_context = "\n".join(res.get("structured_result") for res in res_list if res.get("structured_result"))

    prompt = QA_PROMPT_TEMPLATE.format(query=query, context=struct_context)
    print("=== INPUT ===")
    print(prompt)
    response = llm.gen_response(client=client, prompt=prompt, model=model, temp=temp, top_p=top_p)

    log.info("Chat Response from LLM using DuckDuckDo Search-Engine-Result: %s", response)
    return response
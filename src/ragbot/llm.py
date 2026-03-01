from openai import OpenAI
import ollama
import httpx
import logging as log


# OpenAI Client(Responses-API)
def get_openai_client(api_key:str="KEY", base_url:str="http://localhost:11434/v1", timeout:int=15 ):
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def chat_response(
                    client:OpenAI,
                    user_prompts: list[dict[str, object]]|str,
                    system_prompt: str,
                    model: str = "llama3.2:3b",
                    temp:float=0.2,
                    top_p:float=0.95):
    return client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompts,
        temperature=temp,
        top_p=top_p,
    )


def gen_response(client:OpenAI, prompt, model:str, temp:float=0.2, top_p:float=0.95):
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temp,
        top_p=top_p,
    )
    return response.output_text


def get_openai_list(client:OpenAI):
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        log.error(e)
        return []


def _get_ollama_list():
    """Return available local Ollama model names or raise a runtime error."""
    try:
        return [m["model"] for m in ollama.list()["models"]]
    except httpx.ConnectError as e:
        error_msg = f"No response from Ollama server --> Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)
    except ollama.ResponseError as e:
        error_msg = f"There was an error in the response from Ollama. Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)
    except (KeyError, TypeError) as e:
        error_msg = f"Failed to parse models (this can occur because of an change in their API). Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred. Error: {e}"
        log.error(error_msg)
        raise RuntimeError(error_msg)
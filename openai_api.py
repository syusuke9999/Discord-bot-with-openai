import json
import httpx
from httpx import Timeout
from typing import List, Dict
from main import OPENAI_API_KEY
import logging
import wandb
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger('OpenAI_API')
logger.setLevel(logging.WARNING)


# APIへのリクエストが失敗したさいに、リトライするデコレーター
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=3, max=60))
async def call_openai_api(hyper_parameters, system_message, new_message, message_history: List[Dict] = None):
    wandb.init(project="discord-bot-llm-trace", group="openai_api", tags=["call_openai_api"])
    temperature = float(hyper_parameters["temperature"])
    model_name = str(hyper_parameters["model_name"])
    max_tokens = int(hyper_parameters["max_tokens"])
    top_p = float(hyper_parameters["top_p"])
    frequency_penalty = float(hyper_parameters["frequency_penalty"])
    presence_penalty = float(hyper_parameters["presence_penalty"])
    if message_history is None:
        message_history = []
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "temperature": temperature,
        "model": model_name,
        "messages": [system_message] + message_history + [new_message],
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    timeout = Timeout(120)  # Set timeout to 120 seconds
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception if the status code is not in the 200 range.
            return response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(f"HTTP status error {exc}")
        # Handle HTTP status error
    except json.JSONDecodeError as e:
        logger.exception(f"A JSON decode error occurred: {e}")
        # Handle JSON decode error
    except httpx.TimeoutException as e:
        logger.exception(f"A timeout error occurred: {e}")
        # Handle timeout error
    except httpx.RequestError as e:
        logger.exception(f"A network error occurred: {e}")
        # Handle network error
    wandb.finish()

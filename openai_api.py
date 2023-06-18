import json
import httpx
from httpx import Timeout
from main import model_name
from main import OPENAI_API_KEY
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger('OpenAI_API')
logger.setLevel(logging.WARNING)


# APIへのリクエストが失敗したさいに、リトライするデコレーター
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=3, max=60))
async def call_openai_api(name_of_model, system_message, new_message, message_history):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "temperature": 0,
        "model": name_of_model,
        "messages": [system_message] + message_history + [new_message],
        "max_tokens": 1000,
        "frequency_penalty": 0.6,
        "presence_penalty": 0,
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

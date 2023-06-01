import asyncio
import json
from httpx import Timeout
import httpx
from main import model_name
from main import OPENAI_API_KEY


async def call_openai_api(system_message, new_message, message_history):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "temperature": 0.7,
        "model": model_name,
        "messages": [system_message] + message_history + [new_message],
        "max_tokens": 500,
        "frequency_penalty": 0,
        "presence_penalty": 0.6,
    }
    # 最大リトライ回数
    max_retries = 3
    # リトライ間隔（秒）
    retry_interval = 10
    # タイムアウトを120秒に設定
    timeout = Timeout(120)  # Set timeout to 120 seconds
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, data=json.dumps(data))
                response.raise_for_status()  # ステータスコードが200系以外の場合に例外を発生させる
                return response.json()
        except (httpx.HTTPStatusError, Exception) as e:
            print(f"An error occurred: {e}")
            if i < max_retries - 1:  # 最後のリトライでなければ、次のリトライまで待つ
                print(f"Retrying in {retry_interval} seconds...")
                await asyncio.sleep(retry_interval)
                print("Retrying now!")
            else:  # 最後のリトライでもエラーが発生した場合、エラーを再度送出する
                raise

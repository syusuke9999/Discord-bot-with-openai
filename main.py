import os
import time  # 追加: timeモジュールをインポート
import discord
from discord.ext import commands
import asyncio
import tiktoken
from tiktoken.core import Encoding
import httpx
import redis
from asyncio import sleep
import json
import logging
import datetime
import pytz

debug_mode = False

# Discord接続を初期化
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

jst = pytz.timezone('Asia/Tokyo')

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_TEST_TOKEN = os.getenv("DISCORD_TEST_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')

if not debug_mode:
    # Redis接続を初期化
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD)
    print("Redis connection established!")

model_name = "gpt-4"
encoding: Encoding = tiktoken.encoding_for_model(model_name)
MAX_TOKENS = 3000

logger = logging.getLogger('discord')
logger.setLevel(logging.WARNING)


def count_tokens(text):
    tokens = encoding.encode(text)
    tokens_count = len(tokens)
    return tokens_count


async def call_openai_api(system_message, new_message, user_key, self):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "temperature": 0.7,
        "model": model_name,
        "messages": [
            system_message,
            new_message,
            *self.message_history[user_key]
        ],
        "max_tokens": 500,
        "frequency_penalty": 0,
        "presence_penalty": 0.6,
    }
    # 最大リトライ回数
    max_retries = 10
    # リトライ間隔（秒）
    retry_interval = 15

    for i in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, data=json.dumps(data))
                response.raise_for_status()  # ステータスコードが200系以外の場合に例外を発生させる
                return response.json()
        except (httpx.HTTPStatusError, Exception) as e:
            print(f"An error occurred: {e}")
            print(httpx.HTTPStatusError.with_traceback(self, tb=MyBot.on_message))
            if i < max_retries - 1:  # 最後のリトライでなければ、次のリトライまで待つ
                print(f"Retrying in {retry_interval} seconds...")
                await asyncio.sleep(retry_interval)
            else:  # 最後のリトライでもエラーが発生した場合、エラーを再度送出する
                raise


class MyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_history = {}
        self.total_tokens = 0  # 修正: トークン数の合計を保持するための変数を追加

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    async def on_message(self, message):
        await super().on_message(message)  # 追加: commands.Botのon_messageメソッドを呼び出す
        if message.author == self.user:
            return
        # メンションされた場合
        print("mentioned!  Message content: ", message.content)
        if self.user in message.mentions:
            # メッセージの内容を表示
            print("Message content: ", message.content)
            # デバックモードでない場合、Redisからメッセージ履歴を読み込む
            if not debug_mode:
                # メンションしたユーザーのIDを取得
                user_id = str(message.author.id)
                user_name = message.author.display_name
                user_key = f'{user_id}_{user_name}'
                # Redisからメッセージ履歴を読み込む
                start_time = time.time()  # 追加: Redisサーバーからメッセージの履歴を取得する前に時間を記録
                message_history_json = r.get(f'message_history_{user_key}')
                end_time = time.time()  # 追加: リクエストの後に時間を記録
                elapsed_time = end_time - start_time  # 追加: 経過時間を計算
                print(f"Redisサーバーからメッセージの履歴を取得するのにかかった時間:  {elapsed_time} 秒。")  # 追加: 経過時間を表示
                if message_history_json is not None:
                    self.message_history[user_key] = json.loads(message_history_json)
                else:
                    self.message_history[user_key] = []
            # デバッグモードの場合、メッセージ履歴をリセットする
            else:
                self.message_history = {}
            user_id = str(message.author.id)
            user_name = message.author.display_name
            user_key = f'{user_id}_{user_name}'
            print("user_key: " + user_key + " message.content: ", message.content)
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            system_message_content = f"Today is the year {now_of_year}, the month is {now_of_month} " \
                                     f"and the date {now_of_day}. The current time is {now_of_time}." \
                                     f"You are a Discord bot on a channel for people who are interested in a " \
                                     f"Discord bot that works with OpenAI's API, and would like to have a " \
                                     f"conversation about the possibilities of a Discord bot that works with " \
                                     f"OpenAI's API.Avoid mentioning the topic of the prompt and greet them " \
                                     f"considering the current time.Don't use English, " \
                                     f"please communicate only in Japanese."
            system_message = {"role": "system", "content": system_message_content}
            print("user:" + self.user.display_name + "message.content: ", message.content)
            new_message = {"role": "user", "content": message.content}
            print("Getting response from OpenAI API...")
            start_time = time.time()  # 追加: OpenAIのAPIへのリクエストを送信する前に時間を記録
            async with message.channel.typing():
                response = await call_openai_api(system_message, new_message, user_key, self)
                if response is not None:
                    print(response)
                else:
                    print("OpenAI's API call failed.")
            end_time = time.time()  # 追加: リクエストの後に時間を記録
            elapsed_time = end_time - start_time  # 追加: 経過時間を計算
            print(f"OpenAIのAPIへのリクエストから応答があるまでに要した時間: {elapsed_time} 秒。")  # 追加: 経過時間を表示
            bot_response = response['choices'][0]['message']['content']
            print("ボットの応答: ", bot_response)
            bot_response_tokens = count_tokens(bot_response)
            print("bot_response_tokens: ", bot_response_tokens)
            # デバッグモードでない場合はボットからの応答を含めたメッセージ履歴をRedisに保存
            if not debug_mode:
                # 新しいメッセージを追加する前に、既存のメッセージ履歴をチェック
                for i, old_message in list(enumerate(self.message_history[user_key])):  # list()を使用してリストのコピーを作成
                    # ユーザーのメッセージと同じ内容のメッセージが存在する場合
                    if old_message["role"] == "user" and old_message["content"] == new_message["content"]:
                        # 次のメッセージが存在し、そのメッセージがボットからのものであることを確認
                        if i + 1 < len(self.message_history[user_key]) and \
                                self.message_history[user_key][i + 1]["role"] == "assistant":
                            # そのメッセージと次のメッセージ（ボットの応答）を削除
                            del self.message_history[user_key][i:i + 2]
                # メッセージ履歴に含まれる全てのメッセージのトークン数を計算
                message_tokens = sum(count_tokens(json.dumps(m)) for m in self.message_history[user_key])
                # 新しいメッセージとシステムメッセージのトークン数を追加
                message_tokens += count_tokens(json.dumps(new_message)) + count_tokens(json.dumps(system_message)) + \
                    count_tokens(json.dumps(bot_response))
                # 新しいメッセージを追加するとトークン制限を超える場合、古いメッセージを削除する。
                while message_tokens > MAX_TOKENS:
                    # 最初のメッセージを削除する
                    removed_message = self.message_history[user_key].pop(0)
                    # 削除したメッセージのトークン数を引く
                    message_tokens -= count_tokens(json.dumps(removed_message))
                # トークン数が制限以下になったら新しいメッセージを追加
                self.message_history[user_key].append(new_message)
                self.message_history[user_key].append({"role": "assistant", "content": bot_response})
                # メッセージ履歴に含まれる全てのメッセージのトークン数を計算
                self.total_tokens = sum(count_tokens(json.dumps(m)) for m in self.message_history[user_key])
                # 新しいメッセージとシステムメッセージのトークン数を追加
                self.total_tokens += count_tokens(json.dumps(new_message)) + count_tokens(json.dumps(system_message))
                if not debug_mode:
                    # メッセージ履歴をRedisに保存し、TTLを設定
                    message_history_json = json.dumps(self.message_history[user_key])
                    start_time = time.time()  # 追加: Redisサーバーへメッセージの履歴を保存する前に時間を記録
                    r.set(f'message_history_{user_key}', message_history_json)
                    r.expire(f'message_history_{user_key}', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
                    end_time = time.time()  # 追加: 保存の後に時間を記録
                    elapsed_time = end_time - start_time  # 追加: 経過時間を計算
                    print(f"Redisへ会話履歴を保存するのにかかった時間: {elapsed_time} 秒。")  # 追加: 経過時間を表示
                    print("message was saved to redis!")
            # ボットからの応答の文字数に応じて、タイピング中のアニメーションの表示時間を調整する
            typing_time = min(max(len(bot_response) / 50, 3), 9)  # タイピングスピードを変えるために、分割数を調整する
            print("typing_time: ", typing_time)
            print("await sending message to discord with async typing function!")
            async with message.channel.typing():
                await sleep(typing_time)  # 計算された時間まで待つ
                await message.reply(bot_response)
                print("massage have sent to discord!")
            print("message_history: ", self.message_history)


def main():
    global debug_mode
    intents = discord.Intents.all()
    intents.messages = True
    intents.guilds = True
    client = MyBot(command_prefix='!', intents=intents)
    if not debug_mode:
        client.run(DISCORD_TOKEN)
    else:
        client.run(DISCORD_TEST_TOKEN)


if __name__ == "__main__":
    main()

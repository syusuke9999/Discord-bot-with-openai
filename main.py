import os
import discord
from discord.ext import commands
import openai
import tiktoken
from tiktoken.core import Encoding
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


class MyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_history = {}

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    async def on_message(self, message):
        await super().on_message(message)  # 追加: commands.Botのon_messageメソッドを呼び出す
        print("message.content: ", message.content)
        if message.author == self.user:
            return
        # メンションされた場合
        if self.user.mentioned_in(message):
            # デバックモードでない場合、Redisからメッセージ履歴を読み込む
            if not debug_mode:
                # メンションしたユーザーのIDを取得
                user_id = str(message.author.id)
                user_name = message.author.display_name
                user_key = f'{user_id}_{user_name}'
                # Redisからメッセージ履歴を読み込む
                message_history_json = r.get(f'message_history_{user_key}')
                if message_history_json is not None:
                    self.message_history[user_key] = json.loads(message_history_json)
                else:
                    self.message_history[user_key] = []
            # デバッグモードの場合、メッセージ履歴をリセットする
            else:
                self.message_history = {}
            print("mentioned!")
            user_id = str(message.author.id)
            user_name = message.author.display_name
            user_key = f'{user_id}_{user_name}'
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            system_message_content = f"Today is the year {now_of_year}, the month is {now_of_month} and the date " \
                                     f"{now_of_day}. The current time is {now_of_time}. " \
                                     f"You are a Discord bot residing in a Discord channel for people " \
                                     f"interested in \"Discord bots that work with OpenAI's API\". Please have a " \
                                     f"conversation with users about how \"Discord bots running on OpenAI's API\" " \
                                     f"can be useful to them. Avoid mentioning the topic of the prompt and greet them " \
                                     f"considering the current time. Don't use English, " \
                                     f"please communicate only in Japanese."
            system_message = {"role": "system", "content": system_message_content}
            print(system_message)
            print("user:" + self.user.display_name + "message.content: ", message.content)
            new_message = {"role": "user", "content": message.content}
            message_tokens = count_tokens(message.content)
            system_message_tokens = count_tokens(system_message["content"])
            # 新しいメッセージを追加するとトークン制限を超える場合、古いメッセージを削除する。
            total_tokens = MAX_TOKENS - (message_tokens + system_message_tokens)
            while sum(count_tokens(m["content"]) for m in self.message_history[user_key]) > total_tokens:
                self.message_history[user_key].append(new_message)
            self.message_history[user_key].append(new_message)
            if not debug_mode:
                # メッセージ履歴をRedisに保存し、TTLを設定
                message_history_json = json.dumps(self.message_history[user_key])
                r.set(f'message_history_{user_key}', message_history_json)
                r.expire(f'message_history_{user_key}', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
                print("message was save to redis!")
            print("Getting response from OpenAI API...")
            response = openai.ChatCompletion.create(
                temperature=0.7,
                model=model_name,
                messages=[
                    system_message,
                    *self.message_history[user_key]
                ],
                max_tokens=500,
                frequency_penalty=0,
                presence_penalty=0.6,
            )
            bot_response = response['choices'][0]['message']['content']
            print("bot_response: ", bot_response)
            print("bot_response_tokens: ", count_tokens(bot_response))
            # メッセージ履歴にボットの返答を追加
            self.message_history[user_key].append({"role": "assistant", "content": bot_response})
            # メッセージ履歴をRedisにすぐに保存
            message_history_json = json.dumps(self.message_history[user_key])
            r.set(f'message_history_{user_key}', message_history_json)
            r.expire(f'message_history_{user_key}', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
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
    client = MyBot(command_prefix='!', intents=intents)
    if debug_mode:
        client.run(DISCORD_TEST_TOKEN)
    else:
        client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

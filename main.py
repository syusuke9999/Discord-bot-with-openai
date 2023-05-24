import os
import discord
import openai
import tiktoken
from tiktoken.core import Encoding
import redis
import json
import logging
import datetime
import pytz

jst = pytz.timezone('Asia/Tokyo')

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDISGREEN_URL = os.environ.get('REDISGREEN_URL')

model_name = "gpt-4"
encoding: Encoding = tiktoken.encoding_for_model(model_name)
MAX_TOKENS = 5000

logger = logging.getLogger('discord')
logger.setLevel(logging.WARNING)

# Redis接続を初期化
r = redis.StrictRedis.from_url(REDISGREEN_URL)


def count_tokens(text):
    tokens = encoding.encode(text)
    tokens_count = len(tokens)
    return tokens_count


class MyBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Redisからメッセージ履歴を読み込む
        message_history_json = r.get('message_history')
        if message_history_json is not None:
            self.message_history = json.loads(message_history_json)
        else:
            self.message_history = []

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    async def on_message(self, message):
        print("message.content: ", message.content)
        if message.author == self.user:
            return
        if self.user.mentioned_in(message):
            print("mentioned!")
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            system_message_content = f"Today is the year {now_of_year}, " \
                                     f"the month is {now_of_month} and the date {now_of_day}. " \
                                     f"The current time is {now_of_time}." \
                                     f"You are a Discord bot joining a Discord channel where people enjoy " \
                                     f"online games. Have fun talking about Dead by Daylight and Apex legend, " \
                                     f"and other everyday conversation. with the channel participants." \
                                     f"Avoid mentioning the topic of the prompt and greet them considering " \
                                     f"the current time." \
                                     f"Don't use English, please communicate only in Japanese."
            system_message = {"role": "system", "content": system_message_content}
            print(system_message)
            print("user:" + self.user.display_name + "message.content: ", message.content)
            new_message = {"role": "user", "content": message.content}
            message_tokens = count_tokens(message.content)
            system_message_tokens = count_tokens(system_message["content"])
            # 新しいメッセージを追加するとトークン制限を超える場合、古いメッセージを削除する。
            total_tokens = MAX_TOKENS - (message_tokens + system_message_tokens)
            while sum(count_tokens(m["content"]) for m in self.message_history) > total_tokens:
                self.message_history.pop(0)
            self.message_history.append(new_message)
            # メッセージ履歴をRedisに保存し、TTLを設定
            message_history_json = json.dumps(self.message_history)
            r.set('message_history', message_history_json)
            r.expire('message_history', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
            print("message was save to redis!")
            response = openai.ChatCompletion.create(
                temperature=0.7,
                model=model_name,
                messages=[
                    system_message,
                    *self.message_history
                ]
            )
            print("massage have sent to discord with await function!")
            await message.channel.trigger_typing()  # タイピングアニメーションを開始
            print("Getting response from OpenAI API...")
            bot_response = response['choices'][0]['message']['content']
            self.message_history.append({"role": "assistant", "content": bot_response})
            print("bot_response: ", bot_response)
            print("bot_response_tokens: ", count_tokens(bot_response))
            print("massage have sent to discord with await function!")
            await message.channel.trigger_typing()  # タイピングアニメーションを開始
            await message.reply(bot_response)
            print("message was send to discord!")


def main():
    intents = discord.Intents.all()
    client = MyBot(intents=intents)
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

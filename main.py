import os
import discord
import openai
import tiktoken
from tiktoken.core import Encoding
import redis
import json
import logging

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT: int = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

model_name = "gpt-4"
encoding: Encoding = tiktoken.encoding_for_model(model_name)
MAX_TOKENS = 5000

logger = logging.getLogger('discord')
logger.setLevel(logging.WARNING)

# Redis接続を初期化
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD)


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
        if message.content.startswith('/check'):
            print("check!")
            # '/check' command functionality
            playing_dbd_members = []
            for member in message.guild.members:
                if member.activity and member.activity.name == 'Dead by Daylight':
                    playing_dbd_members.append(member.name)
            if len(playing_dbd_members) > 0:
                message_content = ', '.join(playing_dbd_members)
                await message.channel.send(f'{message_content} はDead by Daylightをプレイしています！')
            else:
                await message.channel.send('現在Dead by Daylightをプレイしているサーバーメンバーはいません。')
            return  # '/check' アクションが実行された場合、これ以上処理を続行しないでください。
        if self.user.mentioned_in(message):
            print("mentioned!")
            print("user:" + self.user.display_name + "message.content: ", message.content)
            new_message = {"role": "user", "content": message.content}
            message_tokens = count_tokens(message.content)
            system_message = {"role": "system",
                              "content": "You are a chatbot joining a small Discord channel focused on "
                                         "Dead by Daylight. Please communicate only in Japanese and engage in "
                                         "discussions related to the game."}
            system_message_tokens = count_tokens(system_message["content"])
            # 新しいメッセージを追加するとトークン制限を超える場合、古いメッセージを削除する。
            total_tokens = MAX_TOKENS - (message_tokens + system_message_tokens)
            while sum(count_tokens(m["content"]) for m in self.message_history) > total_tokens:
                self.message_history.pop(0)
            self.message_history.append(new_message)
            # メッセージ履歴をRedisに保存し、TTLを設定
            message_history_json = json.dumps(self.message_history)
            r.set('message_history', message_history_json)
            r.expire('message_history', 3600 * 24 * 20)  # TTLを20日間（1,728,000秒）に設定
            print("message was save to redis!")
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    system_message,
                    *self.message_history
                ]
            )
            bot_response = response['choices'][0]['message']['content']
            self.message_history.append({"role": "assistant", "content": bot_response})
            print("bot_response: ", bot_response)
            print("bot_response_tokens: ", count_tokens(bot_response))
            print("massage have sent to discord with await function!")
            await message.channel.send(bot_response)
            print("message was send to discord!")


def main():
    intents = discord.Intents.default()
    intents.message_contents = True
    client = MyBot(intents=intents)
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

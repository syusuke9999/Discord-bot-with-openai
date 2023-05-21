import os
import discord
import openai
import tiktoken
from tiktoken.core import Encoding

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
encoding: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
MAX_TOKENS = 4000


def count_tokens(text):
    tokens = encoding.encode(text)
    tokens_count = len(tokens)
    return tokens_count


class MyBot(discord.Client):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_history = []  # `__init__`メソッド内で初期化

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content.startswith('check!'):
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
            new_message = {"role": "user", "content": message.content}
            message_tokens = count_tokens(message.content)

            system_message = {"role": "system",
                              "content": "You are a chatbot participating in a small Discord channel for "
                                         "people who play Dead by Daylight.Please be willing to discuss Dead "
                                         "by Daylight topics. Please do not use English in your "
                                         "conversations. Please use only Japanese in all conversations."}
            system_message_tokens = count_tokens(system_message["content"])

            # Remove old messages if adding a new message exceeds the token limit
            total_tokens = MAX_TOKENS - (message_tokens + system_message_tokens)
            while sum(count_tokens(m["content"]) for m in self.message_history) > total_tokens:
                self.message_history.pop(0)

            self.message_history.append(new_message)

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    system_message,
                    *self.message_history
                ]
            )
            bot_response = response['choices'][0]['message']['content']
            self.message_history.append({"role": "assistant", "content": bot_response})
            await message.channel.send(bot_response)


def main():
    intents = discord.Intents.default()
    client = MyBot(intents=intents)
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

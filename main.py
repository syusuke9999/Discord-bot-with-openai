import os
import time
import discord
from discord.ext import commands
import tiktoken
from tiktoken.core import Encoding
import redis
from asyncio import sleep
import json
import logging
from system_message import SystemMessage, Topic
from RetrievalQA import RetrievalQAFromFaiss

debug_mode = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
TOPIC_ENUM = "DEAD_BY_DAY_LIGHT"
THIS_TOPIC_ENUM = Topic.__members__.get(TOPIC_ENUM)

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


async def send_message(message, bot_response_for_answer):
    await message.reply(bot_response_for_answer)
    print("await reply message to discord with async typing function!")


class MyBot(commands.Bot):
    def __init__(self, command_prefix, intents, enum_of_topic):
        super().__init__(command_prefix, intents=intents)
        self.topic_enum = enum_of_topic
        self.message_histories = {}
        self.total_tokens = 0  # トークン数の合計を保持するための変数を追加
        self.special_channel_ids = [1117363032172003328, 1117412783592591460]  # 特定のチャンネルのIDをリストで設定します。

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    def remove_duplicate_messages(self, user_key, new_message):
        # メッセージの履歴の中に、ユーザーの発言が同じ物が複数あった場合、1つだけ残して消す。
        new_message_history = [self.message_histories[user_key][0], self.message_histories[user_key][1]]
        skip_next = False
        for i, old_message in enumerate(self.message_histories[user_key][2:], start=2):
            if skip_next:
                skip_next = False
                continue
            if old_message["role"] == "user" and old_message["content"] == new_message["content"]:
                if i + 1 < len(self.message_histories[user_key]) and \
                        self.message_histories[user_key][i + 1]["role"] == "assistant":
                    skip_next = True
                    continue
            new_message_history.append(old_message)
        self.message_histories[user_key] = new_message_history

    async def on_message(self, message):
        # commands.Botのon_messageメソッドを呼び出す
        await super().on_message(message)
        if message.author == self.user:
            return
        # メンションされた場合か、特定のチャンネルでメッセージが送信された場合のみ処理を続ける
        if self.user in message.mentions or message.channel.id in self.special_channel_ids:
            if self.user in message.mentions:
                print("mentioned!  Message content: ", message.content)
            elif message.channel.id in self.special_channel_ids:
                print("special channel!  Message content: ", message.content)
            # メッセージの内容を表示
            print("Message content: ", message.content)
            # メンションしたユーザーのIDと名前を取得
            user_id = str(message.author.id)
            user_name = message.author.display_name
            # ユーザーのIDと名前を組み合わせて、ユーザーを一意に識別するユーザーキーを作成
            user_key = f'{user_id}_{user_name}'
            # Redisサーバーからメッセージの履歴を取得する前に時間を記録
            start_time = time.time()
            # ユーザーキーを指定してRedisサーバーからメッセージの履歴を取得
            message_history_json = r.get(f'message_history_{user_key}')
            end_time = time.time()
            elapsed_time = end_time - start_time
            # 経過時間を表示
            print(f"Elapsed time to get data from Redis server: {elapsed_time} seconds")
            if message_history_json is not None:
                self.message_histories[user_key] = json.loads(message_history_json)
            else:
                self.message_histories[user_key] = []
            print(user_key + ":message = " + message.content)
            system_message_instance = SystemMessage(topic=self.topic_enum)
            system_message_content = system_message_instance.get_system_message_content()
            system_message_dict = {"role": "system", "content": system_message_content}
            print("システムメッージ: ", system_message_content)
            new_message_dict = {"role": "user", "content": message.content}
            print("Retrival QAを実行します。")
            start_time = time.time()
            retrival_qa = RetrievalQAFromFaiss()
            # クローリングしたデータからユーザーの質問に関係のありそうなものを探し、GPT-4が質問に対する答えだと判断した場合はここで答えが返ってくる
            # 関連する答えが見つからなかった場合はそのように答える
            bot_response_for_answer = await retrival_qa.GetAnswerFromFaiss(message.content)
            elapsed_time = time.time() - start_time
            print(f"The retrieval QA took {elapsed_time} seconds.")
            print("bot_response_for_answer: ", bot_response_for_answer)
            if bot_response_for_answer is not None:
                print("bot_response_for_answer: ", bot_response_for_answer)
                await send_message(message, bot_response_for_answer)
            # メッセージの履歴を更新
            user_message = str(message.content)
            self.update_message_histories_and_tokens(user_message, bot_response_for_answer, user_key)
            if not debug_mode:
                # メッセージ履歴をRedisに保存し、TTLを設定
                new_message = {"role": "user", "content": message.content}
                self.message_histories[user_key].append(new_message)
                self.message_histories[user_key].append({"role": "assistant",
                                                         "content": bot_response_for_answer})
                message_history_json = json.dumps(self.message_histories[user_key])
                # Redisサーバーへメッセージの履歴を保存するのにかかった時間を計測
                start_time = time.time()
                r.set(f'message_history_{user_key}', message_history_json)
                r.expire(f'message_history_{user_key}', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
                end_time = time.time()
                # 経過時間を計算して表示
                elapsed_time = end_time - start_time
                print(f"Elapsed time to save data to Redis server: {elapsed_time} seconds")  # 経過時間を表示
                self.update_message_histories_and_tokens(user_message, bot_response_for_answer, user_key)
                print("message_history: ", self.message_histories)

    def update_message_histories_and_tokens(self, user_message, bot_response, user_key):
        # メッセージ履歴に含まれる全てのメッセージのトークン数を計算
        total_tokens = sum(count_tokens(json.dumps(m)) for m in self.message_histories[user_key])
        # 新しいメッセージとボットの応答のトークン数を追加
        total_tokens += count_tokens(json.dumps(user_message)) + count_tokens(json.dumps(bot_response))
        # 新しいメッセージを追加するとトークン制限を超える場合、古いメッセージを削除する。
        while total_tokens > MAX_TOKENS:
            # 最初のメッセージを削除する
            removed_message = self.message_histories[user_key].pop(0)
            # 削除したメッセージのトークン数を引く
            total_tokens -= count_tokens(json.dumps(removed_message))
        # トークン数が制限以下になったら新しいメッセージとボットの応答を追加
        self.message_histories[user_key].append({"role": "user", "content": user_message})
        self.message_histories[user_key].append({"role": "assistant", "content": bot_response})
        # メッセージ履歴に含まれる全てのメッセージのトークン数を計算
        self.total_tokens = total_tokens

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        # ボイスチャンネルIDを指定します
        your_voice_chat_channel_id = 1003966899232702537
        if after.channel is not None and after.channel.id == your_voice_chat_channel_id:
            # チャンネルのメンバーが増えて2人以上いるか確認します
            if (before.channel is None and len(after.channel.members) >= 2) or \
                    (before.channel is not None and len(before.channel.members) < len(after.channel.members) and len(
                        after.channel.members) >= 2):
                # メンバーの名前を取得してカンマ区切りの文字列にします
                member_names = ""
                for participant in after.channel.members:
                    if member_names != "":
                        member_names += ", "
                    member_names += participant.name
                # テキストチャンネルIDを指定します
                your_text_chanel_id = 1003966898792312854
                # ボイスチャットに参加しているメンバーが2人以上いる場合、ユーザー名を指定してメッセージを送信する
                await self.get_channel(your_text_chanel_id).send(f'{member_names}さん、Dead by Daylightを楽しんで下さい。')


def main():
    # Discord接続を初期化
    intents = discord.Intents.all()
    # ボイスステータスのインテントを取得する意図を明視するため
    intents.voice_states = True
    bot = MyBot(command_prefix='!', intents=intents, enum_of_topic=THIS_TOPIC_ENUM)
    # ここにあなたのDiscordボットのトークンを指定します
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

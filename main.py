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

debug_mode = False

# Discord接続を初期化
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
topic_enum = os.getenv('TOPIC_ENUM')

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
        # commands.__init__を呼び出す
        super().__init__(*args, **kwargs)
        self.message_histories = {}
        self.topic = Topic[topic_enum]
        self.total_tokens = 0  # トークン数の合計を保持するための変数を追加

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    def remove_duplicate_messages(self, user_key, new_message):
        new_message_history = []
        skip_next = False
        for i, old_message in enumerate(self.message_histories[user_key]):
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
        # メンションされた場合
        print("mentioned!  Message content: ", message.content)
        if self.user in message.mentions:
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
            # Redisサーバーからのデータ取得にかかった時間を計算
            elapsed_time = end_time - start_time
            # 経過時間を表示
            print(f"Elapsed time to get data from Redis server: {elapsed_time} seconds")
            if message_history_json is not None:
                self.message_histories[user_key] = json.loads(message_history_json)
            else:
                self.message_histories[user_key] = []
            print("user_key: " + user_key + " message.content: ", message.content)
            system_message_instance = SystemMessage(topic=self.topic)
            system_message_content = system_message_instance.get_system_message_content()
            system_message = {"role": "system", "content": system_message_content}
            new_message = {"role": "user", "content": message.content}
            print("Getting response from OpenAI API...")
            # OpenAIのAPIへのリクエストを送信してから返事が返って来るまでの時間を測定する
            start_time = time.time()
            from openai_api import call_openai_api
            async with message.channel.typing():
                response = await call_openai_api(system_message, new_message, self.message_histories[user_key])
                if response is not None:
                    print(response)
                else:
                    print("OpenAI's API call failed.")
            # APIを呼び出した後の時間を記録し、開始時間を引くことで経過時間を計算
            elapsed_time = time.time() - start_time
            print(f"The API call took {elapsed_time} seconds.")
            bot_response = response['choices'][0]['message']['content']
            print("ボットの応答: ", bot_response)
            # メッセージ履歴に含まれる全てのメッセージのトークン数を計算
            total_tokens = 0
            # ユーザーのメッセージ履歴にある各メッセージをループ処理
            for message_history in self.message_histories[user_key]:
                # メッセージをJSON文字列に変換
                message_as_json = json.dumps(message_history)
                # JSON文字列のトークン数をカウント
                num_tokens_in_message = count_tokens(message_as_json)
                # 現在のメッセージに含まれるトークンの総数を合計に加算
                total_tokens += num_tokens_in_message
            # トークンの総数をインスタンス変数に格納
            self.total_tokens = total_tokens
            # 新しいメッセージとシステムメッセージのトークン数を追加
            self.total_tokens += count_tokens(json.dumps(new_message)) + count_tokens(json.dumps(system_message)) + \
                count_tokens(json.dumps(bot_response))
            # 新しいメッセージを追加するとトークン制限を超える場合、古いメッセージを削除する。
            while self.total_tokens > MAX_TOKENS:
                # 最初のメッセージを削除する
                removed_message = self.message_histories[user_key].pop(0)
                # 削除したメッセージのトークン数を引く
                self.total_tokens -= count_tokens(json.dumps(removed_message))
            # トークン数が制限以下になったら新しいメッセージを追加
            self.message_histories[user_key].append(new_message)
            self.message_histories[user_key].append({"role": "assistant", "content": bot_response})
            # メッセージ履歴に含まれる全てのメッセージのトークン数を計算
            self.total_tokens = sum(count_tokens(json.dumps(m)) for m in self.message_histories[user_key])
            # 新しいメッセージとシステムメッセージのトークン数を追加
            self.total_tokens += count_tokens(json.dumps(new_message)) + count_tokens(json.dumps(system_message))
            if not debug_mode:
                # メッセージ履歴をRedisに保存し、TTLを設定
                message_history_json = json.dumps(self.message_histories[user_key])
                # Redisサーバーへメッセージの履歴を保存するのにかかった時間を計測
                start_time = time.time()
                r.set(f'message_history_{user_key}', message_history_json)
                r.expire(f'message_history_{user_key}', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
                end_time = time.time()
                # 経過時間を計算して表示
                elapsed_time = end_time - start_time
                print(f"Redisへ会話履歴を保存するのにかかった時間: {elapsed_time} 秒。")  # 追加: 経過時間を表示
                print("メッセージはRedisサーバーへ送信されました。")
            print("await sending message to discord with async typing function!")
            # ボットからの応答の文字数に応じて、タイピング中のアニメーションの表示時間を調整する
            typing_time = min(max(len(bot_response) / 50, 3), 9)  # タイピングスピードを変えるために、分割数を調整する
            print("typing_time: ", typing_time)
            print("await sending message to discord with async typing function!")
            async with message.channel.typing():
                await sleep(typing_time)  # 計算された時間まで待つ
                await message.reply(bot_response)
                print("massage have sent to discord!")
            print("message_history: ", self.message_histories)

    @bot.event
    async def on_voice_state_update(self, member, before, after):
        if self.topic == Topic.DEAD_BY_DAY_LIGHT:
            # Dead by Daylightの「ゲーム」のボイスチャットチャンネルのIDを指定
            voice_channel_id = 1003966899232702537
            # ボイスチャットの状態を取得
            voice_channel = self.get_channel(voice_channel_id)
            voice_states = voice_channel.voice_states
            # Dead by Daylightをプレイ中かつ特定のボイスチャットで話しているメンバーの数をカウント
            dbd_players = sum(1 for voice_member in voice_states.values() if
                              voice_member.activity and voice_member.activity.name == "Dead by Daylight" and
                              voice_member.channel == voice_channel)
            print(f"ボイスチャットに参加しているメンバーの数: {dbd_players}")
            # ボイスチャットに参加しているメンバーが2人以上いて、その2人がDead by Daylightをプレイしている場合
            if dbd_players >= 2:
                # メッセージを送信するチャンネルのIDを指定
                message_channel_id = 1003966898792312854
                # メッセージを送信するチャンネルを取得
                message_channel = self.get_channel(message_channel_id)
                # メンバーの名前を取得してメッセージを作成
                members = [self.get_user(member).name for member, voice_member in voice_states.items()
                           if voice_member.activity and voice_member.activity.name == "Dead by Daylight" and
                           voice_member.channel == voice_channel]
                message = f"{', '.join(members)}さん、Dead by Daylightを楽しんでください！"
                # メッセージを送信
                await message_channel.send(message)


def main():
    intents = discord.Intents.all()
    intents.voice_states = True
    client = MyBot(topic_enum, command_prefix='!', intents=intents)
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

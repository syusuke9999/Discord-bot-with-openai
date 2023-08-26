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
import openai_api
from system_message import Topic, SystemMessage
from RetrievalQA import RetrievalQAFromFaiss
from ConversationWithKnowledge import RetrievalConversationWithFaiss
import langchain
import wandb

assert langchain.__version__ >= "0.0.218", "Please ensure you are using LangChain v0.0.188 or higher"

wbkey = os.getenv("wbkey")
wandb.login(key=wbkey)

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"  # ここで環境変数を設定
wandb.init(project="discord-bot-llm-trace")

debug_mode = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
TOPIC_ENUM = os.getenv("TOPIC")
THIS_TOPIC_ENUM = Topic.__members__.get(TOPIC_ENUM)

if not debug_mode:
    # Redis接続を初期化
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD)
    print("Redis connection established!")

model_name = "gpt-3.5-turbo-16k-0613"

encoding: Encoding = tiktoken.encoding_for_model(model_name)

MAX_TOKENS = 6000

logger = logging.getLogger('discord')
logger.setLevel(logging.WARNING)


def count_tokens(text):
    tokens = encoding.encode(text)
    tokens_count = len(tokens)
    return tokens_count


async def send_message(message, bot_response_for_answer):
    if bot_response_for_answer is not None:
        length = len(bot_response_for_answer)
        typing_time = min(max(length / 50, 3), 9)  # タイピングスピードを変えるために、分割数を調整する
        async with message.channel.typing():
            await sleep(typing_time)  # 計算された時間まで待つ
            await message.reply(bot_response_for_answer)


def truncate_message_histories_and_tokens(token_limit, message_history):
    # メッセージ履歴に含まれる全てのメッセージのトークン数を計算
    total_tokens = sum(count_tokens(json.dumps(m)) for m in message_history)
    # 新しいメッセージを追加するとトークン制限を超える場合、古いメッセージを削除する。
    while total_tokens > token_limit:
        # 最初のメッセージを削除する
        removed_message = message_history.pop(0)
        # 削除したメッセージのトークン数を引く
        total_tokens -= count_tokens(json.dumps(removed_message))
    # 切り詰めたメッセージ履歴に含まれるトークン数を表示
    print("truncated_history_tokens: ", total_tokens)
    return message_history


# noinspection PyUnusedLocal
class MyBot(commands.Bot):
    global model_name

    def __init__(self, command_prefix, intents, enum_of_topic):
        super().__init__(command_prefix, intents=intents)
        self.topic_enum = enum_of_topic
        # 全ての会話履歴
        self.message_histories = {}
        # 個別のモデルに配慮して、トークン数を制限した会話履歴
        self.truncated_message_histories = {}
        self.total_tokens = 0  # トークン数の合計を保持するための変数を追加
        # 特定のチャンネルのIDをリストで設定します。
        self.special_channel_ids = [1117363032172003328, 1139850220914610236, 1143926306644439061]
        self.max_tokens = 3000
        self.model_name = model_name
        self.model_temperature = 0.2
        self.model_top_p = 0
        self.model_presence_penalty = 0
        self.model_frequency_penalty = 0

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
            bot_response: str = ""
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
            new_message_dict = {"role": "user", "content": message.content}
            # Dead by Daylightに関する具体的なトピックがどうかをGPT-3.5に判断させる。
            system_message_instance = SystemMessage(topic=Topic.IS_DEAD_BY_DAY_LIGHT_SPECIFIC_TOPIC)
            system_message_content = system_message_instance.get_system_message_content()
            system_message_dict = {"role": "system", "content": system_message_content}
            print("「検索」か「その他」かの判定を行うシステムメッセージ: ", system_message_content)
            # 判定にはgpt-4-0613を使用する
            self.model_name = "gpt-4-0613"
            self.max_tokens = 1
            self.model_temperature = 0
            self.model_top_p = 0
            self.model_presence_penalty = 0
            self.model_frequency_penalty = 0
            hyper_parameters = {"model_name": self.model_name, "max_tokens": self.max_tokens, "temperature":
                                self.model_temperature, "top_p": self.model_top_p, "presence_penalty":
                                    self.model_presence_penalty, "frequency_penalty": self.model_frequency_penalty}
            # タイピングアニメーションと共に話題が「search」か「conversation」かを判定させる
            response = await openai_api.call_openai_api(hyper_parameters, system_message_dict, new_message_dict)
            # OpenAI APIからのレスポンスが期待に添った形かどうかを確認して内容を抽出する
            try:
                content = response["choices"][0]["message"]["content"]
            except (TypeError, KeyError, IndexError):
                content = None
            if content is not None:
                bot_classification = content
            else:
                print("initial bot_response is None or empty.")
                return
            print("\033[93mAIによるユーザーの発言への反応の判定:\033[0m \033[91m", bot_classification, "\033[0m")
            # 「conversation」に分類されなかった場合は「search」と推定してRetrival QAを実行する（検索優先の原則）
            if "search" in bot_classification:
                print("Retrival QAを実行します")
                async with (message.channel.typing()):
                    start_time = time.time()
                    # メンションしたユーザーのIDと名前を取得
                    user_id = str(message.author.id)
                    user_name = message.author.display_name
                    # ユーザーのIDと名前を組み合わせて、ユーザーを一意に識別するユーザーキーを作成
                    user_key = f'{user_id}_{user_name}'
                    retrival_qa = RetrievalQAFromFaiss()
                    # クローリングしたデータからユーザーの質問に関係のありそうなものを探し、GPT-4が質問に対する答えだと判断した場合はここで答えが返ってくる
                    bot_response, input_query = await retrival_qa.GetAnswerFromFaiss(message.content)
                    elapsed_time = time.time() - start_time
                    print(f"The retrieval qa process took {elapsed_time} seconds.")
                    system_message_instance = SystemMessage(topic=Topic.DETERMINE_ANSWERED_OR_NOT_ANSWERED)
                    system_message_content = system_message_instance.get_system_message_content()
                    system_message_dict = {"role": "system", "content": system_message_content}
                    new_message_dict = {"role": "user", "content": bot_response}
                    print("new_message_dict: ", new_message_dict)
                    print("「回答」か「回答できない」かの判定を行うシステムメッセージ: ", system_message_content)
                    # 判定にはgpt-4-0613を使用する
                    self.model_name = "gpt-4-0613"
                    self.max_tokens = 10
                    self.model_temperature = 0
                    self.model_top_p = 0
                    self.model_presence_penalty = 0
                    self.model_frequency_penalty = 0
                    hyper_parameters = {"model_name": self.model_name, "max_tokens": self.max_tokens, "temperature":
                                        self.model_temperature, "top_p": self.model_top_p, "presence_penalty":
                                            self.model_presence_penalty, "frequency_penalty":
                                            self.model_frequency_penalty}
                    response = await openai_api.call_openai_api(hyper_parameters, system_message_dict,
                                                                new_message_dict)
                    try:
                        content = response["choices"][0]["message"]["content"]
                    except (TypeError, KeyError, IndexError):
                        content = None
                    if content is not None:
                        bot_classification = content
                    else:
                        print("initial bot_response is None or empty.")
                        return
                    print("ユーザーの発言: ", message.content)
                    print("Retrival QAによる回答: ", bot_response)
                    print("\033[93mAIが質問に答えられたかの判定「don't know」,「other」:\033[0m \033[91m",
                          bot_classification, "\033[0m")
                    if "don't Know" in bot_classification:
                        print("\033[93m検索結果から回答を見つけられなかったため、URLは添付しません。\033[0m")
                        if bot_response is not None:
                            await send_message(message, bot_response)
                    elif "answered" in bot_classification:
                        system_message_instance = SystemMessage(topic=Topic.PARAPHRASE_THE_RESPONSE_TEXT)
                        system_message_content = system_message_instance.get_system_message_content()
                        system_message_dict = {"role": "system", "content": system_message_content}
                        new_message_dict = {"role": "user", "content": bot_response}
                        print("new_message_dict: ", new_message_dict)
                        paraphrased_response = await openai_api.call_openai_api(hyper_parameters, system_message_dict)
                        if paraphrased_response is not None:
                            await send_message(message, paraphrased_response)
            # 「会話」に分類されたか分類不能の場合は、gpt-3.5-turbo-16kを使用して会話を続ける
            elif "conversation" in bot_classification:
                self.max_tokens = 10000
                self.model_name = "gpt-3.5-turbo-16k-0613"
                self.model_frequency_penalty = 0.6
                self.model_presence_penalty = 0
                self.model_temperature = 0.4
                self.model_top_p = 1
                system_message_instance = SystemMessage(topic=Topic.DEAD_BY_DAY_LIGHT)
                system_message_content = system_message_instance.get_system_message_content()
                system_message_dict = {"role": "system", "content": system_message_content}
                # メッセージの履歴を10000トークン以下にして送信する
                message_history = truncate_message_histories_and_tokens(10000, self.message_histories[user_key])
                print("\033[93m「会話」に分類されため、gpt-3.5-turbo-16k-0613を使用して会話を続けます\033[0m")
                retrieval_conversation = RetrievalConversationWithFaiss(self)
                bot_response, input_query = await retrieval_conversation.GetResponseWithFaiss(message.content, user_key)
                print("システムメッージ: ", system_message_content)
                print("Send query user conversation to OpenAI API with async typing function: ",
                      message.content)
                async with message.channel.typing():
                    start_time = time.time()
                    if bot_response is not None:
                        print("assistant response for user's conversation: ", bot_response)
                        await send_message(message, bot_response)
                    else:
                        print("bot_response is None or empty.")
                        return
                    elapsed_time = time.time() - start_time
                    print(f"The OpenAI API conversation process took {elapsed_time} seconds.")
            # ユーザーの発言とアシスタントの発言を辞書形式に変換して、メッセージの履歴に追加
            self.message_histories[user_key].append({"role": "user", "content": message.content})
            self.message_histories[user_key].append({"role": "assistant",
                                                     "content": bot_response})
            # 辞書形式のメッセージの履歴をJSON形式に変換
            message_history_json = json.dumps(self.message_histories[user_key])
            # Redisサーバーへメッセージの履歴を保存するのにかかった時間を計測
            start_time = time.time()
            # Redisサーバーへメッセージの履歴を保存し、TTLを設定
            r.set(f'message_history_{user_key}', message_history_json)
            r.expire(f'message_history_{user_key}', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
            end_time = time.time()
            # 経過時間を計算して表示
            elapsed_time = end_time - start_time
            print(f"Elapsed time to save data to Redis server: {elapsed_time} seconds")  # 経過時間を表示
            # APIへの送信に使用するメッセージの履歴を更新

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        if THIS_TOPIC_ENUM == Topic.DEAD_BY_DAY_LIGHT:
            # ボイスチャンネルIDを指定します
            your_voice_chat_channel_id = 1003966899232702537
            if after.channel is not None and after.channel.id == your_voice_chat_channel_id:
                # チャンネルのメンバーが増えて2人以上いるか確認します
                if (before.channel is None and len(after.channel.members) >= 2) or \
                        (before.channel is not None and len(before.channel.members) < len(
                            after.channel.members) and len(
                            after.channel.members) >= 2):
                    # メンバーの名前を取得してカンマ区切りの文字列にします
                    member_names = ""
                    for participant in after.channel.members:
                        if member_names != "":
                            member_names += ", "
                        member_names += participant.name
                    # テキストチャンネルIDを指定します
                    your_text_chanel_id = 1117412783592591460
                    # ボイスチャットに参加しているメンバーが2人以上いる場合、ユーザー名を指定してメッセージを送信する
                    send_text = f'{member_names}さん、Dead by Daylightを楽しんで下さい。'
                    if member_names not in self.message_histories:
                        self.message_histories[member_names] = []
                    self.message_histories[member_names].append({"role": "assistant", "content": send_text})
                    # 辞書形式のメッセージの履歴をJSON形式に変換
                    message_history_json = json.dumps(self.message_histories[member_names])
                    # Redisサーバーへメッセージの履歴を保存するのにかかった時間を計測
                    start_time = time.time()
                    # Redisサーバーへメッセージの履歴を保存し、TTLを設定
                    r.set(f'message_history_{member_names}', message_history_json)
                    r.expire(f'message_history_{member_names}', 3600 * 24 * 10)  # TTLを20日間（1,728,000秒）に設定
                    end_time = time.time()
                    # 経過時間を計算して表示
                    elapsed_time = end_time - start_time
                    print(f"Elapsed time to save data to Redis server: {elapsed_time} seconds")  # 経過時間を表示
                    await self.get_channel(your_text_chanel_id).send(
                        f'{member_names}さん、Dead by Daylightを楽しんで下さい。')


def main():
    # Discord接続を初期化
    intents = discord.Intents.all()
    # ボイスステータスのインテントを取得する意図を明視するため
    intents.voice_states = True
    bot = MyBot(command_prefix='!', intents=intents, enum_of_topic=THIS_TOPIC_ENUM)
    # Discordボットのトークンを指定
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

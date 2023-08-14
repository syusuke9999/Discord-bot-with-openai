from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
from wandb.integration.openai import autolog
import os
import asyncio
from datetime import datetime
import pytz


class RetrievalConversationWithFaiss:
    def __init__(self, bot_instance):
        self.total_tokens = 0
        self.input_txt = ""
        # MyBotのmessage_historiesを参照
        self.message_histories = bot_instance.message_histories

    async def GetResponseWithFaiss(self, query, user_key):
        autolog({
            "project": "discord-bot-llm-trace",
            "group": "GetResponseWithFaiss"
        })
        self.input_txt = query
        llm = load_llm("my_conversation_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, top_k=5)
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())

            # 現在の日付と時刻を取得します（日本時間）。
            now = datetime.now(pytz.timezone('Asia/Tokyo'))
            # 年、月、日を取得します。
            year = now.year
            month = now.month
            day = now.day
            # 直近のメッセージを取得
            recent_messages = self.message_histories[user_key][-10:]  # 5往復分なので、最後の10メッセージ
            # 対話形式に変換
            dialogue_format = ""
            for msg in recent_messages:
                role = "User" if msg['role'] == 'user' else "Assistant"
                dialogue_format += f"{role}: {msg['content']}\n"
            print("dialogue_format: ", dialogue_format)
            custom_prompt = (f"Previous Conversation: {dialogue_format}\n"
                             f"Today is the year {year}, the month is {month} and the date {day}."
                             f"The current time is {now}."
                             "Please use the following context, if it is relevant to the user's question, "
                             "to talk about Dead by Daylight in an enjoyable way, "
                             " with it's relevant context to the user's query."
                             "Please use Japanese only. Don't use English."
                             "日本人として日本語を使って会話をして下さい。"
                             " \n"
                             "Context:{context}"
                             " \n"
                             "subject: {question}"
                             "Fun Conversational Response:")
            stuff_prompt = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": stuff_prompt}
            stuff_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                verbose=True,
                chain_type_kwargs=chain_type_kwargs  # ここで変数stuff_promptを直接渡す
            )
            print("custom_prompt: ", custom_prompt)
            # return_source_documentsプロパティをTrueにセット
            stuff_qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: stuff_qa.apply([self.input_txt]))
            print(f"Input data: {query}")
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"stuff_answer: {answer}")
                autolog.disable(self)
                return answer, self.input_txt
            autolog.disable(self)
            return answer, self.input_txt

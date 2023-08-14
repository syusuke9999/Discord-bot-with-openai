from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import os
import asyncio
from datetime import datetime
import pytz


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, input_txt):
        self.input_txt = input_txt
        llm = load_llm("my_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings)
        source_url = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            search_kwargs = {"k": 6, "return_metadata": True}
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever(
                                                                       search_type="mmr",
                                                                       search_kwargs=search_kwargs))

            # 現在の日付と時刻を取得します（日本時間）。
            now = datetime.now(pytz.timezone('Asia/Tokyo'))
            # 年、月、日を取得します。
            year = now.year
            month = now.month
            day = now.day
            custom_prompt = (f"Today is the year {year}, the month is {month} and the date {day}."
                             f"The current time is {now}."
                             "Use the following pieces of context to answer the question at the end. If you don't "
                             "know the answer,"
                             " just say 「分かりません」, don't try to make up an answer. Answer the question"
                             " as if you were a native Japanese speaker."
                             " \n"
                             "Context:{context}\n"
                             " \n"
                             "Question: {question}\n"
                             "Helpful Answer:")
            stuff_prompt = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": stuff_prompt}
            stuff_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                chain_type_kwargs=chain_type_kwargs  # ここで変数stuff_promptを直接渡す
            )
            # return_source_documentsプロパティをTrueにセット
            stuff_qa.return_source_documents = True
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: stuff_qa.apply([input_txt]))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
            try:
                source_url = response[0]["source_documents"][0].metadata["source"]
            except (TypeError, KeyError, IndexError):
                source_url = None
            return answer, source_url, self.input_txt
        else:
            answer = "申し訳ありません。データベースに不具合が生じているようです。開発者にお問い合わせください。"
            return answer, source_url, self.input_txt

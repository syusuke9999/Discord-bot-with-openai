from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
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
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, top_k=3)
        source_url = ""
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
            custom_prompt = (f"Today is the year {year}, the month is {month} and the date {day}."
                             f"The current time is {now}."
                             "Use the following pieces of context to answer the question at the end. If you don't "
                             "know the answer,"
                             " just say 「分かりません」, don't try to make up an answer. "
                             "Please use Japanese only. Don't use English."
                             " \n"
                             "Context:{context}"
                             " \n"
                             "Question: {question}"
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
                verbose=True,
                chain_type_kwargs=chain_type_kwargs  # ここで変数stuff_promptを直接渡す
            )
            # return_source_documentsプロパティをTrueにセット
            stuff_qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            print(f"Input dict before apply: {input_txt}")
            response = await loop.run_in_executor(None, lambda: stuff_qa.apply([input_txt]))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                stuff_answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                stuff_answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"stuff_answer: {stuff_answer}")
                return stuff_answer, source_url, input_txt
            try:
                source_url = response[0]["source_documents"][0].metadata["source"]
                print(f"source_url: {source_url}")
            except (TypeError, KeyError, IndexError):
                source_url = None
            refine_qa = stuff_qa.from_chain_type(
                chain_type="refine",
                llm=llm,
                retriever=compression_retriever,
                verbose=True,
            )
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            print(f"Input stuff_answer: {stuff_answer}")
            response = await loop.run_in_executor(None, lambda: refine_qa.apply([stuff_answer]))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                refined_answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                refined_answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"refined_answer: {refined_answer}")
                return refined_answer, source_url, input_txt

            return refined_answer, source_url, input_txt
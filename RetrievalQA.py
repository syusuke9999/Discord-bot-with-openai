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
            search_kwargs = {"k": 4, "return_metadata": True}
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever(
                                                                    search_kwargs=search_kwargs))
            refine_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="refine",
                retriever=compression_retriever,
                # chain_type_kwargs=chain_type_kwargs  # ここで変数stuff_promptを直接渡す
            )
            # return_source_documentsプロパティをTrueにセット
            refine_qa.return_source_documents = True
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: refine_qa.apply([input_txt]))
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

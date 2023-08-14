from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from wandb.integration.langchain import WandbTracer
import os
import asyncio


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0

    async def GetAnswerFromFaiss(self, query):
        WandbTracer.init({"group": "GetAnswerFromFaiss"})
        llm = load_llm("my_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, top_k=6)
        source_url = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())
            # 現在の日付と時刻を取得します（日本時間）。
            stuff_qa = RetrievalQA.from_chain_type(
                chain_type="stuff",
                llm=llm,
                retriever=compression_retriever,
                verbose=True,
            )
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: stuff_qa.apply([query]))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                stuff_answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                stuff_answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"stuff_answer: {stuff_answer}")
                return stuff_answer, source_url, self
            try:
                source_url = response[0]["source_url"]
            except (TypeError, KeyError, IndexError):
                source_url = ""
                return stuff_answer, source_url, self
            return stuff_answer, source_url, self

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import asyncio


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0

    async def GetAnswerFromFaiss(self, query):
        llm = load_llm("my_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings)
        source_url = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())
            # 現在の日付と時刻を取得します（日本時間）。
            refine_qa = RetrievalQA.from_chain_type(
                chain_type="refine",
                llm=llm,
                retriever=compression_retriever,
                verbose=True,
            )
            refine_qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: refine_qa.apply([query]))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                refine_answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                refine_answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"stuff_answer: {refine_answer}")
                return refine_answer, source_url, self
            try:
                source_url = response[0]["source_url"]
            except (TypeError, KeyError, IndexError):
                source_url = ""
                print("source_url = """)
                return refine_answer, source_url, self
            return refine_answer, source_url, self

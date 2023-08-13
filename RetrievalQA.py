from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import asyncio


class RetrievalQAFromFaiss:
    def __init__(self, bot_instance):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""
        # MyBotのmessage_historiesを参照
        self.message_histories = bot_instance.message_histories

    async def GetAnswerFromFaiss(self, input_txt, user_key):
        self.input_txt = input_txt
        llm = load_llm("my_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, top_k=4)
        source_url = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())
            refine_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="refine",
                retriever=compression_retriever,
                verbose=True,
            )
            # return_source_documentsプロパティをTrueにセット
            refine_qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            print(f"Input dict before apply: {input_txt}")
            response = await loop.run_in_executor(None, lambda: refine_qa.apply([input_txt]))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                refine_answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                refine_answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"stuff_answer: {refine_answer}")
                return refine_answer, source_url, input_txt
            try:
                source_url = response[0]["source_documents"][0].metadata["source"]
                print(f"source_url: {source_url}")
            except (TypeError, KeyError, IndexError):
                source_url = None
                print(f"source_url: {source_url}")
            return refine_answer, source_url, input_txt

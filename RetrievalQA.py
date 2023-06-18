from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import asyncio
from difflib import SequenceMatcher


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, input_txt):
        llm = load_llm("my_llm.json")
        keywords = ["パーク", "パークについて", "パークは？", "パーク？"]
        excluded_keywords = ["おすすめ", "お勧め", "組み合わせ", "組み合わせについて", "組み合わせを教えて"]
        if any(keyword in input_txt for keyword in keywords) and not any(excluded_keyword in input_txt for
                                                                         excluded_keyword in excluded_keywords):
            input_txt = input_txt.replace("パーク", "パークの性能と効果解説")
            print(input_txt)
        self.input_txt = input_txt
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        source_url: str = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: qa.run(query=self.input_txt))
            relevant_documents = compression_retriever.get_relevant_documents(input_txt)
            # 各ドキュメントとresponseのテキストの類似度を計算
            similarity_scores = [
                SequenceMatcher(None, doc.page_content, response).ratio()
                for doc in relevant_documents
            ]
            # 最も類似度が高いドキュメントのインデックスを取得
            max_index = similarity_scores.index(max(similarity_scores))
            # 最も類似度が高いドキュメントのURLを取得
            try:
                source_url = relevant_documents[max_index].metadata['source']
            except (TypeError, KeyError, IndexError):
                source_url = None
            else:
                return response, source_url, self.input_txt
        else:
            response = "申し訳ありません。データベースに不具合が生じているようです。開発者が修正するまでお待ちください。"
            return response, source_url, self.input_txt

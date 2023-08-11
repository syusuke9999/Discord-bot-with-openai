from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
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
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, top_k=4)
        source_url = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())

            custom_prompt = """
            f"Today is the year {now_of_year}, the month is {now_of_month} and the date {now_of_day}." \
            f"The current time is {now_of_time}. " \
            f"Use the following pieces of context to answer the question at the end. If you don't know the answer," \
            f"just say 「分かりません」, don't try to make up an answer. Answer the question" \
            f"as if you were a native Japanese speaker." \
            f"\n" \
            f"Context:{context}" \
            f"\n" \
            f"Question: {question}
            f"Helpful Answer:"""
            stuff_prompt = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "question"]
            )
            stuff_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                prompt=PromptTemplate(template=stuff_prompt)
            )
            stuff_answer = stuff_qa(input_txt)
            refine_qa = stuff_qa.from_chain_type(
                chain_type="refine",
                llm=llm,
                retriever=compression_retriever,
            )
            # return_source_documentsプロパティをTrueにセット
            stuff_qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            print(f"Input dict before apply: {input_txt}")
            response = await loop.run_in_executor(None, lambda: refine_qa.apply([stuff_answer]))
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

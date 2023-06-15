from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""
        self.llm = load_llm("my_llm.json")

    async def GetAnswerFromFaiss(self, input_txt):
        # ファイルからllmを読み込む
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())
            qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=compression_retriever)
            response = await qa.arun(query=input_txt)
            relevant_document = compression_retriever.get_relevant_documents(input_txt)
            print(relevant_document)
        else:
            response = "申し訳ありません。データベースに不具合が生じているようです。開発者が修正するまでお待ちください。"
        return response

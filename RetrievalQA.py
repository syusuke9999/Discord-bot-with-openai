from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langdetect import detect
import re
import os
import asyncio


def remove_english(text):
    # 英語のセンテンスを削除
    sentences = re.split(r'(\s*[.?!]\s*)', text)
    non_english_sentences = []

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        # センテンスが空でない場合のみ言語検出を行う
        if sentence.strip():
            try:
                if detect(sentence) != 'en':
                    non_english_sentences.append(sentence + sentences[i + 1])
            except:
                # 言語検出に失敗した場合、センテンスをそのまま追加
                non_english_sentences.append(sentence + sentences[i + 1])

    # 最後のセンテンスが残っている場合、追加
    if len(sentences) % 2 != 0:
        non_english_sentences.append(sentences[-1])

    return "".join(non_english_sentences)


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

            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever(),
                                                                   return_source_documents=True)
            refine_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="refine",
                retriever=compression_retriever
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
            print("answer: ", answer)
            # 英語の部分を削除
            text_japanese_only = remove_english(answer)  # 英語の部分を削除
            print(text_japanese_only)
            return text_japanese_only, source_url, self.input_txt
        else:
            text_japanese_only = "申し訳ありません。データベースに不具合が生じているようです。開発者にお問い合わせください。"
            return text_japanese_only, source_url, self.input_txt

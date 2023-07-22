from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import asyncio

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate
from datetime import datetime
import pytz

# 現在の日付と時刻を取得します（日本時間）。
now = datetime.now(pytz.timezone('Asia/Tokyo'))

# 年、月、日を取得します。
year = now.year
month = now.month
days = now.day

# プロンプトのテンプレートを定義します。
prompt_template = f"""
This year is {year}, and this month is {month},  and today is {days}. And now time is {now}.
You are a Discord bot residing in a channel on a Discord server where people gather to enjoy Dead by Daylight. 
Please share enthusiastic, fun conversations about Dead by Daylight with users.
Be sure to answer in Japanese. Do not use English.
You are asked a game-related question by users, please use the following pieces of context to answer the users question. 
If you don't know the answer, just say 「分かりません」, don't try to make up an answer.

{{context}}

Question: {{question}}
Helpful Answer:"""

# PromptTemplateを使用してプロンプトを作成します。
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, input_txt):
        self.input_txt = input_txt
        llm = load_llm("my_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76, top_k=10)
        source_url = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever, prompt=PROMPT)
            # return_source_documentsプロパティをTrueにセット
            qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            print(f"Input dict before apply: {input_txt}")
            response = await loop.run_in_executor(None, lambda: qa.apply([self.input_txt]))
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

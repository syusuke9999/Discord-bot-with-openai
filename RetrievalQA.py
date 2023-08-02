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
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())

            # 現在の日付と時刻を取得します（日本時間）。
            now = datetime.now(pytz.timezone('Asia/Tokyo'))
            # 年、月、日を取得します。
            year = now.year
            month = now.month
            days = now.day
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
            # カスタムのrefineプロンプトを定義します。
            refine_prompt_template = ("新しい文脈を踏まえた上で質問に対する答えがより良いものになるよう、"
                                      "元の答えを洗練させてください: {question}\n"
                                      "元の答え: {existing_answer}\n"
                                      "新しい文脈: {context_str}")
            refine_prompt = PromptTemplate(template=refine_prompt_template,
                                           input_variables=["question", "existing_answer", "context_str"])
            # 初期のLLMチェーンとrefineチェーンを作成します。
            initial_chain = LLMChain(llm=llm, prompt=prompt_template)  # 既存のプロンプト
            refine_chain = LLMChain(llm=llm, prompt=refine_prompt)
            # RefineDocumentsChainを作成します。
            refine_documents_chain = RefineDocumentsChain(initial_llm_chain=initial_chain,
                                                          refine_llm_chain=refine_chain,
                                                          document_variable_name="context_str",
                                                          initial_response_name="existing_answer")
            # Change chain_type from "stuff" to "refine"
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="refine",
                retriever=compression_retriever,
                chain=refine_documents_chain,
            )
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

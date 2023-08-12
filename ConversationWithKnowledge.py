from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
import os
import asyncio
from datetime import datetime
import pytz


class RetrievalConversationWithFaiss:
    conversation_history = []  # クラスレベルで会話の履歴を保持するリストを作成
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetResponseWithFaiss(self, query):
        # 以前の会話履歴を取得
        previous_conversation = ' '.join(RetrievalConversationWithFaiss.conversation_history[-5:]) # 最新の5つのメッセージを取得
        self.input_txt = query
        llm = load_llm("my_conversation_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, top_k=8)
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())

            # 現在の日付と時刻を取得します（日本時間）。
            now = datetime.now(pytz.timezone('Asia/Tokyo'))
            # 年、月、日を取得します。
            year = now.year
            month = now.month
            day = now.day
            custom_prompt = (f"Previous Conversation: {previous_conversation}\n"
                             f"Today is the year {year}, the month is {month} and the date {day}."
                             f"The current time is {now}."
                             "Please use the following context, if it is relevant to the user's question, "
                             "to talk about Dead by Daylight in an enjoyable way, "
                             " with it's relevant context to the user's query."
                             "Please use Japanese only. Don't use English."
                             " \n"
                             "Context:{context}"
                             " \n"
                             "subject: {question}"
                             "Helpful Answer:")
            stuff_prompt = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": stuff_prompt}
            stuff_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                verbose=True,
                chain_type_kwargs=chain_type_kwargs  # ここで変数stuff_promptを直接渡す
            )
            print("custom_prompt: ", custom_prompt)
            # return_source_documentsプロパティをTrueにセット
            stuff_qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: stuff_qa.apply([self.input_txt]))
            print(f"Input data: {query}")
            # クエリと応答を会話履歴に追加
            RetrievalConversationWithFaiss.conversation_history.append(query)
            RetrievalConversationWithFaiss.conversation_history.append(response[0]["result"])
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"stuff_answer: {answer}")
                return answer, self.input_txt
            return answer, self.input_txt

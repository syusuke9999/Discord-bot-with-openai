from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate
import os
import asyncio
from datetime import datetime
import pytz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def extract_top_entities(input_documents, given_query, custom_file_path='custom_entities.txt'):
    # カスタム辞書をテキストファイルから読み込む
    with open(custom_file_path, 'r') as f:
        custom_dictionary = [line.strip() for line in f.readlines()]
    # 文書をテキスト形式に変換（Documentオブジェクトから）
    documents = [doc.page_content for doc in input_documents]
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())
    # 頻度が高い語句を抽出
    sorted_by_tfidf = np.argsort(tfidf_matrix.sum(axis=0).A1)
    top_terms = feature_names[sorted_by_tfidf[-6:]]
    # クエリに固有表現を追加
    modified_ver_query = given_query
    for term in top_terms:
        modified_ver_query = modified_ver_query.replace(term, f"「{term}」")
    # カスタム辞書を用いて固有表現を追加
    for term in custom_dictionary:
        modified_ver_query = modified_ver_query.replace(term, f"「{term}」")
    # N-gram解析（ここではbigramを使用）
    bigrams = re.findall(r'\b\w+\s+\w+\b', given_query)
    for bigram in bigrams:
        if bigram in custom_dictionary:
            modified_ver_query = modified_ver_query.replace(bigram, f"「{bigram}」")
    return modified_ver_query, top_terms


class RetrievalConversationWithFaiss:
    def __init__(self, bot_instance):
        self.total_tokens = 0
        self.input_txt = ""
        # MyBotのmessage_historiesを参照
        self.message_histories = bot_instance.message_histories

    async def GetResponseWithFaiss(self, query, user_key):
        self.input_txt = query
        llm = load_llm("my_conversation_llm.json")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings)
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
            # 直近のメッセージを取得
            recent_messages = self.message_histories[user_key][-10:]  # 5往復分なので、最後の10メッセージ
            # 対話形式に変換
            dialogue_format = ""
            for msg in recent_messages:
                role = "User" if msg['role'] == 'user' else "Assistant"
                dialogue_format += f"{role}: {msg['content']}\n"
            print("dialogue_format: ", dialogue_format)
            custom_prompt = (f"Previous Conversation: {dialogue_format}\n"
                             f"Today is the year {year}, the month is {month} and the date {day}."
                             f"The current time is {now}."
                             "Please use the following context, if it is relevant to the user's question, "
                             "to talk about Dead by Daylight in an enjoyable way, "
                             " with it's relevant context to the user's query."
                             "Please talk with Japanese only."
                             "Please act as if you have preferences, etc., just like other users."
                             "Please enjoy the conversation with emojis."
                             " \n"
                             "Context:{context}"
                             " \n"
                             "subject: {question}\n"
                             "Fun Conversational Response:")
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
            similar_documents = docsearch.similarity_search(query=query)
            print("custom_prompt: ", custom_prompt)
            # return_source_documentsプロパティをTrueにセット
            stuff_qa.return_source_documents = True
            modified_ver_query, entities = extract_top_entities(similar_documents, query)
            print("modified_ver_query: ", modified_ver_query)
            print("entities: ", entities)
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: stuff_qa.apply([modified_ver_query]))
            print(f"modified_ver_query: {modified_ver_query}")
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response[0]["result"]
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                print(f"stuff_answer: {answer}")
                return answer, self.input_txt
            return answer, self.input_txt

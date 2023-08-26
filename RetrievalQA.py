from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
import asyncio


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, initial_query):
        self.input_txt = initial_query
        embeddings = OpenAIEmbeddings()
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            refine_prompt_template = (
                "The original question is as follows: {question}\n"
                "We have provided an existing answer: {existing_answer}\n"
                "------------\n"
                "{context_str}\n"
                "------------\n"
                "Please provide a concise and clear answer to the original question in Japanese, "
                "considering the information above. If the information above doesn't contribute to the answer, "
                "return the original answer in Japanese."
            )
            refine_prompt = PromptTemplate(
                input_variables=["question", "existing_answer", "context_str"],
                template=refine_prompt_template,
            )
            initial_qa_template = (
                "Context information is below. \n"
                "---------------------\n"
                "{context_str}"
                "\n---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the question: {question} in Japanese.\n"
            )
            initial_qa_prompt = PromptTemplate(
                input_variables=["context_str", "question"], template=initial_qa_template
            )
            qa_chain = load_qa_chain(
                ChatOpenAI(temperature=0, model_name="gpt-4-0613", top_p=0, max_tokens=500, presence_penalty=0.6),
                chain_type="refine",
                question_prompt=initial_qa_prompt,
                refine_prompt=refine_prompt
            )
            similar_documents = docsearch.similarity_search(query=initial_query)
            # カスタム辞書
            custom_dictionary = ["予想外の展開", "堕落の介入", "人々のために", "這いずり起こし"]

            # TF-IDFベクトル化
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # 1-gramから3-gramまで考慮
            # 'Documentオブジェクトからテキストを抽出（仮定）
            similar_documents_text = [doc.page_content for doc in similar_documents]
            X = vectorizer.fit_transform(similar_documents_text)
            # ローカル変数 'feature_names' の初期化
            feature_names = None
            try:
                feature_names = np.array(vectorizer.get_feature_names_out())
            except AttributeError:
                print("TfidfVectorizerが適切にフィットされていません。")
            modified_query = None
            if feature_names is not None:
                sorted_by_tfidf = np.argsort(X.sum(axis=0).A1)
                top_terms = feature_names[sorted_by_tfidf[-6:]]
                # クエリに固有表現を追加
                modified_query = initial_query
                # カスタム辞書を使用して固有表現を追加
                for term in custom_dictionary:
                    modified_query = modified_query.replace(term, f"「{term}」")
                # TF-IDFで抽出した語句を使用して固有表現を追加
                for term in top_terms:
                    modified_query = modified_query.replace(term, f"「{term}」")
                print("modified_query: ", modified_query)
            print(f"Modified Query: {modified_query}")
            for doc in similar_documents:
                print("page_content= ", doc.page_content)
                print("metadata= ", str(doc.metadata))
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: qa_chain({"input_documents":
                                                                         similar_documents,
                                                                         "question": modified_query},
                                                                         return_only_outputs=True)
                                                  )
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response['output_text']
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
            print("answer: ", answer)
            return answer, self.input_txt
        else:
            answer = "申し訳ありません。データベースに不具合が生じているようです。開発者にお問い合わせください。"
            return answer, self.input_txt

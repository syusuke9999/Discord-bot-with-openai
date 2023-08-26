from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
import asyncio
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


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
        modified_ver_query = modified_ver_query.replace(term, f"[{term}]")
    # カスタム辞書を用いて固有表現を追加
    for term in custom_dictionary:
        modified_ver_query = modified_ver_query.replace(term, f"[{term}]")
    # N-gram解析（ここではbigramを使用）
    bigrams = re.findall(r'\b\w+\s+\w+\b', given_query)
    for bigram in bigrams:
        if bigram in custom_dictionary:
            modified_ver_query = modified_ver_query.replace(bigram, f"[{bigram}]")
    return modified_ver_query, top_terms


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, initial_query):
        self.input_txt = initial_query
        embeddings = OpenAIEmbeddings()
        answer_with_source = ""
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
            modified_ver_query, entities = extract_top_entities(similar_documents, initial_query)
            print("modified_ver_query: ", modified_ver_query)
            print("entities: ", entities)
            # FAISSを使用して類似度の高いドキュメントを検索
            similar_documents = docsearch.similarity_search(query=initial_query)
            # FAISSのスコアなどを基に関連性のあるドキュメントをフィルタリング
            relevant_documents = [doc for doc in similar_documents if doc.some_score > 0.76]  # 仮の条件
            # 関連性のあるドキュメントからsource URLを抽出
            source_urls = [doc.metadata.get('source', 'Unknown source') for doc in relevant_documents]

            # for doc in similar_documents:
            #    print("page_content= ", doc.page_content)
            #    print("metadata= ", str(doc.metadata))
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: qa_chain({"input_documents":
                                                  similar_documents, "question": modified_ver_query},
                                                                         return_only_outputs=True))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response['output_text']
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
            print("answer: ", answer)
            # 関連性のあるsource URLを答えに添える
            answer_with_source = f"{answer}\n参照元: {', '.join(source_urls)}"

            return answer, self.input_txt, answer_with_source
        else:
            answer = "申し訳ありません。データベースに不具合が生じているようです。開発者にお問い合わせください。"
            return answer, self.input_txt, answer_with_source

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import asyncio
from collections import Counter
import re
import os

from collections import Counter
import re


def extract_top_entities(input_documents, given_query, custom_file_path='custom_entities.txt'):
    documents = [doc.page_content.lower() for doc in input_documents]

    # 空のリストを用意
    bigrams = []

    # 各ドキュメントに対してbigramを抽出
    for doc in documents:
        bigrams.extend(re.findall(r'\b\w+\s+\w+\b', doc))

    # かぎ括弧で囲まれている固有表現を抽出
    bracketed_entities = []
    for doc in documents:
        bracketed_entities.extend(re.findall(r'「(.*?)」', doc))

    # 頻度が多い固有表現をカスタム辞書に保存
    entity_freq = Counter(bracketed_entities)
    new_entities = [entity for entity, freq in entity_freq.items() if freq > 0]

    # カスタム辞書と結合（custom_entitiesが未定義なので、この部分も修正が必要かもしれません）
    top_terms = list(set(bracketed_entities) | set(new_entities) | set(bigrams))

    # クエリに固有表現を追加
    modified_query = given_query
    entities = []
    for term in top_terms:
        if term in modified_query:
            modified_query = modified_query.replace(term, f"「{term}」")
            entities.append(term)

    return modified_query, entities


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
                "The original question is as follows:\n {question}\n"
                "We have provided an existing answer:\n {existing_answer}\n"
                "Please refine the above answer using the context information below "
                "(Only if needed and relevant to the question).\n"
                "------------\n"
                "{context_str}\n"
                "------------\n"
                "If the provided context contributes to a more concise and direct answer, "
                "and is relevant to the original question, please use it to improve your responses. "
                "Please make sure to avoid using unrelated words or phrases in your response."
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
                "answer the question:\n {question} in Japanese. If you couldn't find answer simply replay "
                "「調べたデータからは分かりませんでした。」\n"
            )
            initial_qa_prompt = PromptTemplate(
                input_variables=["context_str", "question"], template=initial_qa_template
            )
            qa_chain = load_qa_chain(
                ChatOpenAI(temperature=0, model_name="gpt-4-0613", max_tokens=500),
                chain_type="refine",
                question_prompt=initial_qa_prompt,
                refine_prompt=refine_prompt
            )
            similar_documents = docsearch.max_marginal_relevance_search(query=initial_query)
            modified_ver_query, entities = extract_top_entities(similar_documents, initial_query)
            print("modified_ver_query: ", modified_ver_query)
            print("entities: ", entities)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: qa_chain({"input_documents":
                                                                         similar_documents,
                                                                         "question": modified_ver_query},
                                                                         return_only_outputs=True))
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

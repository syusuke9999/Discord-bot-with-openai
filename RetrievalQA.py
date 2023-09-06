from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import asyncio
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import boto3
import os
from botocore.exceptions import NoCredentialsError


def get_s3_client():
    try:
        # AWS SSOを使用している場合、プロファイル名を指定する
        session = boto3.Session(profile_name='your-profile-name')
        s3 = session.client('s3')
        return s3
    except NoCredentialsError:
        print("認証情報が見つかりません。AWS SSOでログインしてください。")
        return None


def read_custom_entities_from_s3(bucket_name, file_name):
    s3 = get_s3_client()
    if s3 is None:
        return []
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_name)
        custom_entities = obj['Body'].read().decode('utf-8').splitlines()
        return custom_entities
    except Exception as e:
        print(f"S3からの読み取りに失敗しました: {e}")
        return []


def write_custom_entities_to_s3(bucket_name, file_name, entities):
    s3 = get_s3_client()
    if s3 is None:
        return
    try:
        s3.put_object(Body='\n'.join(entities), Bucket=bucket_name, Key=file_name)
    except Exception as e:
        print(f"S3への書き込みに失敗しました: {e}")


def extract_top_entities(input_documents, given_query, bucket='your-bucket', key='custom_entities.txt'):
    # カスタム辞書をS3から読み込む
    custom_entities = read_custom_entities_from_s3(bucket, key)
    # 文書を小文字に変換
    documents = [doc.page_content.lower() for doc in input_documents]
    # 文章をTF-IDFベクトル化し、各語句のTF-IDFスコアを計算
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = np.array(vectorizer.get_feature_names_out())
    sorted_by_tfidf = np.argsort(tfidf_matrix.sum(axis=0).A1)
    # IF-IDFスコアが高い語句を抽出して、top_termsリストに追加（ここでは上位6語）
    top_terms = feature_names[sorted_by_tfidf[-6:]]
    # N-gram解析（ここではbigramを使用）を行い、結果を一旦、bigramsリストに追加
    bigrams = re.findall(r'\b\w+\s+\w+\b', given_query)
    # かぎ括弧で囲まれている固有表現を抽出
    bracketed_entities = []
    for doc in input_documents:
        bracketed_entities.extend(re.findall(r'「(.*?)」', doc.page_content))
    # 頻度が多い固有表現をカスタム辞書に保存
    entity_freq = Counter(bracketed_entities)
    new_entities = []
    for entity, freq in entity_freq.items():
        if freq > 0:
            new_entities.append(entity)
    # S3にカスタム辞書を保存
    write_custom_entities_to_s3(new_entities, bucket, key)
    # カスタム辞書と結合
    top_terms = list(set(top_terms) | set(custom_entities) | set(bigrams))
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
                "Please refine the above answer using the context information below (if needed).\n"
                "------------\n"
                "{context_str}\n"
                "------------\n"
                "If the context that provided above contribute to provide a concise and direct answer to the original "
                "question in Japanese, please use the context above to improve the quality of your responses."
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
                "answer the question:\n {question} in Japanese.\n"
            )
            initial_qa_prompt = PromptTemplate(
                input_variables=["context_str", "question"], template=initial_qa_template
            )
            qa_chain = load_qa_chain(
                ChatOpenAI(temperature=0, model_name="gpt-4-0613", top_p=0, max_tokens=500),
                chain_type="refine",
                question_prompt=initial_qa_prompt,
                refine_prompt=refine_prompt
            )
            similar_documents = docsearch.similarity_search(query=initial_query)
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

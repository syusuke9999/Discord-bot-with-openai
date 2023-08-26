from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
import asyncio

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


def extract_top_entities(documents, query, top_n=6):
    custom_dict = ["真新しいパーツ", "Ultra Rare"]  # カスタム辞書に追加する固有名詞やフレーズ
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # feature_namesの初期化
    feature_names = None
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        print("TfidfVectorizerが適切にフィットされていません。")

    # TF-IDFでのトップ用語
    top_terms_tfidf = []
    if feature_names is not None:
        sorted_by_tfidf = X.sum(axis=0).argsort()[::-1]
        top_terms_tfidf = feature_names[sorted_by_tfidf[:top_n]]

    # 頻度分析（Counterを使用）
    word_freq = Counter(' '.join(documents).split())
    top_terms_freq = [item[0] for item in word_freq.most_common(top_n)]

    # N-gram解析（この例ではbigrams）
    bigrams = [b for l in documents for b in zip(l.split(' ')[:-1], l.split(' ')[1:])]
    bigram_freq = Counter(bigrams)
    top_bigrams = [' '.join(item[0]) for item in bigram_freq.most_common(top_n)]
    # すべてのトップ用語とbigramsを組み合わせる
    _top_entities = list(set(top_terms_tfidf) | set(top_terms_freq) | set(top_bigrams) | set(custom_dict))
    # オリジナルのクエリにトップエンティティを追加（ブラケットで囲む）
    _modified_query = query
    for entity in _top_entities:
        _modified_query = _modified_query.replace(entity, f"[{entity}]")

    return _modified_query, _top_entities


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
            modified_ver_query, entities = extract_top_entities(similar_documents, initial_query)
            print("modified_ver_query: ", modified_ver_query)
            print("entities: ", entities)
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
            return answer, self.input_txt
        else:
            answer = "申し訳ありません。データベースに不具合が生じているようです。開発者にお問い合わせください。"
            return answer, self.input_txt

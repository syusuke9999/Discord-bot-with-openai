from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from spacy.cli import download
from spacy.matcher import Matcher
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
            # spaCyの日本語モデルをロード
            try:
                nlp = spacy.load("ja_core_news_sm")
            except IOError:
                download("ja_core_news_sm")
                nlp = spacy.load("ja_core_news_sm")
            # マッチャーを初期化
            matcher = Matcher(nlp.vocab)
            # パターンを定義
            pattern1 = [{"LOWER": "効果"}, {"IS_PUNCT": True}]
            pattern2 = [{"LOWER": "取得優先度"}, {"IS_PUNCT": True}]
            pattern3 = [{"LOWER": "秒間"}]
            pattern4 = [{"LOWER": "メートル"}]
            pattern5 = [{"LOWER": "発動"}]
            # パターンをマッチャーに追加
            matcher.add("INFO", [pattern1, pattern2, pattern3, pattern4, pattern5])
            # 抽出したキーワードを保存するリスト
            extracted_keywords = []
            # similar_documentsは、documentオブジェクトのリストと仮定
            for doc_obj in similar_documents:
                text = doc_obj.page_content  # page_contentからテキストを取得
                doc = nlp(text)  # テキストを処理
                # マッチングを実行
                matches = matcher(doc)
                # 結果を表示とキーワードの抽出
                for match_id, start, end in matches:
                    span = doc[start:end]
                    print(f"Matched keyword: {span.text}, Position: {start}-{end}, Sentence: {span.sent.text}")
                    extracted_keywords.append(span.text)
            # 抽出したキーワードを使用してmodified_queryを作成
            modified_query = " AND ".join(extracted_keywords)
            print(f"Modified Query: {modified_query}")
            for doc in similar_documents:
                print("page_content= ", doc.page_content)
                print("metadata= ", str(doc.metadata))
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: qa_chain({"input_documents":
                                                                         similar_documents,
                                                                         "question": initial_query},
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

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langdetect import detect
import re
import os
import asyncio


def remove_english(text):
    # 英語のセンテンスを削除
    sentences = re.split(r'(\s*[.?!]\s*)', text)
    non_english_sentences = []

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i]
        # センテンスが空でない場合のみ言語検出を行う
        if sentence.strip():
            try:
                if detect(sentence) != 'en':
                    non_english_sentences.append(sentence + sentences[i + 1])
            except:
                # 言語検出に失敗した場合、センテンスをそのまま追加
                non_english_sentences.append(sentence + sentences[i + 1])

    # 最後のセンテンスが残っている場合、追加
    if len(sentences) % 2 != 0:
        non_english_sentences.append(sentences[-1])

    return "".join(non_english_sentences)


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, query):
        self.input_txt = query
        llm = load_llm("my_llm.json")
        embeddings = OpenAIEmbeddings()
        source_url = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            refine_prompt_template = (
                "The original question is as follows: {question}\n"
                "We have provided an existing answer: {existing_answer}\n"
                "We have the opportunity to refine the existing answer"
                "(only if needed) with some more context below.\n"
                "------------\n"
                "{context_str}\n"
                "------------\n"
                "Given the new context, refine the original answer to better "
                "answer. The refined answer must be Japanese.\n"
                "If the context isn't useful, return the original answer in Japanese."
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
                "Your answer should be in Japanese.\n"
                "answer the question: {question}\n"
            )
            initial_qa_prompt = PromptTemplate(
                input_variables=["context_str", "question"], template=initial_qa_template
            )
            qa_chain = load_qa_chain(
                ChatOpenAI(temperature=0),
                chain_type="refine",
                return_refine_steps=True,
                question_prompt=initial_qa_prompt,
                refine_prompt=refine_prompt
            )
            similar_documents = docsearch.similarity_search(query=query, k=4)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: qa_chain({"input_documents":
                                                                         similar_documents,
                                                                         "question": query},
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

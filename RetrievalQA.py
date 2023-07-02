from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import asyncio
from system_message import SystemMessage, Topic


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, input_txt):
        self.input_txt = input_txt
        llm = load_llm("my_llm.json")
        if "パークの効果" in input_txt:
            input_txt = input_txt.replace("パークの効果", "パークの性能と効果")
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76, top_k=10)
        source_url = ""
        answer = ""
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                                   base_retriever=docsearch.as_retriever())
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)
            # return_source_documentsプロパティをTrueにセット
            qa.return_source_documents = True
            # applyメソッドを使用してレスポンスを取得
            loop = asyncio.get_event_loop()
            print(f"Input dict before apply: {input_txt}")
            response = await loop.run_in_executor(None, lambda: qa.apply([self.input_txt]))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response[0]["result"]
                source_url = response[0]["source_documents"][0].metadata["source"]
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
                source_url = None
            """
            try:
                source_document_1 = response[0]['source_documents'][0].page_content
                source_document_2 = response[0]['source_documents'][1].page_content
                source_document_3 = response[0]['source_documents'][2].page_content
                source_document_4 = response[0]['source_documents'][3].page_content
            except (TypeError, KeyError, IndexError):
                source_document_1 = None
                source_document_2 = None
                source_document_3 = None
                source_document_4 = None
            else:
                system_message_instance = SystemMessage(topic=Topic.SELECT_MOST_RELEVANT_SOURCE, answer=answer,
                                                        source1=source_document_1, source2=source_document_2,
                                                        source3=source_document_3, source4=source_document_4)
                system_message_content = system_message_instance.get_system_message_content()
                system_message_json = {"role": "system", "content": system_message_content}
                hyper_parameters = {"model_name": "gpt-4", "max_tokens": 2000, "temperature":
                                    0, "top_p": 0, "presence_penalty":
                                        0, "frequency_penalty": 0}
                filled_prompt = system_message_content.format(system_answer=answer, source1=source_document_1,
                                                              source2=source_document_2,
                                                              source3=source_document_3,
                                                              source4=source_document_4)
                user_message_json = {"role": "user", "content": filled_prompt}
                import openai_api

                most_relevant_document = await openai_api.call_openai_api(hyper_parameters, system_message_json,
                                                                          user_message_json)
                index = 0
                source = response[0]['source_documents']
                for i, document in enumerate(source):
                    if most_relevant_document["choices"][0]["message"]["content"] in source[i].metadata['source']:
                        source_url = source.metadata['source']
                        break
                    else:
                        index += 1
                index -= 1
                print("source_documents: ", response[0]['source_documents'][index]['source_urls'])
            """
            return answer, source_url, self.input_txt
        else:
            answer = "申し訳ありません。データベースに不具合が生じているようです。開発者にお問い合わせください。"
            return answer, source_url, self.input_txt

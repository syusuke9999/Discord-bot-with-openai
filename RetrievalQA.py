from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import os
import pickle
import re


def Retrival(input_txt):
    # ファイルからllmを読み込む
    llm = load_llm("my_llm.json")
    # ファイルからdocumentsを読み込む
    with open("documents.pkl", "rb") as f:
        loaded_documents = pickle.load(f)
    text_splitter = TokenTextSplitter(chunk_size=1500, chunk_overlap=0, encoding_name="cl100k_base")
    texts = text_splitter.split_documents(loaded_documents)
    print(len(texts))
    documents = texts
    # documentsはlangchain.schema.Documentのリストとします
    for document in documents:
        document.page_content = re.sub('\n+', '\n', document.page_content)
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    embeddings = OpenAIEmbeddings()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    if not os.path.exists("./faiss_index"):
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("./faiss_index")
    docsearch = FAISS.load_local("./faiss_index", embeddings)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter,
                                                           base_retriever=docsearch.as_retriever())
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)
    response = qa.run(query=input_txt)
    return response

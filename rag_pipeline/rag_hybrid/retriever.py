from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from uuid import uuid4

def extract_content_from_pdf(file: str) -> List:
    loader = PyPDFLoader(file)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks

def setup_retrievers(docs, embedding_model_name):

    docs_content = [doc.page_content for doc in docs]
    bm25_retriever = BM25Retriever.from_documents(docs)

    # Dense retriever setup (vectorstore_retriever)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = Chroma(
        collection_name="collection1",
        embedding_function=embeddings,
    )
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    vectorstore_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Hybrid retriever setup (ensemble of vectorstore and BM25)
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retriever, bm25_retriever], weights=[0.7, 0.3])

    return vectorstore_retriever, ensemble_retriever, bm25_retriever

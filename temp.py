from langchain.retrievers import BM25Retriever, EnsembleRetriever
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.vectorstores import ElasticsearchStore
from rag_pipeline.Qdrant_hybrid_rag.indexing import create_dense_vector, dense_embedding_model, extract_content_from_pdf
import os
from dotenv import load_dotenv

load_dotenv()

es_username = os.getenv("es_user")
es_password = os.getenv("es_password")
index_name = "temp"


def initialize_es_client():
    es_client = Elasticsearch(
        hosts=["https://localhost:9200/"],
        basic_auth=(es_username, es_password),
        verify_certs=False
    )
    return es_client


def create_index(es_client):
    mappings = {
        "properties": {
            "content": {
                "type": "text"
            },
            "dense_vector": {
                "type": "dense_vector",
                "dims": 384  # Set this to the appropriate dimension for your model
            },
            "sparse_vector": {
                "type": "sparse_vector"  # Use object type to store the sparse vector as a dictionary
            }
        }
    }
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, mappings=mappings)


def generate_sparse_vector(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    sparse_vectors = []
    for i in range(tfidf_matrix.shape[0]):
        sparse_vector = {}
        for j in range(tfidf_matrix.shape[1]):
            weight = tfidf_matrix[i, j]
            if weight > 0:
                sparse_vector[feature_names[j]] = weight
        sparse_vectors.append(sparse_vector)

    return sparse_vectors


def test():
    es_client = initialize_es_client()
    create_index(es_client)

    chunks = extract_content_from_pdf("D:\Projects\RAG\data\LSTM.pdf")
    documents = [doc.page_content for doc in chunks]

    sparse_vectors = generate_sparse_vector(documents)

    for i, doc in enumerate(chunks):
        dense_embedding = create_dense_vector(doc, dense_embedding_model)

        document = {
            "content": doc.page_content,
            "dense_vector": dense_embedding,
            "sparse_vector": sparse_vectors[i]  # Store the sparse vector for the current document
        }

        es_client.index(index=index_name, id=str(i), body=document)

    # Create a BM25 retriever
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 10  # Set k to 10 for top results

    # Create a vector store retriever
    langchain_es_vectorstore = ElasticsearchStore(
        es_connection=es_client,
        index_name=index_name,
        embedding=dense_embedding_model  # Provide your embedding model here

    )
    vectorstore_retriever = langchain_es_vectorstore.as_retriever()

    # Create an ensemble retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retriever, keyword_retriever], weights=[0.5, 0.5])

    ''' 
    Perform retrieval 
    We have to take care of the embed_query function not being present 
    in the Sentence Transformer instead of it encode is available 
    But by default .invoke invokes embed_query to generate query vectors
    to perform similarity search.
    '''
    results = ensemble_retriever.invoke(input="What are LSTMS?")

    '''
    The code internally already reranks the results of similarity search 
    using weighted rank fusion and return top 4 results. 
    '''
    return results

if __name__ == "__main__":

# Call the test function
    top_result = test()
    print(top_result)

import os
import logging
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
from rag_pipeline.Qdrant_client import create_qdrant_client


load_dotenv()

Qdrant_API_KEY = os.getenv('QDRANT_API_KEY')
Qdrant_URL = os.getenv('QDRANT_URL')
Collection_Name = os.getenv('COLLECTION_NAME')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


dense_model_name = os.getenv("DENSE_MODEL")
sparse_model_name = os.getenv("SPARSE_MODEL")

dense_embedding_model = SentenceTransformer(dense_model_name)
sparse_embedding_model = SparseTextEmbedding(model_name=sparse_model_name)


def hybrid_search(query, limit= 5):

    client = create_qdrant_client()

    dense_query = list(dense_embedding_model.encode(query))
    sparse_query = list(sparse_embedding_model.embed([query]))[0]

    sparse_query = models.SparseVector(
        indices= sparse_query.indices.tolist(),
        values=sparse_query.values.tolist()
    )

    # sparse_results = client.query_points(
    #     collection_name=Collection_Name,
    #     query=sparse_query
    # )
    #
    # dense_results = client.query_points(
    #     collection_name=Collection_Name,
    #     query=dense_query
    # )

    results = client.query_points(
        collection_name=Collection_Name,
        prefetch= [
            models.Prefetch(
                query = sparse_query,
                using = "sparse",
                limit = limit
            ),
            models.Prefetch(
                query =dense_query,
                using = "dense",
                limit = limit
            )
        ],
        query= models.FusionQuery(fusion=models.Fusion.RRF)
    )
    #
    # sparse_documents = [point for point in sparse_results.points]
    # dense_documents = [point for point in dense_results.points]
    documents = [point for point in results.points]

    return {
        # "sparse_results": sparse_documents,
        # "dense_results": dense_documents,
        "combined_results":documents
    }

def search(query, limit= 5):

    client = create_qdrant_client()
    dense_query = list(dense_embedding_model.encode(query))

    # Perform the search
    results = client.query_points(
        collection_name=Collection_Name,
        query= dense_query,
        limit = limit
    )

    documents = [point for point in results.points]

    return {
        "combined_results":documents
    }
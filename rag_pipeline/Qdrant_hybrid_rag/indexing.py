import logging
import os
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from rag_pipeline.Qdrant_client import create_qdrant_client
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http.models import SparseVector
from tqdm import tqdm
from langchain.schema import Document

load_dotenv()

hybrid_collection = os.getenv("COLLECTION_NAME")

dense_model_name = os.getenv("DENSE_MODEL")
sparse_model_name = os.getenv("SPARSE_MODEL")

dense_embedding_model = SentenceTransformer(dense_model_name)
sparse_embedding_model = SparseTextEmbedding(model_name=sparse_model_name)


def extract_content_from_pdf(file: str) -> List[Document]:
    """
    Extract and split content from a PDF file into chunks.

    Args:
        file (str): Path to the PDF file.

    Returns:
        List: A list of Documents containing various attributes
        like page_content, metadata,etc. extracted from the PDF.
    """
    loader = PyPDFLoader(file)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks


def create_hybrid_collection(client: QdrantClient):
    """
    Create a collection in Qdrant if it does not exist.

    Args:
        client (QdrantClient): An instance of QdrantClient.
    """
    if not client.collection_exists(collection_name=hybrid_collection):
        client.create_collection(
            collection_name=hybrid_collection,
            vectors_config={
                'dense': models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config= {
                "sparse": models.SparseVectorParams(),
            }
        )
        logging.info(f"Created hybrid collection '{hybrid_collection}' in Qdrant.")

def create_sparse_vector(docs: Document, sparse_text_embedding_model) -> SparseVector:
    """
    Create a sparse vector from the text using BM42 approach.

    Args:
        docs (Document): A Document object with 'page_content'.
        sparse_text_embedding_model: An instance of SparseTextEmbedding.

    Returns:
        SparseVector: A Qdrant SparseVector object.
    """
    embeddings = list(sparse_text_embedding_model.embed([docs.page_content]))[0]

    if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
        return SparseVector(
            indices=embeddings.indices.tolist(),
            values=embeddings.values.tolist()
        )
    else:
        raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")


def create_dense_vector(docs: Document, model: SentenceTransformer) :
    """
    Encode a list of Document objects using a HuggingFace model.

    Args:
        docs (Document): A Document object with 'page_content'.
        model (SentenceTransformer): An instance of SentenceTransformer.

    Returns:
        List[float]: A list of embeddings, one for each document.
    """
    # Extract page content from documents
    embeddings = [model.encode(docs.page_content)]

    return embeddings[0].tolist()

def index_documents(file: str):
    """
    Extract text from a PDF and index it into Qdrant.

    Args:
        file (str): Path to the PDF file.
    """
    # Initialize client and embedding models
    client = create_qdrant_client()
    client.delete_collection(collection_name="rag_hybrid")

    # Extract and split PDF content
    documents = extract_content_from_pdf(file)

    create_hybrid_collection(client)


    for i, doc in enumerate(tqdm(documents, total=len(documents))):
        dense_embedding = create_dense_vector(doc, dense_embedding_model)
        sparse_embedding = create_sparse_vector(doc, sparse_embedding_model)

        client.upsert(
            collection_name=hybrid_collection,
            points=[models.PointStruct(
                id = i ,
                vector = {
                    "dense": dense_embedding,
                    "sparse": sparse_embedding
                },
                payload={
                    "content": doc.page_content,
                    "file_name": os.path.basename(file),
                    "metadata": doc.metadata
                }
            )]
        )
    logging.info(f"Upsert points with sparse and dense vectors into Qdrant.")


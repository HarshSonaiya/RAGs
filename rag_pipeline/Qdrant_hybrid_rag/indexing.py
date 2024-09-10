import logging
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http.models import SparseVector
from tqdm import tqdm
from langchain.schema import Document


qdrant_api_key = "QKadpncThByWzafBM2pJGJdArqoCoIeq-I9yggJHjuU3XRk1i6RVhg"
qdrant_url = "http://localhost:6333"
# Collection_Name = "rag"
dense_collection = "rag_dense"
sparse_collection = "rag_sparse"
dense_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")


def extract_content_from_pdf(file: str) -> List:
    """
    Extract and split content from a PDF file into chunks.

    Args:
        file (str): Path to the PDF file.

    Returns:
        List: A list of A list of dictionaries containing various attributes
        like page_content, metadata,etc. extracted from the PDF.
    """
    loader = PyPDFLoader(file)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks


def create_qdrant_client():
    """
    Create and return a Qdrant client.

    Returns:
        QdrantClient: An instance of QdrantClient.
    """
    return QdrantClient(url=qdrant_url)

def create_dense_collection(client: QdrantClient):
    """
    Create a collection in Qdrant if it does not exist.

    Args:
        client (QdrantClient): An instance of QdrantClient.
    """
    if not client.collection_exists(collection_name=dense_collection):
        client.create_collection(
            collection_name=dense_collection,
            vectors_config={
                'dense': models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            }
        )
        logging.info(f"Created dense vector collection '{dense_collection}' in Qdrant.")

def create_sparse_collection(client: QdrantClient):
    """
    Create a collection for sparse vectors in Qdrant if it does not exist.

    Args:
        client (QdrantClient): An instance of QdrantClient.
    """
    if not client.collection_exists(collection_name=sparse_collection):
        client.create_collection(
            collection_name=sparse_collection,
            vectors_config= {},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            }
        )

    logging.info(f"Created sparse vector collection '{sparse_collection}' in Qdrant.")


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
    logging.debug(f"Sparse embeddings: {embeddings}")

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
        List[List[float]]: A list of embeddings, one for each document.
    """
    # Extract page content from documents
    texts = docs.page_content

    # Get embeddings for the documents
    embeddings = model.encode([texts])
    logging.debug(f"Dense embeddings: {embeddings}")

    return embeddings

def index_documents(file: str):
    """
    Extract text from a PDF and index it into Qdrant.

    Args:
        file (str): Path to the PDF file.
    """
    # Initialize client and embedding models
    client = create_qdrant_client()

    # Extract and split PDF content
    documents = extract_content_from_pdf(file)

    # Create Qdrant collection
    create_dense_collection(client)
    create_sparse_collection(client)

    # Index dense embeddings into the dense collection
    dense_points = []
    for i, doc in enumerate(tqdm(documents, total=len(documents))):
        dense_embedding = create_dense_vector(doc, dense_embedding_model)
        dense_point = models.PointStruct(
            id=i,
            vector={"dense":dense_embedding[0]}
        )
        dense_points.append(dense_point)

    client.upsert(
        collection_name=dense_collection,
        points=dense_points
    )
    logging.info(f"Upserted {len(dense_points)} points with dense vectors into Qdrant.")

    # Index sparse embeddings into the sparse collection
    sparse_points = []
    for i, doc in enumerate(tqdm(documents, total=len(documents))):
        sparse_vector = create_sparse_vector(doc, sparse_embedding_model)
        sparse_point = models.PointStruct(
            id=i,
            vector={"sparse":sparse_vector}
        )
        # logging.info(f"Sparse points:{sparse_vector}")
        sparse_points.append(sparse_point)
    logging.info(type(sparse_points[0]),len(sparse_points))
    client.upsert(
        collection_name=sparse_collection,
        points=sparse_points
    )
    logging.info(f"Upserted {len(sparse_points)} points with sparse vectors into Qdrant.")
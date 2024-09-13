import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
qdrant_url = os.getenv('QDRANT_URL')

def create_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client.

    Returns:
        QdrantClient: An instance of QdrantClient.
    """
    return QdrantClient(url=qdrant_url)

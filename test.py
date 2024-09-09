from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")  # Replace with your Qdrant URL
collection_name = "rag"

print(client.get_collection(collection_name=f"{collection_name}"))


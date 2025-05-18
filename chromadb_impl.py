import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings())

collection = client.create_collection(name="text_embeddings")
collection.add(documents=texts, embeddings=embeddings, ids=["1", "2", "3"])

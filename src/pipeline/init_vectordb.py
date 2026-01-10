import json
import os
import chromadb
from chromadb.config import Settings

EMBEDDINGS_FILE = "data/embeddings/embeddings.json"
DB_DIR = "rag_chroma_db"

def init_vectordb(embeddings_file, db_dir):
    # Load embeddings
    with open(embeddings_file, "r") as f:
        data = json.load(f)
    
    # Init ChromaDB
    client = chromadb.PersistentClient(path=db_dir, settings=Settings(anonymized_telemetry=False))
    
    # Create collection
    collection = client.get_or_create_collection(name="t4_docs")
    
    # Add documents
    ids = [item["chunk_id"] for item in data]
    documents = [item["text"] for item in data]
    embeddings = [item["embedding"] for item in data]
    metadatas = [{"page": item["page"]} for item in data]
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Initialized vector DB with {len(data)} documents.")

if __name__ == "__main__":
    init_vectordb(EMBEDDINGS_FILE, DB_DIR)
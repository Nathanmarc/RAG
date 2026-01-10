import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

INPUT_FILE = "data/chunks/chosen_chunks.json"
OUTPUT_DIR = "data/embeddings"

def embed_chunks(chunks, model_name="intfloat/e5-large-v2"):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    
    embedded = []
    for chunk in chunks:
        text = chunk["text"]
        embedding = embeddings_model.embed_query(text)
        embedded.append({
            "chunk_id": chunk["chunk_id"],
            "text": text,
            "page": chunk["page"],
            "embedding": embedding
        })
    return embedded

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load chunks
    with open(INPUT_FILE, "r") as f:
        chunks = json.load(f)
    
    # Embed
    embedded_chunks = embed_chunks(chunks)
    
    # Save
    with open(os.path.join(OUTPUT_DIR, "embeddings.json"), "w") as f:
        json.dump(embedded_chunks, f, indent=2)
    
    print(f"Embedded {len(embedded_chunks)} chunks.")

if __name__ == "__main__":
    main()
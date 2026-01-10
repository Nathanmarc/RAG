import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/chunks"

def chunk_text(processed_text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    for page in processed_text:
        text = page["text"]
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "page": page["page"],
                "chunk_id": f"{page['page']}_{i}",
                "text": chunk
            })
    return all_chunks

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load processed text
    with open(os.path.join(INPUT_DIR, "processed_text.json"), "r") as f:
        processed_text = json.load(f)
    
    # Test different configs
    configs = [
        {"chunk_size": 256, "overlap": 50},
        {"chunk_size": 512, "overlap": 100},
        {"chunk_size": 1024, "overlap": 200}
    ]
    
    for config in configs:
        chunks = chunk_text(processed_text, config["chunk_size"], config["overlap"])
        filename = f"chunks_{config['chunk_size']}_{config['overlap']}.json"
        with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"Created {len(chunks)} chunks for config {config}")
    
    # Choose one, say 512/100
    chosen_chunks = chunk_text(processed_text, 512, 100)
    with open(os.path.join(OUTPUT_DIR, "chosen_chunks.json"), "w") as f:
        json.dump(chosen_chunks, f, indent=2)

if __name__ == "__main__":
    main()
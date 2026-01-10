import json
import re
import os

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

def preprocess_text(text_pages):
    processed = []
    for page in text_pages:
        text = page["text"]
        # Clean text: remove extra whitespaces, normalize
        text = re.sub(r'\s+', ' ', text).strip()
        # Standardize headers: perhaps capitalize or something, but for now, keep
        # Parse specific metrics
        # Find TDP
        tdp_match = re.search(r'TDP[:\s]*(\d+)W', text, re.IGNORECASE)
        if tdp_match:
            text = text.replace(tdp_match.group(0), f"TDP: {tdp_match.group(1)}W")
        
        # Find INT8/FP16 mentions
        text = re.sub(r'int8', 'INT8', text, flags=re.IGNORECASE)
        text = re.sub(r'fp16', 'FP16', text, flags=re.IGNORECASE)
        
        # TensorRT flags: look for common flags
        flags = re.findall(r'--\w+', text)
        if flags:
            text += f" TensorRT flags found: {', '.join(flags)}"
        
        processed.append({
            "page": page["page"],
            "text": text
        })
    return processed

def preprocess_tables(tables):
    # For tables, perhaps convert to text or keep as is
    # Normalize table data
    processed_tables = []
    for table in tables:
        # Assume table is list of lists
        normalized_table = []
        for row in table["table"]:
            normalized_row = [str(cell).strip() if cell else "" for cell in row]
            normalized_table.append(normalized_row)
        processed_tables.append({
            "page": table["page"],
            "table": normalized_table
        })
    return processed_tables

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load raw data
    with open(os.path.join(INPUT_DIR, "text.json"), "r") as f:
        text_pages = json.load(f)
    
    with open(os.path.join(INPUT_DIR, "tables.json"), "r") as f:
        tables = json.load(f)
    
    # Preprocess
    processed_text = preprocess_text(text_pages)
    processed_tables = preprocess_tables(tables)
    
    # Save
    with open(os.path.join(OUTPUT_DIR, "processed_text.json"), "w") as f:
        json.dump(processed_text, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "processed_tables.json"), "w") as f:
        json.dump(processed_tables, f, indent=2)
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    main()
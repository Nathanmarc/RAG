import pdfplumber
import json
import os
from pathlib import Path

PDF_PATH = "t4-tensor-core-datasheet-951643.pdf"
OUTPUT_DIR = "data/raw"

def extract_pdf(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    text_pages = []
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                text_pages.append({
                    "page": i + 1,
                    "text": text
                })
            
            # Extract tables
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append({
                    "page": i + 1,
                    "table": table
                })
    
    # Save text
    with open(os.path.join(output_dir, "text.json"), "w") as f:
        json.dump(text_pages, f, indent=2)
    
    # Save tables
    with open(os.path.join(output_dir, "tables.json"), "w") as f:
        json.dump(tables, f, indent=2)
    
    print(f"Extracted {len(text_pages)} text pages and {len(tables)} tables.")

if __name__ == "__main__":
    extract_pdf(PDF_PATH, OUTPUT_DIR)
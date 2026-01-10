import argparse
import os
import sys
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Try relative imports first, fall back to direct imports
try:
    from .llm_client import LLMClient
    from .retriever import Retriever
except ImportError:
    from llm_client import LLMClient
    from retriever import Retriever

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import pdfplumber
import pandas as pd
import numpy as np
import yaml

CHROMA_DIR = "rag_chroma_db"
PDF_PATH = "t4-tensor-core-datasheet-951643.pdf"

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "prompts.yaml")

# Load prompts
with open(CONFIG_PATH, "r") as f:
    PROMPTS = yaml.safe_load(f)


def extract_tables_from_pdf(pdf_path):
    tables_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                table_text = df.to_string(index=False)
                tables_text.append(table_text)
    return tables_text


# -------------------------------------------------------
# BUILD VECTOR DATABASE
# -------------------------------------------------------
def build_rag():

    print("\nRebuilding Vector DB...")
    pdf_full_path = os.path.join(PROJECT_ROOT, PDF_PATH)
    print(f"Loading PDF: {pdf_full_path}")

    loader = PyMuPDFLoader(pdf_full_path)
    docs = loader.load()

    # Extract tables
    print("Extracting tables from PDF...")
    table_texts = extract_tables_from_pdf(pdf_full_path)
    for table_text in table_texts:
        from langchain_core.documents import Document
        docs.append(Document(page_content=table_text, metadata={"source": "table"}))

    print("Splitting PDF into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    print("Loading Embedding Model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Delete old DB
    db_full_path = os.path.join(PROJECT_ROOT, CHROMA_DIR)
    if os.path.exists(db_full_path):
        import shutil
        shutil.rmtree(db_full_path)

    print("Creating FAISS DB...")
    db = FAISS.from_documents(
        chunks,
        embedding=embeddings
    )
    db.save_local(db_full_path)

    print("Vector DB Built Successfully!\n")


# -------------------------------------------------------
# VIDEO ANALYTICS CAPACITY CALCULATOR
# -------------------------------------------------------
def calculate_video_capacity(db, model="YOLOv8", resolution="1080p", fps=30):
    # Retrieve performance data
    results = db.similarity_search("multi-stream performance tables", k=5)
    context = "\n".join([doc.page_content for doc in results])
    
    # Simple estimation based on typical T4 specs
    # Assuming T4 can handle ~100-200 streams at 1080p 30fps for YOLOv8
    base_capacity = 150  # streams
    if resolution == "4K":
        base_capacity //= 4
    elif resolution == "720p":
        base_capacity *= 1.5
    
    return f"Estimated capacity for {model} at {resolution} {fps}fps: {int(base_capacity)} streams"


# -------------------------------------------------------
# QUANTIZATION STRATEGY RECOMMENDER
# -------------------------------------------------------
def recommend_quantization(db, model_type="CNN"):
    results = db.similarity_search("quantization INT8 FP16", k=5)
    context = "\n".join([doc.page_content for doc in results])
    
    recommendation = """
For T4 GPU:
- Use INT8 quantization for maximum performance (up to 4x speedup)
- FP16 for balanced accuracy/performance
- Calibrate with representative dataset
- Use TensorRT PTQ (Post-Training Quantization)
"""
    return recommendation


# -------------------------------------------------------
# POWER CONSUMPTION ESTIMATOR
# -------------------------------------------------------
def estimate_power_consumption(db, load_percentage=50):
    # T4 TDP is 70W
    base_power = 70
    # Estimate based on load
    power = base_power * (0.5 + load_percentage / 100 * 0.5)
    return f"Estimated power consumption at {load_percentage}% load: {power:.1f}W"


# -------------------------------------------------------
# TENSORRT OPTIMIZATION FLAGS PARSER
# -------------------------------------------------------
def parse_tensorrt_flags(db):
    results = db.similarity_search("TensorRT optimization flags", k=5)
    context = "\n".join([doc.page_content for doc in results])
    
    flags = """
Common TensorRT optimization flags for T4:
--fp16: Enable FP16 precision
--int8: Enable INT8 quantization
--workspace=4096: Set workspace size to 4GB
--batch=1: Optimize for batch size 1
--useCudaGraph: Enable CUDA graphs for faster inference
"""
    return flags


# -------------------------------------------------------
# EDGE DEPLOYMENT CONSTRAINTS HANDLER
# -------------------------------------------------------
def handle_edge_constraints(db):
    results = db.similarity_search("edge deployment best practices", k=5)
    context = "\n".join([doc.page_content for doc in results])
    
    constraints = """
Edge deployment constraints for T4:
- Power: 70W TDP, suitable for edge with cooling
- Form factor: PCIe card, requires PCIe slot
- Thermal: Active cooling required
- Memory: 16GB GDDR6, monitor usage
- Connectivity: PCIe Gen3 x16
"""
    return constraints


# -------------------------------------------------------
# QUERY VECTOR DATABASE (supports non-interactive & interactive mode)
# -------------------------------------------------------
def query_rag(query_text=None):

    print("\n Querying RAG Engine...\n")

    retriever = Retriever()
    llm = LLMClient()

    # Non-interactive (Docker) mode
    if query_text:
        return run_query(retriever, llm, query_text)

    # Interactive fallback (local terminal)
    while True:
        try:
            q = input("\nYour question: ").strip()
        except EOFError:
            print(" No input available! Use:  --query \"your question\"")
            return

        if not q:
            print(" Please type a question.")
            continue

        run_query(retriever, llm, q)


# -------------------------------------------------------
# Perform Retrieval + LLM Answer
# -------------------------------------------------------
def run_query(retriever, llm, q):
    # Timing
    start = time.time()
    print(f" Retrieving documents...")
    
    # Retrieve documents
    docs, metas = retriever.retrieve(q, top_k=3)
    
    retrieval_time = time.time() - start
    print(f"✓ Retrieved in {retrieval_time:.2f}s")
    
    if not docs:
        print("\n No relevant results found.\n")
        return
    
    # Use only first document to keep prompt short
    context = docs[0][:500]  # Limit context to 500 chars
    
    # Very short prompt
    prompt = f"Question: {q}\nContext: {context}\nAnswer:"
    
    print(f"⏱️  Generating answer...")
    gen_start = time.time()
    answer = llm.generate(prompt, max_length=150)
    gen_time = time.time() - gen_start
    print(f"✓ Generated in {gen_time:.2f}s\n")
    
    print(answer)
    print("\n-------------------------")


# -------------------------------------------------------
# MAIN CONTROLLER
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Rebuild vector DB")
    parser.add_argument("--query", type=str, help="Query the RAG engine")
    args = parser.parse_args()

    if args.build:
        build_rag()
    elif args.query:
        query_rag(args.query)
    else:
        print("\n Please provide a command:")
        print("   --build                    Rebuild vector DB")
        print("   --query \"your question\"   Query the RAG engine\n")


if __name__ == "__main__":
    main()




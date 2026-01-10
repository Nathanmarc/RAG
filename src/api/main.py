import sys
import os

# Add current directory to path for direct imports
sys.path.insert(0, os.path.dirname(__file__))

from rag_engine import query_rag
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
import time

app = FastAPI(title="T4 RAG Consultant API")

# Metrics
query_counter = Counter('rag_queries_total', 'Total number of queries')
query_latency = Histogram('rag_query_duration_seconds', 'Query latency in seconds')

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    start_time = time.time()
    query_counter.inc()
    try:
        result = query_rag(request.query)
        query_latency.observe(time.time() - start_time)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return generate_latest()

# Rate limiting middleware can be added with slowapi or similar
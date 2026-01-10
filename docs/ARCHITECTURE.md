# System Architecture Documentation

## Overview

This document provides a comprehensive technical overview of the end-to-end MLOps system for GPU product knowledge base. It details the architecture, component interactions, data flows, and design decisions.

---

## 1. System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                   │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐  │
│  │  Web Browser │         │   Python CLI │         │  External    │  │
│  │ (Streamlit)  │         │  Applications│         │  Systems     │  │
│  └──────────────┘         └──────────────┘         └──────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
┌───────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ FastAPI Server (Port 8000)                                     │  │
│  │  • /query - Main query endpoint                               │  │
│  │  • /health - Health check                                      │  │
│  │  • /metrics - Prometheus metrics                              │  │
│  │  • /docs - Swagger UI                                         │  │
│  │                                                                │  │
│  │ Middleware:                                                    │  │
│  │  • Rate Limiting (slowapi)                                    │  │
│  │  • Request Validation (Pydantic)                              │  │
│  │  • CORS Headers                                               │  │
│  │  • Request Logging                                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
                                  │
┌───────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LOGIC LAYER                            │
│                                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  RAG Engine      │  │   Retriever      │  │   LLM Client     │   │
│  │  ───────────     │  │  ───────────────  │  │  ─────────────   │   │
│  │ • Query route    │  │ • Query embedding│  │ • Generate text │   │
│  │ • Response       │  │ • Similarity     │  │ • Temperature    │   │
│  │   format         │  │   search         │  │   control        │   │
│  │ • Confidence     │  │ • Re-ranking     │  │ • Prompt format  │   │
│  │   scoring        │  │   logic          │  │                  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Prompt Management & Templates (prompts.yaml)                │   │
│  │  • Capacity Calculator                                       │   │
│  │  • Quantization Strategy                                     │   │
│  │  • TensorRT Optimization                                     │   │
│  │  • General RAG                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
                                  │
┌───────────────────────────────────────────────────────────────────────┐
│                   DATA & VECTOR DATABASE LAYER                        │
│                                                                        │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Vector Database         │  │  Embedding Models                │  │
│  │  ──────────────────      │  │  ─────────────────               │  │
│  │ • ChromaDB               │  │ • e5-large-v2 (768 dims)         │  │
│  │   └─ Persistent storage  │  │ • all-MiniLM-L6-v2 (384 dims)   │  │
│  │   └─ Collection: t4_docs │  │ • Pooling: mean                  │  │
│  │                          │  │                                  │  │
│  │ • FAISS (Optional)       │  │ Embedding Optimization:          │  │
│  │   └─ In-memory index     │  │ • Quantization (binary/product) │  │
│  │   └─ Scalable retrieval  │  │ • Dimension reduction            │  │
│  └──────────────────────────┘  └──────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Re-ranking: cross-encoder/ms-marco-MiniLM-L-6-v2            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
                                  │
┌───────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING PIPELINE                           │
│                                                                        │
│  PDF Input                                                             │
│    │                                                                    │
│    ├─► Extract (pdfplumber)                                            │
│    │   • Text extraction                                               │
│    │   • Table detection & extraction                                  │
│    │   • Page metadata                                                 │
│    │                                                                    │
│    ├─► Preprocess (preprocess.py)                                      │
│    │   • Text normalization                                            │
│    │   • Standardize formatting                                        │
│    │   • Extract specs (TDP, INT8, FP16, TensorRT flags)              │
│    │                                                                    │
│    ├─► Chunk (chunk_text.py)                                           │
│    │   • RecursiveCharacterTextSplitter                                │
│    │   • Configurable chunk_size, overlap                              │
│    │   • Preserve semantic boundaries                                  │
│    │                                                                    │
│    └─► Embed (embed.py)                                                │
│        • Generate embeddings for each chunk                             │
│        • Store with metadata                                            │
│                                                                        │
│    └─► Initialize VectorDB (init_vectordb.py)                          │
│        • Store embeddings in ChromaDB                                  │
│        • Index for fast retrieval                                      │
└───────────────────────────────────────────────────────────────────────┘
                                  │
┌───────────────────────────────────────────────────────────────────────┐
│              MONITORING & OBSERVABILITY LAYER                         │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Prometheus │  │   Grafana    │  │   MLflow     │  │  Logging  │ │
│  │   ──────────  │  │  ─────────── │  │  ──────────  │  │  ────────  │ │
│  │ • Metrics    │  │ • Dashboards │  │ • Experiments│  │ • Struct. │ │
│  │ • Alerts     │  │ • Alerts     │  │ • Parameters │  │   Logging │ │
│  │ • Scrape cfg │  │ • Visualization  │ • Metrics    │  │ • Audit   │ │
│  │ • PromQL     │  │               │  │ • Model reg. │  │   Trail   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Key Metrics:                                                 │   │
│  │  • rag_queries_total: Counter of all queries                │   │
│  │  • rag_query_duration_seconds: Histogram of latencies       │   │
│  │  • rag_retrieval_latency: Time to retrieve documents        │   │
│  │  • rag_llm_latency: Time for LLM generation                 │   │
│  │  • rag_query_errors: Count of failed queries               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
                                  │
┌───────────────────────────────────────────────────────────────────────┐
│            CI/CD & DEPLOYMENT INFRASTRUCTURE                          │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   GitHub     │  │   Docker     │  │  Docker      │  │ DVC Data  │ │
│  │   Actions    │  │  Containers  │  │  Compose     │  │ Versioning│ │
│  │  ──────────  │  │  ──────────── │  │  ──────────  │  │ ────────  │ │
│  │ • Automated  │  │ • Multi-stage │  │ • Orchest.   │  │ • Data    │ │
│  │   testing    │  │   builds      │  │ • Services   │  │   pipeline│ │
│  │ • Build      │  │ • Image push  │  │ • Networking │  │ • Versioning  │
│  │   artifact   │  │ • Registry    │  │ • Volumes    │  │ • Tracking │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Diagram

### Query Processing Flow

```
User Query
   │
   ▼
┌──────────────────────────────────────┐
│ FastAPI /query Endpoint              │
│ • Validate request (Pydantic)        │
│ • Rate limit check                   │
│ • Log incoming query                 │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Query Routing & Analysis             │
│ • Classify query type                │
│ • Select appropriate prompt template │
│ • Route to specialized processor     │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Query Embedding                      │
│ • Tokenize query                     │
│ • Generate embedding (e5-large-v2)  │
│ • Normalize to unit sphere           │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Initial Retrieval                    │
│ • Search in ChromaDB/FAISS           │
│ • Cosine similarity search           │
│ • Retrieve top 2K documents          │
│   (for later re-ranking)             │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Re-ranking (Optional)                │
│ • Cross-encoder scoring              │
│ • Sort by relevance score            │
│ • Select top K results               │
│   (default K=5)                      │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Context Assembly                     │
│ • Concatenate retrieved documents    │
│ • Add metadata (page, section)       │
│ • Format for LLM prompt              │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Prompt Construction                  │
│ • Select template (prompts.yaml)     │
│ • Inject query & context             │
│ • Set system prompt                  │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ LLM Generation                       │
│ • Send to Ollama/HuggingFace         │
│ • Control temperature (0.7)          │
│ • Max tokens: 512                    │
│ • Stream response (optional)         │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Post-processing & Scoring            │
│ • Extract answer text                │
│ • Calculate confidence score         │
│ • Extract source citations           │
│ • Format response JSON               │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Logging & Metrics                    │
│ • Record latency                     │
│ • Log to MLflow                      │
│ • Update Prometheus metrics          │
│ • Audit trail                        │
└──────────────────────────────────────┘
   │
   ▼
Response to User
{
  "answer": "...",
  "confidence": 0.92,
  "sources": [...],
  "latency_ms": 245
}
```

### Data Processing Pipeline

```
PDF File
  │
  ├─► extract_pdf.py
  │   └─► data/raw/
  │       ├─► text.json (pages & text)
  │       └─► tables.json (extracted tables)
  │
  ├─► preprocess.py
  │   └─► data/processed/
  │       ├─► processed_text.json (cleaned text)
  │       └─► processed_tables.json (normalized tables)
  │
  ├─► chunk_text.py
  │   └─► data/chunks/
  │       ├─► chunks_256_50.json
  │       ├─► chunks_512_100.json (selected)
  │       ├─► chunks_1024_200.json
  │       └─► chosen_chunks.json
  │
  ├─► embed.py
  │   └─► data/embeddings/
  │       └─► embeddings.json
  │
  └─► init_vectordb.py
      └─► rag_chroma_db/ (Vector DB)
```

---

## 3. Component Interaction Diagram

```
┌─────────────────┐
│  main.py        │ ◄─── FastAPI Server
│  (API Server)   │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
    ┌──────────┐      ┌──────────────┐
    │query()  │      │ health()     │
    │metrics()│      │ docs()       │
    └────┬─────┘      └──────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  rag_engine.py                  │
│  • query_rag()                  │
│  • run_query()                  │
│  • build_rag()                  │
└────────┬────────────────────────┘
         │
         ├──────────────┬──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐  ┌─────────────┐  ┌──────────┐
    │retriever│  │ llm_client  │  │prompts   │
    │ .py     │  │ .py         │  │.yaml     │
    └────┬────┘  └──────┬──────┘  └──────────┘
         │               │
         │               ▼
         │          ┌──────────────────────┐
         │          │ HuggingFace or Ollama│
         │          │ (LLM Model)          │
         │          └──────────────────────┘
         │
         ▼
    ┌────────────────────────────────┐
    │  Retriever (retriever.py)      │
    │  • retrieve()                  │
    │  • embedding model             │
    │  • cross-encoder reranker      │
    └────────┬───────────────────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │  ChromaDB / FAISS           │
    │  • Query embedding search   │
    │  • Return top-K docs        │
    │  • Metadata filtering       │
    └─────────────────────────────┘
         ▲
         │
    ┌────┴─────────────────┐
    │                      │
    ▼                      ▼
┌────────────┐      ┌────────────────┐
│ init_      │      │ chunk_text.py  │
│vectordb.py │      │ • Split docs   │
│ • Load     │      │ • Add metadata │
│   embed    │      └────────────────┘
│ • Store    │            ▲
│   in DB    │            │
└────────────┘      ┌─────┴─────────┐
                    │               │
                    ▼               ▼
            ┌──────────────┐  ┌─────────────┐
            │preprocess.py│  │embed.py     │
            │ • Clean txt │  │ • Generate  │
            │ • Normalize │  │   embeddings│
            └──────────────┘  └─────────────┘
                    ▲               ▲
                    │               │
                    └───┬───────────┘
                        │
                        ▼
                ┌────────────────┐
                │extract_pdf.py  │
                │ • Load PDF     │
                │ • Extract text │
                │ • Get tables   │
                └────────┬───────┘
                         │
                         ▼
                    ┌─────────┐
                    │PDF File │
                    └─────────┘
```

---

## 4. Technology Stack Justification

### Frontend & UI
- **Streamlit**: Rapid prototyping, no JavaScript needed, built-in caching
- **FastAPI**: Async support, auto-documentation, fast performance

### Data Processing
- **pdfplumber**: Precise text/table extraction, preserves formatting
- **LangChain**: Abstracts different LLM/embedding providers, handles chunking
- **sentence-transformers**: Fast, local embedding generation, semantic search

### Vector Database
- **ChromaDB**: User-friendly, persistent storage, metadata filtering
- **FAISS**: High performance at scale, supports GPU acceleration

### LLM Integration
- **Ollama**: Local LLM hosting, privacy-preserving, easy model switching
- **HuggingFace Transformers**: Fine-tuning capability, model hub access

### Monitoring & MLOps
- **MLflow**: Experiment tracking, model registry, reproducibility
- **Prometheus + Grafana**: Industry standard, real-time monitoring, alerting
- **DVC**: Data versioning, pipeline reproducibility

### Deployment
- **Docker**: Containerization for reproducible environments
- **Docker Compose**: Local multi-service orchestration
- **GitHub Actions**: CI/CD automation

---

## 5. Scalability Considerations

### Horizontal Scaling
1. **API Layer**: Deploy multiple FastAPI instances behind load balancer (Nginx)
2. **Vector DB**: Shard embeddings across multiple ChromaDB instances
3. **LLM**: Multi-GPU setup with distributed inference (Ray, vLLM)

### Vertical Scaling
1. **Embeddings**: Use quantized models (product quantization)
2. **VectorDB**: Implement HNSW indexing instead of flat search
3. **LLM**: Deploy larger models on GPU with reduced precision (FP16/INT8)

### Performance Optimization
1. **Caching**: Implement Redis for query result caching
2. **Async Processing**: Use Celery for background jobs
3. **Batch Processing**: Process multiple queries in parallel
4. **Index Optimization**: Regular maintenance of vector indices

### Fault Tolerance
1. **Redundancy**: Multi-instance deployment with health checks
2. **Backup**: Regular backups of vector database
3. **Circuit Breaker**: Handle LLM service failures gracefully
4. **Graceful Degradation**: Return cached results if LLM unavailable

---

## 6. Security Considerations

### API Security
- Rate limiting (prevent abuse)
- Request validation (Pydantic)
- CORS configuration
- API key authentication (future)

### Data Security
- Vector DB encryption at rest
- Sensitive data filtering in logs
- Audit trail of all queries
- Data retention policies

### Infrastructure Security
- Container scanning (Trivy)
- Secret management (environment variables)
- Network isolation (Docker networks)
- HTTPS/TLS for cloud deployments

---

## 7. Failure Modes & Recovery

### Failure Scenario: Vector DB Unavailable
- Status: Error response to user
- Recovery: Auto-restart container, fallback to cache
- Prevention: Health checks, automated recovery

### Failure Scenario: LLM Service Down
- Status: Graceful degradation, return retrieved docs only
- Recovery: Retry with exponential backoff
- Prevention: Model versioning, alternate providers

### Failure Scenario: Memory Exhaustion
- Status: Service crash
- Recovery: Automated restart with memory limits
- Prevention: Implement query batching, cache eviction

---

## 8. Future Architecture Enhancements

1. **Multi-Model Ensemble**: Combine multiple retriever architectures
2. **Adaptive Chunking**: Semantic-aware chunking based on document structure
3. **Query Expansion**: Use query reformulation for better retrieval
4. **Reranking Ensemble**: Combine multiple re-ranking models
5. **Knowledge Graph**: Extract and store structured knowledge
6. **Fine-tuning Pipeline**: Custom model adaptation per GPU product
7. **A/B Testing**: Compare different configurations in production
8. **Federated Learning**: Train models across multiple datasets

---

**Last Updated:** November 2025

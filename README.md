# End-to-End MLOps System: GPU Product Knowledge Base

## Project Overview

This is a complete MLOps system that transforms GPU product documentation (PDF datasheets) into an intelligent, queryable knowledge base using Retrieval-Augmented Generation (RAG). The system acts as a "GPU specialist AI" that becomes an expert in a specific GPU product and can answer technical questions with precision and contextual accuracy.

### Key Features

- **Intelligent PDF Processing**: Extracts and intelligently chunks GPU documentation with semantic preservation
- **Vector Database**: Semantic search using embeddings (sentence-transformers/e5-large-v2)
- **RAG Engine**: Combines retrieval + LLM generation for accurate, contextualized answers
- **Confidence Scoring**: Each response includes confidence metrics and source citations
- **MLOps Pipeline**: Complete experiment tracking with MLflow, data versioning with DVC
- **Real-time Monitoring**: Prometheus + Grafana for system metrics and alerts
- **Interactive UI**: Streamlit-based chat interface with conversation history
- **Production-Ready**: Docker, Docker Compose, and Kubernetes-ready deployment
- **CI/CD Automation**: GitHub Actions for automated testing and deployment

### Example Query Types

- **Video Analytics Capacity**: "How many YOLOv8 video streams can T4 process simultaneously?"
- **Quantization Guidance**: "How to quantize my model to INT8 for T4 inference?"
- **TensorRT Optimization**: "What's the optimal TensorRT configuration for T4?"
- **Power Analysis**: "What's the power consumption at different inference loads?"
- **Edge Deployment**: "What are the constraints for T4 edge deployment?"
- **General Q&A**: Any technical question about the GPU product

---

## Architecture Diagram

The system follows a layered architecture with clear separation of concerns:

**User Interface Layer** → **Application Layer** → **Data Layer** → **Infrastructure Layer**

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Processing** | pdfplumber, PyPDF2 | Extract text and tables from datasheets |
| **Text Processing** | LangChain, spaCy | Preprocessing and chunking |
| **Embeddings** | sentence-transformers (e5-large-v2) | Generate semantic embeddings |
| **Vector DB** | ChromaDB, FAISS | Store and retrieve embeddings |
| **Retrieval** | LangChain, Cross-Encoder | Semantic search + re-ranking |
| **LLM** | Ollama (Llama2), HuggingFace Transformers | Natural language generation |
| **API Framework** | FastAPI | REST API with async support |
| **Frontend** | Streamlit | Interactive chat interface |
| **MLOps Tracking** | MLflow | Experiment tracking and model registry |
| **Data Versioning** | DVC | Track dataset changes |
| **Monitoring** | Prometheus | Metrics collection |
| **Visualization** | Grafana | Real-time dashboards |
| **Containerization** | Docker, Docker Compose | Reproducible deployments |
| **CI/CD** | GitHub Actions | Automated pipelines |

---

## Quick Start

### 1. Install Dependencies

```bash
git clone <repo-url>
cd ko
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build Vector Database

```bash
python rag_engine.py --build
```

### 3. Query the System

```bash
python rag_engine.py --query "How many YOLOv8 streams can T4 process?"
```

### 4. Start REST API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# Access at http://localhost:8000/docs
```

### 5. Launch Streamlit UI

```bash
streamlit run app.py --server.port 8501
```

### 6. Run with Docker

```bash
docker-compose up -d
```

---

## API Documentation

### Query Endpoint
```
POST /query
Content-Type: application/json

{
  "query": "How many YOLOv8 video streams can T4 process simultaneously?"
}
```

### Health Check
```
GET /health
```

### Metrics
```
GET /metrics
```

### Swagger Documentation
```
GET /docs
GET /redoc
```

---

## Environment Variables

Create `.env` file:

```bash
LLM_MODEL=llama2
LLM_BASE_URL=http://localhost:11434
VECTORDB_PATH=rag_chroma_db
EMBEDDING_MODEL=intfloat/e5-large-v2
TOP_K_RETRIEVAL=5
API_PORT=8000
RATE_LIMIT_PER_MINUTE=60
```

---

## Documentation

- **ARCHITECTURE.md**: System design and component interactions
- **DEPLOYMENT_GUIDE.md**: Cloud deployment procedures
- **evaluation_report.md**: Test results and metrics

---

## Troubleshooting

### "No relevant results found"
```bash
# Rebuild vector database
python rag_engine.py --build
```

### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve
```

### Out of Memory
```bash
# Use smaller model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

This thing barely works. 

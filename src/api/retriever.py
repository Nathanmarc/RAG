import os
import time
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Retriever:
    _instance = None
    _reranker_model = None
    
    def __new__(cls, db_path=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Singleton pattern to avoid reloading models
        if cls._instance is None:
            cls._instance = super(Retriever, cls).__new__(cls)
            
            # Resolve db_path to project root
            if db_path is None:
                # Get project root (two levels up from this file: src/api/ -> project root)
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                db_path = os.path.join(project_root, "rag_chroma_db")
            
            cls._instance.db_path = db_path
            cls._instance.embeddings = None
            cls._instance.reranker = None
            cls._instance.db = None
            cls._instance._load_models(embedding_model)
        return cls._instance
    
    def _load_models(self, embedding_model):
        # Load embedding model
        start = time.time()
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"✓ Embedding model loaded in {time.time() - start:.2f}s")
        
        # Load FAISS index if it exists
        if os.path.exists(self.db_path):
            start = time.time()
            print(f"Loading FAISS index from: {self.db_path}")
            try:
                self.db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"✓ FAISS index loaded in {time.time() - start:.2f}s")
            except PermissionError as e:
                print(f"⚠ Permission denied accessing database: {e}")
                print(f"  Make sure no other process is using the files")
                self.db = None
            except Exception as e:
                print(f"✗ Error loading FAISS index: {e}")
                self.db = None
        else:
            print(f"⚠ Database not found at: {self.db_path}")
            print(f"   Run: python -m src.api.rag_engine --build")
            self.db = None
    
    def _load_reranker(self):
        """Lazy load reranker only when needed"""
        if self.reranker is None:
            print("Loading reranker model...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return self.reranker

    def retrieve(self, query, top_k=5, rerank=False):
        """Retrieve documents using FAISS vector store
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank: Whether to use cross-encoder reranking (slower but more accurate)
        """
        if self.db is None:
            return [], []
        
        # Retrieve documents (fast, no reranking by default)
        start = time.time()
        results = self.db.similarity_search(query, k=top_k)
        search_time = time.time() - start
        print(f"✓ Search completed in {search_time:.3f}s, found {len(results)} results")
        
        if not results:
            return [], []
        
        documents = [doc.page_content for doc in results]
        metadatas = [doc.metadata for doc in results]
        
        # Optional reranking (disabled by default for speed)
        if rerank and len(documents) > 1:
            reranker = self._load_reranker()
            pairs = [[query, doc] for doc in documents]
            scores = reranker.predict(pairs)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            reranked_docs = [documents[i] for i in ranked_indices[:top_k]]
            reranked_metas = [metadatas[i] for i in ranked_indices[:top_k]]
            return reranked_docs, reranked_metas
        
        return documents, metadatas

# Usage
if __name__ == "__main__":
    retriever = Retriever()
    if retriever.db is not None:
        docs, metas = retriever.retrieve("T4 TDP power")
        print(docs)
    else:
        print("Vector database not found. Run: python rag_engine.py --build")
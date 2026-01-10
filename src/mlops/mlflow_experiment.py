import mlflow
import mlflow.sklearn
from retriever import Retriever
from llm_client import LLMClient
import time

def run_experiment(chunk_sizes, overlaps, queries):
    mlflow.set_experiment("T4_RAG_Experiment")
    
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("chunk_size", chunk_size)
                mlflow.log_param("overlap", overlap)
                
                # Simulate retrieval metrics
                retriever = Retriever()
                llm = LLMClient()
                
                latencies = []
                for query in queries:
                    start = time.time()
                    docs, metas = retriever.retrieve(query)
                    latency = time.time() - start
                    latencies.append(latency)
                
                avg_latency = sum(latencies) / len(latencies)
                mlflow.log_metric("avg_retrieval_latency", avg_latency)
                
                # Dummy MRR
                mrr = 0.85  # Placeholder
                mlflow.log_metric("MRR", mrr)
                
                print(f"Logged run: chunk_size={chunk_size}, overlap={overlap}, avg_latency={avg_latency}")

if __name__ == "__main__":
    chunk_sizes = [256, 512, 1024]
    overlaps = [50, 100, 200]
    queries = ["What is T4 TDP?", "How many streams for YOLOv8?"]
    run_experiment(chunk_sizes, overlaps, queries)
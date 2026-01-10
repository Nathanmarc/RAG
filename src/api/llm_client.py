from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

class LLMClient:
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name=None):
        # Singleton pattern to avoid reloading model
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name=None):
        # Skip re-initialization if already done
        if LLMClient._initialized:
            return
        
        # Use Mistral 7B Instruct - good quality + reasonable speed
        if model_name is None:
            model_name = os.environ.get("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        
        print(f"Loading LLM: {model_name}")
        
        try:
            # Use text-generation pipeline with optimizations
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load with GPU optimizations: {e}")
            print("Falling back to CPU mode...")
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                trust_remote_code=True
            )
        
        LLMClient._initialized = True

    def generate(self, prompt, max_length=500):
        try:
            # Truncate prompt to avoid exceeding model limits
            # Keep only the last 500 characters (roughly 100 tokens)
            if len(prompt) > 500:
                prompt = prompt[-500:]
            
            result = self.pipeline(
                prompt,
                max_new_tokens=max_length,
                do_sample=True,  # Enable sampling for varied responses
                temperature=0.8,  # Higher for more variation
                top_p=0.95,
                top_k=50,  # Add top-k for better diversity
                truncation=True,
                return_full_text=False
            )
            return result[0]['generated_text'].strip() if result else ""
        except Exception as e:
            print(f"Error generating response: {e}")
            # If generation fails, return empty - rag_engine will show docs instead
            return ""

# Usage
if __name__ == "__main__":
    client = LLMClient()
    response = client.generate("What is the T4 GPU?")
    print(response)
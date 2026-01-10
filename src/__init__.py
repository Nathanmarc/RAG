"""GPU Knowledge Base RAG System - Root Package."""

__version__ = "1.0.0"
__author__ = "MLOps Team"
__description__ = "End-to-end MLOps system for GPU product knowledge base using RAG"

# Import main modules for easy access
from src.pipeline import *
from src.api import *
from src.ui import *
from src.mlops import *

__all__ = ["pipeline", "api", "ui", "mlops"]

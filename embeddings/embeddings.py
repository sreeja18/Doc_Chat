import numpy as np
import uuid
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class Embedding_Manager:    
    """Handles embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embeddings manager

        Args:
            model_name: Huggingface model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Sentence Transformer model"""
        try:
            print(f"Loading embeddings model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully, Embedding Dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


# Initialize embedding manager
#embedding_manager = Embedding_Manager() calling in MAIN
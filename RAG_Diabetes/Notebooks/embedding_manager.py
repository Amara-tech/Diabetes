import google.generativeai as genai
from typing import List
import numpy as np


class EmbeddingManager:
    """Handles document embedding using Google Gemini embedding models"""

    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initialize the EmbeddingManager with a Gemini model.
        
        Args:
            model_name: Google embedding model name (e.g., 'text-embedding-004')
        """
        self.model_name = model_name
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using the Gemini embedding API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            np.ndarray of embeddings with shape (len(texts), embedding_dim)
        """
        print(f"Generating embeddings using {self.model_name} for {len(texts)} texts...")

        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text
            )
            embeddings.append(result['embedding'])

        embeddings_array = np.array(embeddings)
        print(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array

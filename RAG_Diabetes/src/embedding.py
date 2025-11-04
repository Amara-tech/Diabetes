from typing import List
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split loaded documents into smaller chunks."""
        self.documents = documents
        if not self.documents:
            raise ValueError("No documents loaded. Run process_documents() first.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        split_docs = text_splitter.split_documents(self.documents)
        print(f"Split {len(self.documents)} documents into {len(split_docs)} chunks.")
        if split_docs:
            print(f"\nExample chunk:\nContent: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")
        return split_docs
                
    def generate_embeddings(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """
        Generate embeddings for a list of texts using the Gemini embedding API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            np.ndarray of embeddings with shape (len(texts), embedding_dim)
        """
        print(f"Generating embeddings using {self.model_name} for {len(texts)} texts...")

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = genai.embed_content(
                model=self.model_name,
                content=batch
        )
            print("Result keys:", result.keys())
            print("Result sample:", result)  
            batch_embeddings = [r['embedding'] for r in result['embeddings']]
            embeddings.extend(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")

        embeddings_array = np.array(embeddings)
        print(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array

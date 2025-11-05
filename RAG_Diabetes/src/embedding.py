from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
client = genai.Client(api_key=api_key)


class EmbeddingManager:
    """Handles document embedding using Google Gemini embedding models"""

    def __init__(self, model_name: str = "gemini-embedding-001", output_dim: int=768):
        """
        Initialize the EmbeddingManager with a Gemini model.
        
        Args:
            model_name: Google embedding model name (e.g., 'text-embedding-004')
        """
        self.model_name = model_name
        self.client = genai.Client()
        self.output_dim = output_dim
        
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
                
    def generate_embeddings(self, texts: List[str], batch_size: int = 5) -> np.ndarray:
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
            print(f"Processing batch {i//batch_size + 1}...")
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=genai.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",   
                    output_dimensionality=self.output_dim
                )
                )
                batch_embeddings = [np.array(e.values) for e in result.embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding text: {e}")
        embeddings_array = np.array(embeddings)
        print(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array


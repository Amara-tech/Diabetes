from typing import List, Any
import chromadb
import uuid
import numpy as np
import os

class VectorStore:
    """Manages document embeddings in a Chromadb vector store"""
    
    def __init__(self, collection_name: str = "pdf_document", persist_directory: str = "../RAG_DOCs/vector_store"):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
        
    def _initialize_store(self):
        """Initialize Chromadb client and collection"""    
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata= {"description": "PDF document embedding for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing document in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise    
        
    def add_docs(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Preparing {len(documents)} documents to add to vector store...")
        
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        # --- LOOP ---
        # 1. Build the lists first
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Make a copy of the metadata to avoid modifying the original doc
            metadata = dict(doc.metadata) 
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_text.append(doc.page_content)
            
            embeddings_list.append(embedding.tolist())
        # --- LOOP ENDS ---
        
        # --- BATCH ADD ---
        # 2. Now, add everything in a single, efficient batch operation
        try:
            print(f"Adding {len(ids)} documents to collection in one batch...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(ids)} documents to vector db.")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")    
            raise
    
    def delete_vector_db(self):
        """Completely delete the vector database directory and reset the client."""
        import shutil

        if os.path.exists(self.persist_directory):
            print(f"Deleting existing vector DB at: {self.persist_directory}")
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        else:
            print("No existing vector DB found to delete.")

        # Recreate clean directory and reset client
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embedding for RAG"}
        )
        print(f"Fresh vector DB initialized at: {self.persist_directory}")
        
  
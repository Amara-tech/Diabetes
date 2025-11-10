from typing import List, Any
import chromadb
import uuid
import numpy as np
import os
import shutil


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
        
        # BUILD THE LISTS  
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata) 
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        
        #  ADD TO DB IN SMALLER CHUNKS 
        BATCH_SIZE = 4000  
        
        total_added = 0
        for i in range(0, len(ids), BATCH_SIZE):
            # Create slices for the current mini-batch
            batch_ids = ids[i:i + BATCH_SIZE]
            batch_embeddings = embeddings_list[i:i + BATCH_SIZE]
            batch_metadatas = metadatas[i:i + BATCH_SIZE]
            batch_documents = documents_text[i:i + BATCH_SIZE]
            
            print(f"Adding batch {i//BATCH_SIZE + 1} ({len(batch_ids)} documents)...")
            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                total_added += len(batch_ids)
                print(f"Successfully added {total_added} documents so far.")
            
            except Exception as e:
                print(f"Error adding batch {i//BATCH_SIZE + 1}: {e}")    
                raise
        
        print(f"\nSuccessfully added all {total_added} documents to vector db.")
        print(f"Total documents in collection: {self.collection.count()}")
    
    
    def delete_vector_db(self):
        """
        Completely deletes the persistent vector store directory from disk
        and re-initializes a fresh, empty one.
        """
        if os.path.exists(self.persist_directory):
            print(f"Deleting existing vector DB at: {self.persist_directory}")
            try:
                shutil.rmtree(self.persist_directory)
                print("Deletion successful.")
            except Exception as e:
                print(f"Error deleting directory {self.persist_directory}: {e}")
                raise
        else:
            print("No existing vector DB found to delete.")

        # After deletion, re-initialize to have a clean, ready-to-use store
        print("Re-initializing a fresh vector store...")
        self._initialize_store()    
        
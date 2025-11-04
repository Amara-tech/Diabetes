from src.vectorstore import VectorStore
from typing import List, Any, Dict
from src.embedding import EmbeddingManager



class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    def __init__(self, vector_strore: VectorStore, embedding_manager: EmbeddingManager):
        """Initialize the retriever

        Args:
            vector_strore (VectorStore): Vector store containing document embeddings
            embedding_manager (EmbeddingManager): Manager for generating embeddings
        """
        self.vector_store = vector_strore
        self.embedding_manager = embedding_manager
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
        """   
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        #Generate Embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        #Search Vector Store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    #Convert distances to similarity score(ChromaDB uses cosine distances)
                    similarity_score = 1-distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'rank': i+1
                        })
                print(f"Retrieved {len(retrieved_docs)} document(after filtering)")
            else:
                print("No documents found")  
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
                           
        
        
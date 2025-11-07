from src.data_loader import DocumentProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.generator import RecommendationGenerator
from src.search import RAGRetriever
import google.generativeai as genai
from dotenv import load_dotenv
import os

class Recommender:
    """High-level RAG system for diabetes recommendation"""

    def __init__(self, docs_path: str = "RAG_DOCs/Recommend"):
        """Initialize RAG components"""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)

        # Initialize components
        self.doc_processor = DocumentProcessor(docs_path)
        self.embedding_manager = EmbeddingManager()
        self.vectorstore = VectorStore(collection_name="recommendation_document", persist_directory="RAG_DOCs/vector_store_reco")
        self.retriever = RAGRetriever(self.vectorstore, self.embedding_manager)
        self.generator = RecommendationGenerator()

        # Load and embed documents
        self.docs = None
        self.embeddings = None
        self.initialize_knowledge_base()
        
    def initialize_knowledge_base(self):
        """Load, split, and embed documents once"""
        # Check if vector store already has documents
        existing_count = self.vectorstore.collection.count()
        
        if existing_count > 0:
            print(f"Knowledge base already initialized with {existing_count} documents.")
            print("\n\n Skipping document loading.")
            return
        
        # Only load and embed if vector store is empty
        docs = self.doc_processor.process_all()
        split_docs = self.embedding_manager.split_documents(docs)
        texts = [d.page_content for d in split_docs]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vectorstore.add_docs(split_docs, embeddings)
        self.docs = split_docs
        self.embeddings = embeddings
        print("Knowledge base initialized successfully.")

    def recommend(self, query: str, user_data: dict, prediction: str):
        """Main RAG pipeline"""
        # Retrieve
        retrieved_docs = self.retriever.retrieve(query)

        # Augment & Generate
        result = self.generator.generate(
            retrieved_docs=retrieved_docs,
            user_data=user_data,
            prediction=prediction
        )

        return result
    
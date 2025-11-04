from src.data_loader import DocumentProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.generator import RecommendationGenerator
from src.search import RAGRetriever
import google.generativeai as genai
from dotenv import load_dotenv
import os

class RAGRecommender:
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
        self.vectorstore = VectorStore(collection_name="pdf_document")
        self.retriever = RAGRetriever(self.vectorstore, self.embedding_manager)
        self.generator = RecommendationGenerator()

        # Load and embed documents
        self.docs = None
        self.embeddings = None
        self.initialize_knowledge_base()
    def initialize_knowledge_base(self):
        """Load, split, and embed documents once"""
        docs = self.doc_processor.process_all()
        split_docs = self.embedding_manager.split_documents(docs)
        texts = [d.page_content for d in split_docs]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vectorstore.add_docs(split_docs, embeddings)
        self.docs = split_docs
        self.embeddings = embeddings
        print("Knowledge base initialized successfully.")

    def recommend(self, query: str, user_data: str, prediction: str):
        """Main RAG pipeline"""
        # Retrieve
        retrieved_docs = self.retriever.retrieve(query, self.vectorstore)

        # Step 2: Augment & Generate
        result = self.generator.generate(
            retrieved_docs=retrieved_docs,
            user_data=user_data,
            prediction=prediction
        )

        return result
    
if __name__ == "__main__":
    rag = RAGRecommender()

    user_query = "What dietary steps can help control blood sugar?"
    user_data = "Age: 45, BMI: 29, Glucose: 150, BloodPressure: 85"
    prediction = "At risk of diabetes"

    response = rag.recommend(user_query, user_data, prediction)
    print("\n🩺 Health Recommendation:\n")
    print(response)
    
    
    
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
import google.generativeai as genai

from src.data_loader import DocumentProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.generator import PreprocessingGeneration
from src.search import RAGRetriever


class Preprocessing:
    def __init__(self, docs_path: str = "RAG_DOCs/Prepros"):
        """Initialize RAG components and configure API."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)
        
        # Initialize components
        self.docs_path = docs_path
        self.embedding_manager = EmbeddingManager()
        self.vectorstore = VectorStore(collection_name="preprocessing_document", persist_directory="RAG_DOCs/vector_store_prepro")
        self.retriever = RAGRetriever(self.vectorstore, self.embedding_manager)
        self.generator = PreprocessingGeneration()

        # Load and embed documents
        self.docs = None
        self.embeddings = None
        self.initialize_knowledge_base()

    def load_documents(self):
        """
        Load all .pdf and .csv files in the given directory into LangChain Documents.
        - PDFs are processed with DocumentProcessor (page by page)
        - CSVs are processed manually, one row per Document
        """
        pdf_docs_all = []
        csv_docs_all = []

        for filename in os.listdir(self.docs_path):
            file_path = os.path.join(self.docs_path, filename)
            if filename.startswith("."):
                continue

            if filename.lower().endswith(".pdf"):
                print(f"Detected PDF file: {filename}")
                doc_processor = DocumentProcessor(self.docs_path, rag_type="preprocessing")
                pdf_docs = doc_processor.process_all()
                pdf_docs_all.extend(pdf_docs)

            elif filename.lower().endswith(".csv"):
                print(f"Detected CSV file: {filename}")
                csv_docs = self._load_csv_as_documents(file_path)
                csv_docs_all.extend(csv_docs)

        return pdf_docs_all, csv_docs_all

    def _load_csv_as_documents(self, file_path: str):
        """Convert each CSV row into a LangChain Document."""
        df = pd.read_csv(file_path)
        documents = []
        for index, row in df.iterrows():
            content = "\n".join(f"{col}: {val}" for col, val in row.items())
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": file_path, "row_index": index}
                )
            )
        return documents
    def initialize_knowledge_base(self):
        existing_count = self.vectorstore.collection.count()

        # Detect incomplete or corrupted DB (for example, fewer than expected docs)
        MIN_EXPECTED_DOCS = 3000

        if existing_count > 0:
            if existing_count < MIN_EXPECTED_DOCS:
                print(f"Detected incomplete vector DB ({existing_count} docs). Rebuilding...")
                self.vectorstore.delete_vector_db()
            else:
                print(f"Knowledge base already initialized with {existing_count} documents.")
                return
        else:
            print("No existing vector DB found. Creating new one...")

            # Rebuild embeddings fresh
            pdf_docs, csv_docs = self.load_documents()

            # Handle CSV documents (no splitting)
            if csv_docs:
                print(f"Embedding {len(csv_docs)} CSV documents...")
                csv_texts = [d.page_content for d in csv_docs]
                csv_embeddings = self.embedding_manager.generate_embeddings(csv_texts)
                self.vectorstore.add_docs(csv_docs, csv_embeddings)
                print(f"Added {len(csv_docs)} CSV docs to vector DB")

            # Handle PDF documents (with splitting)
            if pdf_docs:
                print(f"Splitting and embedding {len(pdf_docs)} PDF documents...")
                split_pdf_docs = self.embedding_manager.split_documents(pdf_docs)
                pdf_texts = [d.page_content for d in split_pdf_docs]
                pdf_embeddings = self.embedding_manager.generate_embeddings(pdf_texts)
                self.vectorstore.add_docs(split_pdf_docs, pdf_embeddings)
                print(f"✓ Added {len(split_pdf_docs)} split PDF docs to vector DB")
            print("Knowledge base initialized successfully.")
    
    
    
    # In Preprocessing class
    def recommend(self, query: str, user_data_dict: dict):
        retrieved_docs = self.retriever.retrieve(query, top_k=5)
        result = self.generator.infer_missing_data(
            retrieved_docs=retrieved_docs,
            user_data=user_data_dict,
            query=query  
        )
        return result 
               
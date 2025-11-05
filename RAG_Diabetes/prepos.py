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
        self.vectorstore = VectorStore(collection_name="preprocessing_document")
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
        if not os.path.isdir(self.docs_path):
            raise ValueError(f"{self.docs_path} is not a valid directory.")

        all_docs = []

        for filename in os.listdir(self.docs_path):
            file_path = os.path.join(self.docs_path, filename)

            # --- Ignore hidden/system files ---
            if filename.startswith("."):
                print(f"Skipping system file: {filename}")
                continue

            # --- Handle PDFs normally ---
            if filename.lower().endswith(".pdf"):
                print(f"Detected PDF file: {filename}")
                doc_processor = DocumentProcessor(self.docs_path, rag_type="preprocessing")
                pdf_docs = doc_processor.process_all()
                splitted_docs = self.embedding_manager.split_documents(pdf_docs)
                all_docs.extend(splitted_docs)

            # --- Handle CSVs row-by-row ---
            elif filename.lower().endswith(".csv"):
                print(f"Detected CSV file: {filename}")
                csv_docs = self._load_csv_as_documents(file_path)
                all_docs.extend(csv_docs)

            else:
                print(f"Skipping unsupported file: {filename}")

        if not all_docs:
            raise ValueError("No supported documents (.pdf or .csv) found in the directory.")

        print(f"\nLoaded total of {len(all_docs)} documents from {self.docs_path}")
        self.documents = all_docs
        return all_docs

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
        """Load, split, and embed documents once"""
        # Check if vector store already has documents
        existing_count = self.vectorstore.collection.count()
        
        if existing_count > 0:
            print(f"✓ Knowledge base already initialized with {existing_count} documents.")
            print("  Skipping document loading.")
            return
        
        # Only load and embed if vector store is empty
        split_docs = self.load_documents()
        texts = [d.page_content for d in split_docs]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vectorstore.add_docs(split_docs, embeddings)
        self.docs = split_docs
        self.embeddings = embeddings
        print("Knowledge base initialized successfully.")
        
    def recommend(self, user_data: str):
        """Main RAG pipeline"""
        # Retrieve
        retrieved_docs = self.retriever.retrieve(user_data)

        # Step 2: Augment & Generate
        result = self.generator.infer_missing_data(
            retrieved_docs=retrieved_docs,
            user_data=user_data,
        )

        return result    
               
if __name__ == "__main__":
    rag = Preprocessing()
    query = "Which patients have high blood glucose and HbA1c levels?"

    print(f"\nQuery: {query}\n")

    # Step 1: Retrieve relevant docs from vector store
    results = rag.retriever.search(query, top_k=3)

    print("\n📄 Retrieved Documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Document {i} ---")
        print(doc.page_content[:300])  # Show part of the text
        print(f"Metadata: {doc.metadata}")

    # Step 2: Generate an AI-based response using the retrieved docs
    print("\nGenerating AI response...")
    response = rag.generator.generate_response(query, results)

    print("\nFinal Answer:")
    print(response)
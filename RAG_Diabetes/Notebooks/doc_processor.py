import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter




class DocumentProcessor:
    def __init__(self, directory_path, rag_type="recommendation"):
        """
        Initialize the processor for a specific RAG type.
        
        Args:
            directory_path (str): Path to the directory containing the documents.
            rag_type (str): "recommendation" or "preprocessing"
        """
        self.directory = Path(directory_path)
        self.rag_type = rag_type.lower()
        self.all_docs = []

    def process_pdfs(self):
        """Process all PDF files in the directory."""
        pdf_files = list(self.directory.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process.")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'
                    doc.metadata['rag_type'] = self.rag_type
                self.all_docs.extend(docs)
                print(f"Loaded {len(docs)} pages.")
            except Exception as e:
                print(f"Error loading {pdf_file.name}: {e}")
    
    def process_csvs(self):
        """Process all CSV files in the directory."""
        csv_files = list(self.directory.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files to process.")
        
        for csv_file in csv_files:
            print(f"\nProcessing: {csv_file.name}")
            try:
                loader = CSVLoader(file_path=str(csv_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source_file'] = csv_file.name
                    doc.metadata['file_type'] = 'csv'
                    doc.metadata['rag_type'] = self.rag_type
                self.all_docs.extend(docs)
                print(f"Loaded {len(docs)} records.")
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")

    def process_all(self):
        """Process all supported files in the directory."""
        self.process_pdfs()
        self.process_csvs()
        print(f"\nTotal documents loaded: {len(self.all_docs)}")
        return self.all_docs
    
    def split_documents(self, chunk_size=1000, chunk_overlap=200):
        """Split loaded documents into smaller chunks."""
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

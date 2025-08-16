import os
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document

class DocumentLoader:
    """Class to handle loading various document types"""
    
    def __init__(self, documents_dir: str = "data/raw_documents"):
        """Initialize document loader with documents directory"""
        self.documents_dir = documents_dir
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
    
    def get_supported_files(self) -> List[str]:
        """Get list of supported files in the documents directory"""
        if not os.path.exists(self.documents_dir):
            print(f"Documents directory '{self.documents_dir}' does not exist")
            return []
        
        supported_files = []
        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in self.supported_extensions:
                    supported_files.append(filename)
                else:
                    print(f"Unsupported file type: {filename}")
        
        return supported_files
    
    def load_document(self, filename: str) -> List[Document]:
        """Load a single document"""
        file_path = os.path.join(self.documents_dir, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        try:
            loader_class = self.supported_extensions[file_ext]
            
            
            if file_ext == '.csv':
                loader = loader_class(file_path, encoding='utf-8')
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
            print(f"Loaded {len(documents)} pages/sections from {filename}")
            return documents
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []
    
    def load_all_documents(self) -> List[Document]:
        """Load all supported documents from the directory"""
        supported_files = self.get_supported_files()
        if not supported_files:
            print("No supported documents found")
            return []
        
        all_documents = []
        for filename in supported_files:
            documents = self.load_document(filename)
            all_documents.extend(documents)
        
        print(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def get_document_info(self, documents: List[Document]) -> Dict[str, Any]:
        """Get information about loaded documents"""
        if not documents:
            return {"total_documents": 0, "total_pages": 0, "sources": []}
        
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in documents]))
        total_pages = len(documents)
        
        return {
            "total_documents": len(sources),
            "total_pages": total_pages,
            "sources": sources,
            "average_page_length": sum(len(doc.page_content) for doc in documents) / total_pages if total_pages > 0 else 0
        }
    
    def print_document_summary(self, documents: List[Document]):
        """Print a summary of loaded documents"""
        info = self.get_document_info(documents)
        print("\nDocument Loading Summary:")
        print(f"   Total documents: {info['total_documents']}")
        print(f"   Total pages/sections: {info['total_pages']}")
        print(f"   Sources: {', '.join(info['sources'])}")
        if info['average_page_length'] > 0:
            print(f"   Average page length: {info['average_page_length']:.0f} characters")

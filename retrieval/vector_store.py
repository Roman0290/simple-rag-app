import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain.schema import Document
from utils.config_loader import get_chroma_persist_directory

class VectorStore:
    """Class to handle Chroma DB vector store operations"""
    
    def __init__(self, embedding_function, collection_name: str = "rag_documents", persist_directory: str = None):
        """Initialize vector store with Chroma DB"""
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.persist_directory = persist_directory or get_chroma_persist_directory()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store"""
        try:
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
            print(f"Chroma vector store initialized successfully")
            print(f"   Collection: {self.collection_name}")
            print(f"   Persist directory: {self.persist_directory}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        if not documents:
            print("No documents to add")
            return False
        
        try:
            print(f"Adding {len(documents)} documents to vector store...")
            
            self.vector_store.add_documents(documents)
            
            print(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> bool:
        """Add raw texts to the vector store"""
        if not texts:
            print("No texts to add")
            return False
        
        try:
            print(f"Adding {len(texts)} texts to vector store...")
           
            self.vector_store.add_texts(texts, metadatas=metadatas)
            
            
            print(f"Successfully added {len(texts)} texts to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding texts to vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            print(f"Found {len(results)} similar documents for query: '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"Found {len(results)} similar documents with scores for query: '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"Error during similarity search with score: {e}")
            return []
    
    def get_collection_info(self) -> dict:
        """Get information about the vector store collection"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_directory": self.persist_directory,
                "embedding_function": str(type(self.embedding_function).__name__)
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def print_collection_summary(self):
        """Print a summary of the vector store collection"""
        info = self.get_collection_info()
        if info:
            print("\nVector Store Summary:")
            print(f"   Collection: {info['collection_name']}")
            print(f"   Total documents: {info['total_documents']}")
            print(f"   Persist directory: {info['persist_directory']}")
            print(f"   Embedding function: {info['embedding_function']}")
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            print("Clearing vector store collection...")
            self.vector_store._collection.delete(where={})
            print("Vector store collection cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def delete_documents(self, where_clause: dict) -> bool:
        """Delete documents based on a where clause"""
        try:
            print(f"Deleting documents with criteria: {where_clause}")
            self.vector_store._collection.delete(where=where_clause)
            print("Documents deleted successfully")
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def reload(self):
        """Reload the vector store from disk"""
        try:
            print("Reloading vector store from disk...")
            self._initialize_vector_store()
            print("Vector store reloaded successfully")
        except Exception as e:
            print(f"Error reloading vector store: {e}")
            raise

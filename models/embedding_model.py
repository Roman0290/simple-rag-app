from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.config_loader import get_embedding_model_name
import os

class EmbeddingModel:
    """Class to handle embedding model operations"""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding model"""
        self.model_name = model_name or get_embedding_model_name()
        self.embeddings = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the HuggingFace embedding model"""
        try:
            # Use CPU for embeddings to avoid GPU memory issues
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"Embedding model '{self.model_name}' initialized successfully")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            # Fallback to a smaller model
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Trying fallback model: {fallback_model}")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=fallback_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print(f"Fallback embedding model initialized successfully")
            except Exception as e2:
                raise Exception(f"Failed to initialize any embedding model: {e2}")
    
    def get_embeddings(self):
        """Get the embedding model instance"""
        return self.embeddings
    
    def embed_text(self, text: str) -> list:
        """Embed a single text string"""
        if not self.embeddings:
            raise ValueError("Embedding model not initialized")
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list) -> list:
        """Embed a list of text documents"""
        if not self.embeddings:
            raise ValueError("Embedding model not initialized")
        return self.embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        if not self.embeddings:
            raise ValueError("Embedding model not initialized")
        # Test with a simple text to get dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)

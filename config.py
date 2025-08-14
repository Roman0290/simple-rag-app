#!/usr/bin/env python3
"""
Configuration file for RAG Chatbot
Centralizes all configuration settings and provides easy access to them.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

class Config:
    """Configuration class for RAG Chatbot"""
    
    def __init__(self):
        """Initialize configuration with default values"""
        self._load_environment()
        self._set_defaults()
    
    def _load_environment(self):
        """Load environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
    
    def _set_defaults(self):
        """Set default configuration values"""
        # Project paths
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.raw_documents_dir = self.data_dir / "raw_documents"
        self.processed_chunks_dir = self.data_dir / "processed_chunks"
        
        # Chroma DB settings
        self.chroma_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        
        # Embedding model settings
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.embedding_normalize = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
        
        # Document processing settings
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
        
        # Retrieval settings
        self.default_k = int(os.getenv("DEFAULT_K", "4"))
        self.max_k = int(os.getenv("MAX_K", "10"))
        self.search_type = os.getenv("SEARCH_TYPE", "similarity")
        
        # Groq LLM settings
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        
        # Streamlit app settings
        self.streamlit_port = int(os.getenv("STREAMLIT_PORT", "8501"))
        self.streamlit_host = os.getenv("STREAMLIT_HOST", "localhost")
        self.streamlit_debug = os.getenv("STREAMLIT_DEBUG", "false").lower() == "true"
        
        # Logging settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "rag_chatbot.log")
        
        # Performance settings
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self.cache_embeddings = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
        
        # Security settings
        self.enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
        
        # Feature flags
        self.enable_reranking = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
        self.enable_summarization = os.getenv("ENABLE_SUMMARIZATION", "true").lower() == "true"
        self.enable_question_generation = os.getenv("ENABLE_QUESTION_GENERATION", "true").lower() == "true"
        self.enable_fact_checking = os.getenv("ENABLE_FACT_CHECKING", "true").lower() == "true"
    
    def get_groq_api_key(self) -> str:
        """Get Groq API key"""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        return self.groq_api_key
    
    def get_chroma_persist_directory(self) -> Path:
        """Get Chroma DB persistence directory"""
        return self.chroma_persist_dir
    
    def get_documents_directory(self) -> Path:
        """Get documents directory"""
        return self.raw_documents_dir
    
    def get_embedding_model_config(self) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return {
            "model_name": self.embedding_model,
            "device": self.embedding_device,
            "normalize_embeddings": self.embedding_normalize
        }
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_chunk_size": self.max_chunk_size
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return {
            "default_k": self.default_k,
            "max_k": self.max_k,
            "search_type": self.search_type
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model_name": self.groq_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit configuration"""
        return {
            "port": self.streamlit_port,
            "host": self.streamlit_host,
            "debug": self.streamlit_debug
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "cache_embeddings": self.cache_embeddings
        }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags"""
        return {
            "reranking": self.enable_reranking,
            "summarization": self.enable_summarization,
            "question_generation": self.enable_question_generation,
            "fact_checking": self.enable_fact_checking
        }
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required settings
        if not self.groq_api_key:
            errors.append("GROQ_API_KEY is required")
        
        # Check directory paths
        if not self.raw_documents_dir.exists():
            errors.append(f"Documents directory does not exist: {self.raw_documents_dir}")
        
        # Check numeric values
        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0:
            errors.append("CHUNK_OVERLAP must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.temperature < 0 or self.temperature > 2:
            errors.append("TEMPERATURE must be between 0 and 2")
        
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 RAG Chatbot Configuration:")
        print("=" * 40)
        
        print(f"📁 Project Root: {self.project_root}")
        print(f"📁 Documents Directory: {self.raw_documents_dir}")
        print(f"🗄️ Chroma DB Directory: {self.chroma_persist_dir}")
        print(f"📚 Collection Name: {self.collection_name}")
        
        print(f"\n🤖 Embedding Model: {self.embedding_model}")
        print(f"🔧 Device: {self.embedding_device}")
        print(f"📏 Chunk Size: {self.chunk_size}")
        print(f"🔄 Chunk Overlap: {self.chunk_overlap}")
        
        print(f"\n🔍 Default K: {self.default_k}")
        print(f"🔍 Search Type: {self.search_type}")
        
        print(f"\n🚀 Groq Model: {self.groq_model}")
        print(f"🌡️ Temperature: {self.temperature}")
        print(f"📝 Max Tokens: {self.max_tokens}")
        
        print(f"\n🌐 Streamlit Port: {self.streamlit_port}")
        print(f"🌐 Streamlit Host: {self.streamlit_host}")
        
        print(f"\n⚡ Batch Size: {self.batch_size}")
        print(f"⚡ Max Workers: {self.max_workers}")
        
        print(f"\n🚩 Feature Flags:")
        flags = self.get_feature_flags()
        for feature, enabled in flags.items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {feature}")
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"🔄 Updated {key}: {value}")
            else:
                print(f"⚠️ Unknown configuration key: {key}")
    
    def save_to_env(self, env_file: str = ".env"):
        """Save current configuration to .env file"""
        env_content = f"""# RAG Chatbot Configuration
# Generated automatically

# Groq Settings
GROQ_API_KEY={self.groq_api_key or 'your_groq_api_key_here'}

# Chroma DB Settings
CHROMA_PERSIST_DIR={self.chroma_persist_dir}
CHROMA_COLLECTION_NAME={self.collection_name}

# Embedding Model Settings
EMBEDDING_MODEL={self.embedding_model}
EMBEDDING_DEVICE={self.embedding_device}
EMBEDDING_NORMALIZE={str(self.embedding_normalize).lower()}

# Document Processing Settings
CHUNK_SIZE={self.chunk_size}
CHUNK_OVERLAP={self.chunk_overlap}
MAX_CHUNK_SIZE={self.max_chunk_size}

# Retrieval Settings
DEFAULT_K={self.default_k}
MAX_K={self.max_k}
SEARCH_TYPE={self.search_type}

# Groq LLM Settings
GROQ_MODEL={self.groq_model}
TEMPERATURE={self.temperature}
MAX_TOKENS={self.max_tokens}

# Streamlit Settings
STREAMLIT_PORT={self.streamlit_port}
STREAMLIT_HOST={self.streamlit_host}
STREAMLIT_DEBUG={str(self.streamlit_debug).lower()}

# Logging Settings
LOG_LEVEL={self.log_level}
LOG_FILE={self.log_file}

# Performance Settings
BATCH_SIZE={self.batch_size}
MAX_WORKERS={self.max_workers}
CACHE_EMBEDDINGS={str(self.cache_embeddings).lower()}

# Security Settings
ENABLE_RATE_LIMITING={str(self.enable_rate_limiting).lower()}
MAX_REQUESTS_PER_MINUTE={self.max_requests_per_minute}

# Feature Flags
ENABLE_RERANKING={str(self.enable_reranking).lower()}
ENABLE_SUMMARIZATION={str(self.enable_summarization).lower()}
ENABLE_QUESTION_GENERATION={str(self.enable_question_generation).lower()}
ENABLE_FACT_CHECKING={str(self.enable_fact_checking).lower()}
"""
        
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print(f"✅ Configuration saved to {env_file}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()

def get_env_variable(key: str, default: str = None) -> str:
    """Get environment variable value"""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set")
    return value

def get_groq_api_key() -> str:
    """Get Groq API key from environment variables"""
    return get_env_variable("GROQ_API_KEY")

def get_chroma_persist_directory() -> str:
    """Get Chroma DB persistence directory"""
    return os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def get_embedding_model_name() -> str:
    """Get embedding model name"""
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def get_chunk_size() -> int:
    """Get chunk size for document splitting"""
    return int(os.getenv("CHUNK_SIZE", "1000"))

def get_chunk_overlap() -> int:
    """Get chunk overlap for document splitting"""
    return int(os.getenv("CHUNK_OVERLAP", "200"))

def get_groq_model_name() -> str:
    """Get Groq model name"""
    return os.getenv("GROQ_MODEL", "llama3-70b-8192")

def get_temperature() -> float:
    """Get temperature for LLM generation"""
    return float(os.getenv("TEMPERATURE", "0.7"))

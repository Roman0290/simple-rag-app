from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.config_loader import get_chunk_size, get_chunk_overlap

class ChunkingStrategy:
    """Class to handle document chunking strategies"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize chunking strategy"""
        self.chunk_size = chunk_size or get_chunk_size()
        self.chunk_overlap = chunk_overlap or get_chunk_overlap()
        self.text_splitter = self._create_text_splitter()
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create the text splitter with configured parameters"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            print("⚠️ No documents to chunk")
            return []
        
        print(f"Chunking {len(documents)} documents...")
        print(f"   Chunk size: {self.chunk_size}")
        print(f"   Chunk overlap: {self.chunk_overlap}")
        
        try:
            # Split all documents
            all_chunks = []
            for i, doc in enumerate(documents):
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
                print(f"   Document {i+1}: {len(chunks)} chunks")
            
            print(f"Total chunks created: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            print(f"❌ Error during chunking: {e}")
            return []
    
    def chunk_single_document(self, document: Document) -> List[Document]:
        """Split a single document into chunks"""
        try:
            chunks = self.text_splitter.split_documents([document])
            print(f"Created {len(chunks)} chunks from document")
            return chunks
        except Exception as e:
            print(f"❌ Error chunking document: {e}")
            return []
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """Get statistics about the created chunks"""
        if not chunks:
            return {"total_chunks": 0, "avg_chunk_length": 0, "min_length": 0, "max_length": 0}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    def print_chunk_summary(self, chunks: List[Document]):
        """Print a summary of the chunking process"""
        stats = self.get_chunk_statistics(chunks)
        print("\nChunking Summary:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Average chunk length: {stats['avg_chunk_length']:.0f} characters")
        print(f"   Min chunk length: {stats['min_length']} characters")
        print(f"   Max chunk length: {stats['max_length']} characters")
        print(f"   Chunk size setting: {stats['chunk_size']}")
        print(f"   Chunk overlap setting: {stats['chunk_overlap']}")
    
    def update_chunk_parameters(self, chunk_size: int, chunk_overlap: int):
        """Update chunking parameters and recreate text splitter"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = self._create_text_splitter()
        print(f"Updated chunking parameters: size={chunk_size}, overlap={chunk_overlap}")

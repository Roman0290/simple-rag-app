#!/usr/bin/env python3
"""
Main script for RAG Chatbot document ingestion and indexing
"""

import os
import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_loader import load_environment
from models.embedding_model import EmbeddingModel
from retrieval.document_loader import DocumentLoader
from retrieval.chunking_strategy import ChunkingStrategy
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from generation.llm_chain import LLMChain

class RAGSystem:
    """Main class to orchestrate the RAG system"""
    
    def __init__(self):
        """Initialize the RAG system"""
        load_environment()
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.llm_chain = None
        self.documents = []
        self.chunks = []
    
    def initialize_components(self):
        """Initialize all system components"""
        print("Initializing RAG System Components...")
        
        try:
            # Initialize embedding model
            print("\n1. Initializing embedding model...")
            self.embedding_model = EmbeddingModel()
            
            # Initialize vector store
            print("\n2. Initializing vector store...")
            self.vector_store = VectorStore(self.embedding_model.get_embeddings())
            
            # Initialize retriever
            print("\n3. Initializing retriever...")
            self.retriever = Retriever(self.vector_store)
            
            # Initialize LLM chain
            print("\n4. Initializing LLM chain...")
            self.llm_chain = LLMChain()
            
            print("\nAll components initialized successfully!")
            
        except Exception as e:
            print(f"\n❌ Error initializing components: {e}")
            raise
    
    def load_documents(self, documents_dir: str = "data/raw_documents"):
        """Load documents from the specified directory"""
        print(f"\nLoading documents from: {documents_dir}")
        
        try:
            document_loader = DocumentLoader(documents_dir)
            self.documents = document_loader.load_all_documents()
            
            if not self.documents:
                print("⚠️ No documents loaded. Please check the documents directory.")
                return False
            
            document_loader.print_document_summary(self.documents)
            return True
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return False
    
    def chunk_documents(self, chunk_size: int = None, chunk_overlap: int = None):
        """Chunk the loaded documents"""
        print(f"\nChunking documents...")
        
        try:
            chunking_strategy = ChunkingStrategy(chunk_size, chunk_overlap)
            self.chunks = chunking_strategy.chunk_documents(self.documents)
            
            if not self.chunks:
                print("⚠️ No chunks created from documents.")
                return False
            
            chunking_strategy.print_chunk_summary(self.chunks)
            return True
            
        except Exception as e:
            print(f"❌ Error chunking documents: {e}")
            return False
    
    def index_documents(self):
        """Index the chunked documents in the vector store"""
        print(f"\nIndexing documents in vector store...")
        
        try:
            if not self.chunks:
                print("⚠️ No chunks to index. Please load and chunk documents first.")
                return False
            
            success = self.vector_store.add_documents(self.chunks)
            
            if success:
                self.vector_store.print_collection_summary()
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error indexing documents: {e}")
            return False
    
    def test_retrieval(self, test_query: str = "What is this document about?"):
        """Test the retrieval system"""
        print(f"\nTesting retrieval with query: '{test_query}'")
        
        try:
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve_documents(test_query)
            
            if not retrieved_docs:
                print("⚠️ No documents retrieved. Please check the vector store.")
                return False
            
            # Print retrieval summary
            self.retriever.print_retrieval_summary(test_query)
            
            # Print document previews
            self.retriever.print_document_previews(retrieved_docs)
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing retrieval: {e}")
            return False
    
    def test_generation(self, test_query: str = "What is this document about?"):
        """Test the generation system"""
        print(f"\nTesting generation with query: '{test_query}'")
        
        try:
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve_documents(test_query)
            
            if not retrieved_docs:
                print("⚠️ No documents retrieved for generation test.")
                return False
            
            # Combine context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Generate answer
            answer = self.llm_chain.generate_answer(test_query, context)
            
            print(f"\nGenerated Answer:")
            print(f"   {answer}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing generation: {e}")
            return False
    
    def run_full_pipeline(self, documents_dir: str = "data/raw_documents", 
                         chunk_size: int = None, chunk_overlap: int = None):
        """Run the complete RAG pipeline"""
        print("Starting RAG Pipeline...")
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Load documents
            if not self.load_documents(documents_dir):
                return False
            
            # Chunk documents
            if not self.chunk_documents(chunk_size, chunk_overlap):
                return False
            
            # Index documents
            if not self.index_documents():
                return False
            
            # Test retrieval
            if not self.test_retrieval():
                return False
            
            # Test generation
            if not self.test_generation():
                return False
            
            print("\nRAG Pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ RAG Pipeline failed: {e}")
            return False
    
    def interactive_mode(self):
        """Run the system in interactive mode"""
        print("\n🎮 Interactive Mode - RAG Chatbot")
        print("Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() in ['help', 'h']:
                    self._print_help()
                elif user_input.lower() in ['status', 's']:
                    self._print_status()
                elif user_input.lower() in ['test', 't']:
                    self._run_tests()
                elif user_input:
                    self._process_user_query(user_input)
                else:
                    continue
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _print_help(self):
        """Print help information"""
        print("\n📖 Available Commands:")
        print("   help, h     - Show this help message")
        print("   status, s   - Show system status")
        print("   test, t     - Run system tests")
        print("   quit, exit, q - Exit the program")
        print("   <any text>  - Ask a question to the RAG system")
    
    def _print_status(self):
        """Print system status"""
        print("\n📊 System Status:")
        print(f"   Documents loaded: {len(self.documents)}")
        print(f"   Chunks created: {len(self.chunks)}")
        
        if self.vector_store:
            self.vector_store.print_collection_summary()
        
        if self.llm_chain:
            self.llm_chain.print_llm_summary()
    
    def _run_tests(self):
        """Run system tests"""
        print("\n🧪 Running System Tests...")
        
        # Test retrieval
        self.test_retrieval()
        
        # Test generation
        self.test_generation()
        
        # Test LLM connection
        if self.llm_chain:
            self.llm_chain.test_llm_connection()
    
    def _process_user_query(self, query: str):
        """Process a user query"""
        print(f"\nProcessing query: '{query}'")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve_documents(query)
            
            if not retrieved_docs:
                print("❌ No relevant documents found for your query.")
                return
            
            # Combine context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Generate answer
            answer = self.llm_chain.generate_answer(query, context)
            
            print(f"\nAnswer:")
            print(f"   {answer}")
            
            # Show sources
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
            print(f"\nSources: {', '.join(sources)}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG Chatbot Document Ingestion and Indexing")
    parser.add_argument("--ingest", action="store_true", help="Run full document ingestion pipeline")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--documents-dir", default="data/raw_documents", help="Documents directory path")
    parser.add_argument("--chunk-size", type=int, help="Chunk size for document splitting")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap for document splitting")
    parser.add_argument("--test", action="store_true", help="Run system tests")
    
    args = parser.parse_args()
    
    # Create RAG system
    rag_system = RAGSystem()
    
    try:
        if args.ingest:
            # Run full pipeline
            success = rag_system.run_full_pipeline(
                documents_dir=args.documents_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            if not success:
                sys.exit(1)
                
        elif args.interactive:
            # Initialize components first
            rag_system.initialize_components()
            
            # Check if documents are already indexed
            if rag_system.vector_store and rag_system.vector_store.get_collection_info().get('total_documents', 0) == 0:
                print("⚠️ No documents indexed. Running ingestion first...")
                if not rag_system.run_full_pipeline(args.documents_dir, args.chunk_size, args.chunk_overlap):
                    print("❌ Failed to index documents. Cannot start interactive mode.")
                    sys.exit(1)
            
            # Start interactive mode
            rag_system.interactive_mode()
            
        elif args.test:
            # Run tests
            rag_system.initialize_components()
            rag_system._run_tests()
            
        else:
            # Default: run full pipeline
            print("Running default RAG pipeline...")
            success = rag_system.run_full_pipeline(
                documents_dir=args.documents_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            if not success:
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n👋 Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from .vector_store import VectorStore
from .reranker import Reranker

class Retriever:
    """Class to handle document retrieval from vector store, with optional reranking"""
    def __init__(self, vector_store: VectorStore, k: int = 4, search_type: str = "similarity", use_reranker: bool = False, reranker_top_k: int = None):
        """Initialize retriever with vector store and optional reranker"""
        self.vector_store = vector_store
        self.k = k
        self.search_type = search_type
        self.retriever = self._create_retriever()
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self.reranker = Reranker() if use_reranker else None
    
    def _create_retriever(self) -> BaseRetriever:
        """Create a LangChain retriever from the vector store"""
        try:
            retriever = self.vector_store.vector_store.as_retriever(
                search_type=self.search_type,
                search_kwargs={"k": self.k}
            )
            print(f"Retriever initialized with k={self.k}, search_type={self.search_type}")
            return retriever
        except Exception as e:
            print(f"Error creating retriever: {e}")
            raise
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query, with optional reranking"""
        try:
            print(f"Retrieving documents for query: '{query[:50]}...'")
            documents = self.retriever.get_relevant_documents(query)
            print(f"Retrieved {len(documents)} relevant documents before reranking")
            if self.use_reranker and self.reranker:
                print("Applying reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
                documents = self.reranker.rerank(query, documents, top_k=self.reranker_top_k or self.k)
                print(f"Documents reranked. Returning top {len(documents)}.")
            return documents
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """Retrieve documents with similarity scores"""
        try:
            print(f"Retrieving documents with scores for query: '{query[:50]}...'")
            
            
            results = self.vector_store.similarity_search_with_score(query, k=self.k)
            
            print(f"Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            print(f"Error during retrieval with scores: {e}")
            return []
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """Get statistics about the retrieval process"""
        try:
            documents = self.retrieve_documents(query)
            
            if not documents:
                return {
                    "query": query,
                    "documents_retrieved": 0,
                    "total_content_length": 0,
                    "average_document_length": 0,
                    "sources": []
                }
            
            total_length = sum(len(doc.page_content) for doc in documents)
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in documents]))
            
            return {
                "query": query,
                "documents_retrieved": len(documents),
                "total_content_length": total_length,
                "average_document_length": total_length / len(documents),
                "sources": sources,
                "retrieval_k": self.k,
                "search_type": self.search_type
            }
            
        except Exception as e:
            print(f"Error getting retrieval stats: {e}")
            return {}
    
    def print_retrieval_summary(self, query: str):
        """Print a summary of the retrieval process"""
        stats = self.get_retrieval_stats(query)
        if stats:
            print("\nRetrieval Summary:")
            print(f"   Query: {stats['query'][:100]}...")
            print(f"   Documents retrieved: {stats['documents_retrieved']}")
            print(f"   Total content length: {stats['total_content_length']} characters")
            print(f"   Average document length: {stats['average_document_length']:.0f} characters")
            print(f"   Sources: {', '.join(stats['sources'])}")
            print(f"   Retrieval k: {stats['retrieval_k']}")
            print(f"   Search type: {stats['search_type']}")
    
    def update_retrieval_parameters(self, k: int = None, search_type: str = None, use_reranker: bool = None, reranker_top_k: int = None):
        """Update retrieval and reranker parameters"""
        if k is not None:
            self.k = k
        if search_type is not None:
            self.search_type = search_type
        if use_reranker is not None:
            self.use_reranker = use_reranker
            self.reranker = Reranker() if use_reranker else None
        if reranker_top_k is not None:
            self.reranker_top_k = reranker_top_k
        
        self.retriever = self._create_retriever()
        print(f"Updated retrieval parameters: k={self.k}, search_type={self.search_type}, use_reranker={self.use_reranker}, reranker_top_k={self.reranker_top_k}")
    
    def get_document_preview(self, documents: List[Document], max_chars: int = 200) -> List[str]:
        """Get preview of retrieved documents"""
        previews = []
        for i, doc in enumerate(documents):
            content = doc.page_content[:max_chars]
            if len(doc.page_content) > max_chars:
                content += "..."
            
            source = doc.metadata.get('source', 'Unknown')
            previews.append(f"Document {i+1} ({source}): {content}")
        
        return previews
    
    def print_document_previews(self, documents: List[Document], max_chars: int = 200):
        """Print previews of retrieved documents"""
        previews = self.get_document_preview(documents, max_chars)
        print(f"\nRetrieved Document Previews (max {max_chars} chars):")
        for preview in previews:
            print(f"   {preview}")
    
    def filter_documents_by_source(self, documents: List[Document], source_filter: str) -> List[Document]:
        """Filter retrieved documents by source"""
        filtered = [doc for doc in documents if source_filter.lower() in doc.metadata.get('source', '').lower()]
        print(f"Filtered {len(filtered)} documents by source filter: '{source_filter}'")
        return filtered
    
    def get_retrieval_quality_score(self, query: str, documents: List[Document]) -> float:
        """Calculate a simple quality score for retrieval (placeholder for future enhancement)"""
        if not documents:
            return 0.0
        
      
        avg_length = sum(len(doc.page_content) for doc in documents) / len(documents)
        
      
        score = min(avg_length / 1000, 1.0)  # Cap at 1.0
        
        return score

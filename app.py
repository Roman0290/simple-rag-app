#!/usr/bin/env python3
"""
Streamlit web application for RAG Chatbot
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_loader import load_environment
from models.embedding_model import EmbeddingModel
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from generation.llm_chain import LLMChain

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    .status-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

class RAGChatbotApp:
    """Streamlit app for RAG Chatbot"""
    
    def __init__(self):
        """Initialize the app"""
        load_environment()
        self.initialize_session_state()
        self.initialize_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
    
    def initialize_components(self):
        """Initialize RAG system components"""
        try:
            with st.spinner("Initializing RAG system components..."):
                # Initialize embedding model
                self.embedding_model = EmbeddingModel()
                
                # Initialize vector store
                self.vector_store = VectorStore(self.embedding_model.get_embeddings())
                
                # Initialize retriever
                self.retriever = Retriever(self.vector_store)
                
                # Initialize LLM chain
                self.llm_chain = LLMChain()
                
                st.session_state.system_initialized = True
                st.success("✅ RAG system initialized successfully!")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            st.session_state.system_initialized = False
    
    def check_system_status(self):
        """Check and display system status"""
        if not st.session_state.system_initialized:
            st.error("❌ RAG system not initialized. Please check the configuration.")
            return False
        
        # Check vector store status
        try:
            collection_info = self.vector_store.get_collection_info()
            total_docs = collection_info.get('total_documents', 0)
            
            if total_docs == 0:
                st.warning("⚠️ No documents indexed. Please run the ingestion process first.")
                st.session_state.documents_loaded = False
                return False
            else:
                st.session_state.documents_loaded = True
                return True
                
        except Exception as e:
            st.error(f"❌ Error checking system status: {str(e)}")
            return False
    
    def display_system_info(self):
        """Display system information in sidebar"""
        st.sidebar.header("🤖 System Information")
        
        if st.session_state.system_initialized:
            # Display embedding model info
            st.sidebar.subheader("Embedding Model")
            st.sidebar.write(f"Model: {self.embedding_model.model_name}")
            st.sidebar.write(f"Dimension: {self.embedding_model.get_embedding_dimension()}")
            
            # Display vector store info
            st.sidebar.subheader("Vector Store")
            collection_info = self.vector_store.get_collection_info()
            st.sidebar.write(f"Collection: {collection_info.get('collection_name', 'N/A')}")
            st.sidebar.write(f"Documents: {collection_info.get('total_documents', 0)}")
            
            # Display LLM info
            st.sidebar.subheader("LLM")
            llm_info = self.llm_chain.get_llm_info()
            st.sidebar.write(f"Provider: {llm_info.get('provider', 'N/A')}")
            st.sidebar.write(f"Model: {llm_info.get('model_name', 'N/A')}")
            st.sidebar.write(f"Temperature: {llm_info.get('temperature', 'N/A')}")
        else:
            st.sidebar.error("System not initialized")
    
    def display_chat_interface(self):
        """Display the main chat interface"""
        st.markdown('<div class="main-header">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources for assistant messages
                if message["role"] == "assistant" and message.get("sources"):
                    st.markdown("**Sources:**")
                    for source in message["sources"]:
                        st.markdown(f'<div class="source-box">📚 {source}</div>', unsafe_allow_html=True)
    
    def process_user_query(self, user_query: str):
        """Process a user query and generate response"""
        if not st.session_state.documents_loaded:
            st.error("❌ No documents loaded. Please run the ingestion process first.")
            return
        
        try:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Show assistant is thinking
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching for relevant information..."):
                    # Retrieve relevant documents
                    retrieved_docs = self.retriever.retrieve_documents(user_query)
                    
                    if not retrieved_docs:
                        response = "❌ No relevant documents found for your query. Please try rephrasing or check if documents are properly indexed."
                        sources = []
                    else:
                        # Combine context from retrieved documents
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        
                        # Generate answer
                        response = self.llm_chain.generate_answer(user_query, context)
                        
                        # Get sources
                        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
                
                # Display response
                st.markdown(response)
                
                # Display sources
                if sources:
                    st.markdown("**Sources:**")
                    for source in sources:
                        st.markdown(f'<div class="source-box">📚 {source}</div>', unsafe_allow_html=True)
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
    
    def display_sidebar_controls(self):
        """Display sidebar controls"""
        st.sidebar.header("🎛️ Controls")
        
        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat", type="primary"):
            st.session_state.messages = []
            st.rerun()
        
        # System test button
        if st.sidebar.button("🧪 Test System"):
            self.run_system_tests()
        
        # Settings
        st.sidebar.header("⚙️ Settings")
        
        # Retrieval settings
        st.sidebar.subheader("Retrieval")
        k_value = st.sidebar.slider("Number of documents to retrieve (k)", 1, 10, 4)
        if st.sidebar.button("Update Retrieval Settings"):
            self.retriever.update_retrieval_parameters(k=k_value)
            st.sidebar.success("Retrieval settings updated!")
        
        # LLM settings
        st.sidebar.subheader("LLM")
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        if st.sidebar.button("Update LLM Settings"):
            self.llm_chain.update_llm_parameters(temperature=temperature)
            st.sidebar.success("LLM settings updated!")
    
    def run_system_tests(self):
        """Run system tests"""
        st.sidebar.header("🧪 System Tests")
        
        # Test retrieval
        if st.sidebar.button("Test Retrieval"):
            with st.spinner("Testing retrieval..."):
                test_query = "What is this document about?"
                retrieved_docs = self.retriever.retrieve_documents(test_query)
                if retrieved_docs:
                    st.sidebar.success(f"✅ Retrieval test passed: {len(retrieved_docs)} documents")
                else:
                    st.sidebar.error("❌ Retrieval test failed")
        
        # Test LLM
        if st.sidebar.button("Test LLM"):
            with st.spinner("Testing LLM..."):
                if self.llm_chain.test_llm_connection():
                    st.sidebar.success("✅ LLM test passed")
                else:
                    st.sidebar.error("❌ LLM test failed")
    
    def display_help(self):
        """Display help information"""
        st.sidebar.header("❓ Help")
        
        with st.sidebar.expander("How to use"):
            st.markdown("""
            1. **Ask Questions**: Type your question in the chat input
            2. **View Sources**: Each response shows the source documents
            3. **Adjust Settings**: Use sidebar controls to customize behavior
            4. **Clear Chat**: Use the clear button to start fresh
            """)
        
        with st.sidebar.expander("Example Questions"):
            st.markdown("""
            - "What is this document about?"
            - "Can you summarize the main points?"
            - "What are the key findings?"
            - "Explain the methodology used"
            """)
    
    def run(self):
        """Run the Streamlit app"""
        # Display system status
        if not self.check_system_status():
            st.error("❌ System not ready. Please check the configuration and ensure documents are indexed.")
            return
        
        # Display sidebar
        self.display_system_info()
        self.display_sidebar_controls()
        self.display_help()
        
        # Display main interface
        self.display_chat_interface()
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            self.process_user_query(prompt)

def main():
    """Main function"""
    try:
        app = RAGChatbotApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Fatal error: {str(e)}")
        st.error("Please check your configuration and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()

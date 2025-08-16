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
    page_icon="",
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
        background: linear-gradient(90deg, #f8fafc 60%, #e0e7ef 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-top: 0.75rem;
        font-size: 1rem;
        font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace;
        color: #2d3748;
        border: 1.5px solid #b3c6e0;
        box-shadow: 0 2px 8px 0 rgba(44, 62, 80, 0.07);
        transition: box-shadow 0.2s;
        font-weight: 500;
        letter-spacing: 0.01em;
        display: inline-block;
        max-width: 100%;
        word-break: break-all;
    }
    .source-box:hover {
        box-shadow: 0 4px 16px 0 rgba(44, 62, 80, 0.15);
        border-color: #2196f3;
        background: linear-gradient(90deg, #e3f2fd 60%, #bbdefb 100%);
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
             
                self.embedding_model = EmbeddingModel()
                
                self.vector_store = VectorStore(self.embedding_model.get_embeddings())
                
                self.retriever = Retriever(self.vector_store)
                
                self.llm_chain = LLMChain()
                
                st.session_state.system_initialized = True
                st.success("RAG system initialized successfully!")
                
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            st.session_state.system_initialized = False
    
    def check_system_status(self):
        """Check and display system status"""
        if not st.session_state.system_initialized:
            st.error("RAG system not initialized. Please check the configuration.")
            return False
        try:
            collection_info = self.vector_store.get_collection_info()
            total_docs = collection_info.get('total_documents', 0)
            
            if total_docs == 0:
                st.warning("No documents indexed. Please run the ingestion process first.")
                st.session_state.documents_loaded = False
                return False
            else:
                st.session_state.documents_loaded = True
                return True
                
        except Exception as e:
            st.error(f"Error checking system status: {str(e)}")
            return False
    
    def display_system_info(self):
        """Display a friendly welcome message in the sidebar"""
        st.sidebar.header(":wave: Welcome to DocuBuddy!")
        st.sidebar.markdown("""
            <div style='font-size:1.1rem;line-height:1.6;'>
            Let's talk about your documents.<br>
            <ul>
                <li>Upload PDF or TXT files using the sidebar below.</li>
                <li>Click <b>Ingest Uploaded Files</b> to index them.</li>
                <li>Ask questions in the chat!</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
    
    def display_chat_interface(self):
        """Display the main chat interface"""
        st.markdown('<div class="main-header">DocuBuddy 📄🤝</div>', unsafe_allow_html=True)
       
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and message.get("sources"):
                    st.markdown("<span style='font-size:1.1rem;font-weight:600;color:#2196f3;'>Sources:</span>", unsafe_allow_html=True)
                    for source in message["sources"]:
                        st.markdown(f'<div class="source-box"><span style="color:#1565c0;">📄</span> <span style="color:#374151;">{source}</span></div>', unsafe_allow_html=True)
    
    def process_user_query(self, user_query: str):
        """Process a user query and generate response"""
        if not st.session_state.documents_loaded:
            st.error("No documents loaded. Please run the ingestion process first.")
            return

        try:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching for relevant information..."):
                    uploaded_filenames = st.session_state.get("uploaded_filenames", None)
                    if uploaded_filenames:
                        from retrieval.document_loader import DocumentLoader
                        from retrieval.chunking_strategy import ChunkingStrategy
                        doc_loader = DocumentLoader("data/raw_documents")
                        all_chunks = []
                        for fname in uploaded_filenames:
                            documents = doc_loader.load_document(fname)
                            chunker = ChunkingStrategy()
                            chunks = chunker.chunk_documents(documents)
                            all_chunks.extend(chunks)
                        
                        self.vector_store.add_documents(all_chunks)
                     
                        retrieved_docs = []
                        for chunk in all_chunks:
                            if user_query.lower() in chunk.page_content.lower():
                                retrieved_docs.append(chunk)
                        
                        if not retrieved_docs:
                            retrieved_docs = self.retriever.retrieve_documents(user_query)
                    else:
                    
                        retrieved_docs = self.retriever.retrieve_documents(user_query)

                    if not retrieved_docs:
                        response = "No relevant documents found for your query. Please try rephrasing or check if documents are properly indexed."
                        sources = []
                    else:
                   
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        
                        response = self.llm_chain.generate_answer(user_query, context)
                   
                        sources = list({os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in retrieved_docs})

                st.markdown(response)

              
                if sources:
                    st.markdown("<span style='font-size:1.1rem;font-weight:600;color:#2196f3;'>Sources:</span>", unsafe_allow_html=True)
                    for source in sources:
                        st.markdown(f'<div class="source-box"><span style="color:#1565c0;">📄</span> <span style="color:#374151;">{source}</span></div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    
    def display_sidebar_controls(self):
        """Display sidebar controls"""
        st.sidebar.header("Controls")

        
        uploaded_files = st.sidebar.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)
        upload_success = False
        uploaded_filenames = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = Path("data/raw_documents") / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_filenames.append(uploaded_file.name)
            st.sidebar.success(f"Uploaded and saved: {', '.join(uploaded_filenames)}")
            upload_success = True
           
            st.session_state["uploaded_filenames"] = uploaded_filenames

        if upload_success and st.sidebar.button("Ingest Uploaded Files"):
            from retrieval.document_loader import DocumentLoader
            from retrieval.chunking_strategy import ChunkingStrategy
            doc_loader = DocumentLoader("data/raw_documents")
            all_chunks = []
            for fname in uploaded_filenames:
                documents = doc_loader.load_document(fname)
                chunker = ChunkingStrategy()
                chunks = chunker.chunk_documents(documents)
                all_chunks.extend(chunks)
            success = self.vector_store.add_documents(all_chunks)
            if success:
                st.sidebar.success("Documents ingested and indexed!")
                st.session_state.documents_loaded = True
            else:
                st.sidebar.error("Failed to index the documents.")

        
        if st.sidebar.button("🗑️ Clear Chat", type="primary"):
            st.session_state.messages = []
            st.rerun()

        
        st.sidebar.header("Settings")

       
        st.sidebar.subheader("Retrieval")
        k_value = st.sidebar.slider("Documents to retrieve (k)", 1, 10, 4)
        if st.sidebar.button("Update Retrieval"):
            self.retriever.update_retrieval_parameters(k=k_value)
            st.sidebar.success("Updated!")

      
        st.sidebar.subheader("LLM")
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        if st.sidebar.button("Update LLM"):
            self.llm_chain.update_llm_parameters(temperature=temperature)
            st.sidebar.success("Updated!")
    
    def run_system_tests(self):
        """Run system tests"""
        st.sidebar.header("System Tests")
        
        if st.sidebar.button("Test Retrieval"):
            with st.spinner("Testing..."):
                test_query = "What is this document about?"
                retrieved_docs = self.retriever.retrieve_documents(test_query)
                if retrieved_docs:
                    st.sidebar.success(f"{len(retrieved_docs)} documents retrieved")
                else:
                    st.sidebar.error("Failed")
        
      
        if st.sidebar.button("Test LLM"):
            with st.spinner("Testing..."):
                if self.llm_chain.test_llm_connection():
                    st.sidebar.success("LLM OK")
                else:
                    st.sidebar.error("LLM Failed")
    
    def display_help(self):
        """Display help information"""
        st.sidebar.header("❓ Help")
        
        with st.sidebar.expander("How to use"):
            st.markdown("""
            1. **Ask Questions**: Type your question in the chat
            2. **View Sources**: Each response shows source documents
            3. **Adjust Settings**: Use sidebar to customize behavior
            4. **Clear Chat**: Use clear button to start fresh
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
        if not self.check_system_status():
            st.error("System not ready. Please check the configuration and ensure documents are indexed.")
            return
        
        self.display_system_info()
        self.display_sidebar_controls()
        self.display_help()
    
        self.display_chat_interface()
    
        if prompt := st.chat_input("Ask me anything about your documents..."):
            self.process_user_query(prompt)

def main():
    """Main function"""
    try:
        app = RAGChatbotApp()
        app.run()
    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
        st.error("Please check your configuration and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()

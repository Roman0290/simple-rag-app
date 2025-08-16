
# Simple RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your own documents using Chroma DB and Groq LLM. Upload, ingest, and chat with your PDFs and text files—all in a modern Streamlit web app.


## Features

- **Multi-file Upload**: Upload and ingest multiple PDF or TXT files at once
- **Document Support**: PDF, TXT (CSV, MD, DOCX supported in backend)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: Chroma DB with persistence
- **LLM**: Groq API (llama3-70b-8192)
- **Modern UI**: Streamlit web app with attractive chat and source display
- **Chunking**: Configurable size and overlap
- **Source Attribution**: Answers show which files were used (by filename)

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/Roman0290/simple-rag-app.git
   cd simple-rag-app
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   # Or: source .venv/bin/activate  # On Linux/Mac
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Create .env file with your Groq API key
   echo GROQ_API_KEY=your_api_key_here > .env
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Upload & Ingest Documents**
   - Use the sidebar to upload one or more PDF/TXT files
   - Click "Ingest Uploaded Files" to index them for retrieval

5. **Chat**
   - Ask questions about your documents in the chat interface
   - Answers will cite the filenames used as sources

## Configuration

Key settings in `.env`:
- `GROQ_API_KEY`: Your Groq API key
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap (default: 200)
- `GROQ_MODEL`: LLM model (default: llama3-70b-8192)


## Project Structure

```
rag_chatbot/
├── data/raw_documents/     # Uploaded documents
├── models/                 # Embedding models
├── retrieval/              # Document loading, chunking, vector search
├── generation/             # LLM chains & prompts
├── utils/                  # Configuration
├── app.py                  # Streamlit web app
└── requirements.txt        # Dependencies
```


## Requirements

- Python 3.8+
- Groq API key
- See `requirements.txt` for packages

## Notes

- Uploaded and processed files in `data/` are ignored by git (see `.gitignore`).
- Only PDF and TXT files are supported in the UI upload; backend supports more types.
- Sources in answers are shown by filename for clarity.


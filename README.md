# Hybrid PDF RAG System

A professional, high-performance **Retrieval-Augmented Generation (RAG)** system built to load, chunk, index, and query multiple PDF documents. This project implements a **hybrid retrieval strategy** combining lexical search (BM25) and semantic vector search (FAISS) fused together via Reciprocal Rank Fusion (RRF) to provide highly relevant context to an LLM.

---

## 🚀 Key Features

* **Multi-PDF Parsing:** Extracts structured text page-by-page from PDFs using `PyMuPDF`.
* **Smart Chunking:** Uses LangChain's `RecursiveCharacterTextSplitter` to create overlapping text segments for contextual integrity.
* **Semantic Embeddings:** Uses the high-performance local `BAAI/bge-small-en-v1.5` model via `sentence-transformers`.
* **Hybrid Search:** Combines BM25 lexical search (via `rank-bm25`) and dense vector search (via `FAISS`) to ensure both keyword accuracy and semantic matching.
* **Reciprocal Rank Fusion (RRF):** Fuses sparse and dense search results using rank-based scores to yield optimal context chunks.
* **Flexible LLM Integration:** Integrates with local OpenAI-compatible APIs (like LM Studio or Ollama) or cloud services (like OpenAI, DeepSeek, or Anthropic) using environment variables.

---

## 📁 Project Structure

```text
├── data/                  # Drop your PDF files here (ignored by git)
│   └── .gitkeep           # Preserves data directory structure
├── storage/               # Saved FAISS index and chunk serialization files
│   └── .gitkeep           # Preserves storage directory structure
├── loaders/
│   └── pdf_loader.py      # Extracting text from PDF files
├── utils/
│   ├── chunker.py         # Splitting text into overlapping chunks
│   └── embedder.py        # SentenceTransformers local embedding generator
├── retrievers/
│   ├── retriever.py       # Basic cosine similarity retriever
│   ├── bm25_retriever.py  # BM25 lexical search wrapper
│   ├── vector_store.py    # FAISS dense vector database wrapper
│   └── hybrid_retriever.py# Hybrid retriever with Reciprocal Rank Fusion
├── llm/
│   └── llm.py             # OpenAI API communication helper
├── tests/
│   ├── test-llm.py        # LLM integration testing script
│   └── test_bm25.py       # BM25 retriever testing script
├── app.py                 # Main interactive CLI application
├── build_index.py         # Standalone script to pre-build & save vector index
├── requirements.txt       # Project python dependencies
└── .gitignore             # Configured Git exclusions
```

---

## 🛠️ Setup and Installation

### 1. Prerequisites
Make sure you have Python 3.9+ installed on your system.

### 2. Clone and Setup Environment
Navigate to your project directory and initialize a virtual environment:
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install required dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root of the project to securely manage endpoints and API keys without committing them to git.

```ini
# --- LLM Config ---
# For LM Studio (Default Local Development):
OPENAI_API_BASE=http://127.0.0.1:1234/v1/
OPENAI_API_KEY=lm-studio
OPENAI_API_MODEL=nvidia/nemotron-3-nano-4b

# For OpenAI (Production Deployment):
# OPENAI_API_BASE=https://api.openai.com/v1
# OPENAI_API_KEY=your-actual-api-key-here
# OPENAI_API_MODEL=gpt-4o
```

---

## 🏃 How to Run

### Step 1: Add PDF Files
Put your PDF files (e.g., recipe books, manuals, research papers) inside the `data/` directory.

### Step 2: Build the Search Index
Run the indexing script to build and save the FAISS vector database:
```bash
python build_index.py
```
This processes all PDFs, generates chunks, computes semantic vectors, and writes index data securely inside `storage/`.

### Step 3: Launch the Interactive App
Run the main RAG application to ask questions:
```bash
python app.py
```
* The app automatically loads the PDFs, builds the BM25 lookup + FAISS index, and initializes the interactive command line prompt.
* Type `exit` to close the prompt.

---

## 🛡️ Security Best Practices

To safeguard your data and credentials before publishing this project on platforms like GitHub, the repository has been configured with the following safeguards:

1. **Environment Separation (.env):** API credentials, base URLs, and target models are separated from the code. Never hardcode keys directly in `llm/llm.py` or anywhere else.
2. **Git Exclusions (.gitignore):** 
   - Excludes local environment folders (`.venv/`).
   - Excludes local document source files (`data/*.pdf`) to prevent checking in large binaries or proprietary documents.
   - Excludes generated database files (`storage/`) which contain serializations of document text.
   - Excludes sensitive credential files (`.env`).

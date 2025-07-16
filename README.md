# Fusion-RAG
Streamlit application for hybrid RAG using BM25, FAISS &amp; Google Gemini model to answer queries from PDF docs. Supports real-time chat with context-aware LLM responses.

---

# 🔍 Fusion RAG with Hybrid Search

A Streamlit-based Retrieval-Augmented Generation (RAG) app that uses **hybrid search** (BM25 + FAISS) and **Google Gemini (Gemini 2.5 Pro)** to answer natural language queries from uploaded PDF documents. It features real-time streaming responses with contextual understanding via LLMs.

---

## 🚀 Features

- 📄 Load and process multiple PDF documents from a directory  
- 🔍 Hybrid retrieval using BM25 (sparse) + FAISS (dense) embeddings  
- 🤖 AI-powered answers using Google Gemini (2.5 Pro) LLM  
- 🧠 Document chunking using LangChain's `RecursiveCharacterTextSplitter`  
- 💬 Conversational UI with history using Streamlit Chat  
- 🔐 Secure API key handling with `.env` support  

---

## 📦 Tech Stack

- **Frontend/UI**: Streamlit  
- **LLM**: Google Generative AI (Gemini 2.5 Pro) via `langchain_google_genai`  
- **Retrieval**: BM25 (`BM25Retriever`), FAISS, and EnsembleRetriever from LangChain  
- **Embedding Models**: Google Generative AI Embeddings (`embedding-001`)  
- **Document Loaders**: PyPDFLoader (from LangChain)  
- **Vector Stores**: FAISS  
- **Text Splitting**: RecursiveCharacterTextSplitter  
- **Environment Management**: `python-dotenv`  
- **Language**: Python 3.10+  

---


## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/PRP2003/Fusion-RAG.git
cd Fusion-RAG
```
### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure .env File
```bash
GOOGLE_API_KEY='your_google_api_key_here'
```
### ▶️ Run the Application
```bash
streamlit run fusion-rag.py
```
### 🧪 Example Use Cases

1. Query internal reports or research papers 

2. Extract key information from financial or legal PDFs

3. Build intelligent document Q&A systems

### 📁 Project Structure
```bash
fusion-rag-hybrid/
│
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .env                      # API key config (not committed)
└── README.md                 # Project documentation
```

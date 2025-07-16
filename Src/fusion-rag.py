import streamlit as st
import os
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader, JSONLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_community.embeddings import FastEmbedEmbeddings, HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, ElasticSearchBM25Retriever, EnsembleRetriever
from langchain_community.llms import ollama
from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import StructuredOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Streamlit App Title
st.title('Fusion RAG with Hybrid Search 🔍')
st.write('Ask your query about the documents where you have provided the data')

# Input Fields
st.session_state.google_api_key = st.text_input("Enter your Google API Key", type="password")
st.session_state.pdf_folder_path = st.text_input("Enter PDF Folder Path")

# Validate inputs
if not st.session_state.google_api_key or not st.session_state.pdf_folder_path:
    st.warning("Please enter your Google API key and PDF folder path to begin.")
    st.stop()

# Set the environment variable
os.environ['GOOGLE_API_KEY'] = st.session_state.google_api_key

# Load and split documents
@st.cache_resource
def load_process_documents(pdf_folder_path):
    all_docs = []

    for root, _, files in os.walk(pdf_folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                all_docs.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(all_docs)
    return docs

docs = load_process_documents(st.session_state.pdf_folder_path)

# Hybrid Retrieval Setup
@st.cache_resource
def setup_retrieve(_docs):
    bm25retriever = BM25Retriever.from_documents(_docs)
    bm25retriever.k = 3

    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    ensemble_model = EnsembleRetriever(
        retrievers=[bm25retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_model

retriever = setup_retrieve(docs)

# LLM Setup
llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=0.7)

# Prompt Template
prompt_template = """
Answer the question based only on the following context:

{context}

Question: {prompt}
"""

final_prompt = ChatPromptTemplate.from_template(prompt_template)

# RAG Chain
rag_chain = (
    {'context': retriever, 'prompt': RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)

# Chat History Initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display Previous Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Handling
if user_query := st.chat_input('What is your question?'):
    st.chat_message('user').markdown(user_query)
    st.session_state.messages.append({'role': 'user', 'content': user_query})

    # Generate Response
    # response = rag_chain.invoke(user_query)
    response = rag_chain.stream(user_query)

        # Stream response
    with st.chat_message('assistant'):
        stream_placeholder = st.empty()
        full_response = ""
        for chunk in rag_chain.stream(user_query):
            full_response += chunk
            stream_placeholder.markdown(full_response)

    st.session_state.messages.append({'role': 'assistant', 'content': full_response})


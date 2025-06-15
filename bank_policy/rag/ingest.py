# bank_policy/rag/ingestion.py

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
RAW_DATA_DIR = Path("data/raw")
VECTORSTORE_DIR = Path("models/vectorstore")


def load_documents():
    """Load PDFs and TXTs from raw data folder."""
    documents = []
    for file_name in os.listdir(RAW_DATA_DIR):
        file_path = RAW_DATA_DIR / file_name
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
        elif file_name.endswith(".txt"):
            loader = TextLoader(str(file_path))
        else:
            print(f"Skipping unsupported file: {file_name}")
            continue
        documents.extend(loader.load())
    return documents


def split_documents(documents):
    """Split documents into chunks of 500 chars with 50 char overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def embed_and_store(doc_chunks):
    """Create embeddings and save FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Vector store saved at {VECTORSTORE_DIR}")


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    print("Splitting documents into chunks...")
    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    print("Embedding chunks and saving vector store...")
    embed_and_store(chunks)

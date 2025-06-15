# bank_policy/rag/ingest.py

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# Define input/output paths
RAW_DATA_DIR = Path("data/raw")
VECTORSTORE_DIR = Path("models/vectorstore")


def load_documents():
    """Load .pdf and .txt documents from the raw data directory."""
    documents = []
    for file_name in os.listdir(RAW_DATA_DIR):
        file_path = RAW_DATA_DIR / file_name
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
        elif file_name.endswith(".txt"):
            loader = TextLoader(str(file_path))
        else:
            print(f" Skipping unsupported file: {file_name}")
            continue
        documents.extend(loader.load())
    return documents


def split_documents(docs):
    """Split documents into manageable chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def embed_and_store(chunks):
    """Create vector store from chunks and save it locally."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f" Vector store saved at: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print("Splitting into chunks...")
    chunks = split_documents(docs)

    print(" Generating embeddings and saving vector store...")
    embed_and_store(chunks)

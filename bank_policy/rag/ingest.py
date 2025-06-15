# bank_policy/rag/ingest.py

import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()  # Loads the .env file into environment variables

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable")


load_dotenv()

# Define input and output paths
RAW_DATA_DIR = Path("data/raw")
VECTORSTORE_DIR = Path("models/vectorstore")  # Consistent with structure

def load_documents():
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

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_and_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Vector store saved at: {VECTORSTORE_DIR}")

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print("Splitting into chunks...")
    chunks = split_documents(docs)

    print("Generating embeddings and saving...")
    embed_and_store(chunks)

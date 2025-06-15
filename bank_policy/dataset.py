# bank_policy/dataset.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from bank_policy.config import RAW_DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL_NAME
from pathlib import Path
import os


def ingest_documents():
    print("[INFO] Loading documents...")
    docs = []
    for file in RAW_DATA_DIR.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())

    print(f"[INFO] Loaded {len(docs)} documents.")

    print("[INFO] Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    print(f"[INFO] Total chunks: {len(split_docs)}")

    print("[INFO] Embedding documents...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print("[✅] Vector store saved at:", VECTORSTORE_DIR)


if __name__ == "__main__":
    ingest_documents()

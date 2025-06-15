# bank_policy/rag/retrieve.py

import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Define where your vector store is saved
VECTORSTORE_DIR = Path("models/vectorstore")

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(VECTORSTORE_DIR), embeddings,
                            allow_dangerous_deserialization=True)

def retrieve_documents(query: str, k: int = 3):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    print("Retrieval Mode")
    query = input("Ask a question about the document: ")

    results = retrieve_documents(query)

    print("\nTop results:")
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1} ---\n{doc.page_content}")

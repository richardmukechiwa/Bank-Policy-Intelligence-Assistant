import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings  # ✅ NEW
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from bank_policy.config import VECTORSTORE_DIR, DOCUMENTS_DIR
from langchain_huggingface import HuggingFaceEndpoint



def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ UPDATED
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True  # ⚠️ Only if you're sure the data is safe
    )
    return vectorstore


if __name__ == "__main__":
    print("Ask your questions about the documents! Type 'exit' to quit.\n")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅

llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-small",
        temperature=0,
        max_new_tokens=256  
    )

    

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

while True:
    query = input("Your question: ")
    if query.lower() == "exit":
        break
    response = qa_chain.invoke({"query": query})
    print(f"\nAnswer:\n{response['result']}\n")
    
    # Optional: Print sources
    for i, doc in enumerate(response["source_documents"], start=1):
        print(f"Source Document {i}:\n{doc.page_content}\n---\n")


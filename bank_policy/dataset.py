# bank_policy/dataset.py

import os
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.utils import convert_files_to_docs, clean_wiki_text
from bank_policy.config import RAW_DATA_DIR, FAISS_INDEX_PATH, FAISS_CONFIG_PATH

def build_faiss_index():
    print("📂 Loading documents...")
    all_docs = convert_files_to_docs(
        dir_path=str(RAW_DATA_DIR),
        clean_func=clean_wiki_text,
        split_paragraphs=True
    )

    print("📚 Creating FAISS Document Store...")
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
    document_store.write_documents(all_docs)

    print("🔍 Initializing Retriever...")
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        passage_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=False
    )

    print("📌 Embedding and saving index...")
    document_store.update_embeddings(retriever)
    document_store.save(str(FAISS_INDEX_PATH))
    document_store.save_index(str(FAISS_CONFIG_PATH))
    print("✅ FAISS index saved to:", FAISS_INDEX_PATH)


if __name__ == "__main__":
    build_faiss_index()

# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, PromptNode, PromptTemplate
from haystack.pipelines import ExtractiveQAPipeline
from bank_policy.config import FAISS_INDEX_PATH, FAISS_CONFIG_PATH, MODEL_DIR

# Load .env and HuggingFace token (optional if using public model)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="Bank Policy RAG Assistant")
st.title("🏦 Bank Policy Intelligence Assistant")
st.markdown("Ask natural language questions about internal policies, compliance, or regulations.")

@st.cache_resource
def load_pipeline():
    # Load FAISS store
    doc_store = FAISSDocumentStore.load(str(FAISS_INDEX_PATH), index_config_path=str(FAISS_CONFIG_PATH))

    retriever = DensePassageRetriever(
        document_store=doc_store,
        query_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        passage_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=False
    )

    prompt_template = PromptTemplate(
        name="bank_rag_prompt",
        prompt_text="Given the following documents:\n{documents}\n\nAnswer the question:\n{query}"
    )

    prompt_node = PromptNode(
        model_name_or_path="google/flan-t5-base",
        api_key=hf_token,
        default_prompt_template=prompt_template,
        max_length=512
    )

    pipeline = ExtractiveQAPipeline(reader=prompt_node, retriever=retriever)
    return pipeline


qa_pipeline = load_pipeline()

query = st.text_input("🔍 Ask your question")

if query:
    with st.spinner("Retrieving and answering..."):
        result = qa_pipeline.run(query=query, params={"Retriever": {"top_k": 3}})
        st.subheader("📘 Answer")
        st.success(result["answers"][0].answer)

        with st.expander("📚 Source Documents"):
            for doc in result["documents"]:
                st.markdown(f"**Score:** {doc.score:.2f}")
                st.text(doc.content[:300] + "...")
st.markdown(f"**URL:** {doc.meta['url']}")

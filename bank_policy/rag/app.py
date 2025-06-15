# app.py

import streamlit as st
from bank_policy.modeling.predict import create_qa_chain
import os





os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


st.set_page_config(page_title="Bank Policy Intelligence Assistant")

st.title("📄 Bank Policy Intelligence Assistant")
st.markdown("Ask questions about your bank’s internal policies or compliance documents.")

query = st.text_input("🔎 Ask a question")

if query:
    with st.spinner("Thinking..."):
        qa_chain = create_qa_chain()
        result = qa_chain.invoke({"query": query})

        st.markdown("### 📘 Answer")
        st.success(result["result"])

        with st.expander("📚 Source Documents"):
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                st.markdown(f"**Source {i}:** `{source}`")
                st.text(doc.page_content[:300] + "...")

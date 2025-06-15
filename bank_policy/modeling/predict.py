# bank_policy/modeling/predict.py

from bank_policy.config import VECTORSTORE_DIR, EMBEDDING_MODEL_NAME, LLM_REPO_ID
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True)

def create_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        model_kwargs={"temperature": 0.3, "max_length": 512},
        huggingfacehub_api_token=hf_token
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


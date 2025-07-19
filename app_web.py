# ✅ app_web.py - Streamlit Multi-PDF RAG Bot

import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableConfig

# ✅ Load environment
load_dotenv()
model_name = os.getenv("LLM_MODEL", "llama3")

# ✅ UI setup
st.set_page_config(page_title="Multi-PDF RAG Bot", layout="wide")
st.title("📄 Multi-PDF RAG Chatbot")

# ✅ Load PDFs and extract text
@st.cache_resource
def load_pdf_text():
    pdf_folder = "data"
    pdf_files = ["resume.pdf"]
    all_text = ""
    for file_name in pdf_files:
        file_path = os.path.join(pdf_folder, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        all_text += f"\n--- Start of {file_name} ---\n{text}\n--- End of {file_name} ---\n"
    return all_text

# ✅ Vectorstore setup
@st.cache_resource
def setup_rag(all_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([all_text])
    if not docs:
        raise ValueError("❌ No text chunks generated.")
    embedding = OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever()
    llm = Ollama(model=model_name)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Initialize
with st.spinner("🔍 Processing PDF and loading model..."):
    all_text = load_pdf_text()
    rqa = setup_rag(all_text)

# ✅ Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("💬 Ask your question:")
if query:
    with st.spinner("🤖 Thinking..."):
        result = rqa.invoke(query, config=RunnableConfig())
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", result["result"]))

# ✅ Display chat
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💻 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")

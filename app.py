# ✅ app.py - Multi-PDF RAG Bot

import os
import fitz  # PyMuPDF
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

# ✅ PDF files list
pdf_folder = "data"
pdf_files = [ "resume.pdf"]

# ✅ Extract text from all PDFs
print("📄 Extracting text from PDFs...")
all_text = ""
for file_name in pdf_files:
    file_path = os.path.join(pdf_folder, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    all_text += f"\n--- Start of {file_name} ---\n{text}\n--- End of {file_name} ---\n"

# ✅ Check for empty content
if not all_text.strip():
    raise ValueError("❌ PDFs have no extractable text.")

# ✅ Split into chunks
print("🧩 Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([all_text])

if len(docs) == 0:
    raise ValueError("❌ No text chunks generated. Check PDFs for extractable content.")

# ✅ Create FAISS vector store
print("📦 Generating embeddings...")
embedding = OllamaEmbeddings(model=model_name)
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# ✅ Setup RAG
print("🤖 Loading LLaMA 3 model...")
llm = Ollama(model=model_name)
rqa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Chat
print("\n🟢 Multi-PDF RAG Chatbot ready! Ask anything (type 'exit' to quit)\n")
while True:
    query = input("💬 You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = rqa.invoke(query, config=RunnableConfig())
    print("🤖 Bot:", result["result"])

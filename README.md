# Multi-PDF-RAG-Chatbot
A simple multi-PDF RAG bot using LangChain, FAISS, and Ollama for local document question answering.


####🧠 Multi-PDF RAG Chatbot using LLaMA 3
This project is a local RAG (Retrieval-Augmented Generation) chatbot that can read and answer questions from multiple PDF files using LLaMA 3 running on Ollama, without needing OpenAI API keys.

✅ Features

📄 Accepts multiple PDFs as knowledge base

🧩 Splits text into chunks for better context handling

📦 Uses FAISS for vector storage and semantic retrieval

🤖 Uses local LLaMA 3 model for inference via Ollama

🔁 Chat loop interface to interact with the data

###🛠️ Tech Stack
Python
LangChain
Ollama (LLaMA 3)
FAISS
PyMuPDF
dotenv


####🧪 How to Run
Install Ollama & pull model

ollama run llama3
Install dependencies

pip install -r requirements.txt
Add PDF files to data/ folder

📁 data/
├── resume.pdf
└── data.pdf
Run the chatbot

python app.py


🖼️ Sample Output
<img width="1829" height="674" alt="image" src="https://github.com/user-attachments/assets/aaa14899-3ac7-4471-8ef8-819f62a167b0" />


####🚀 Future Enhancements
Integrate embedding caching to avoid recomputation
Optionally serve as a local personal assistant bot


####🧠 Credits
Built using LangChain
Embeddings and LLMs via Ollama

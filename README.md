# 📄 Conversational PDF Assistant With Chat History

A **Streamlit**-based web application that allows users to upload PDF documents and interact with them through natural language conversations. The app supports chat history, session management, and uses **RAG (Retrieval-Augmented Generation)** techniques for accurate responses.

---

## 🚀 Features

- Upload one or more PDF files.
- Ask questions based on the uploaded content.
- Persistent session-based chat history.
- Context-aware question reformulation using LangChain.
- Fast and accurate responses powered by **Groq's Llama3-8b-8192** model.
- Vector database storage using **ChromaDB**.
- Embedding generation via **HuggingFace (MiniLM-L6-v2)**.

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) – UI
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [Groq](https://groq.com/) – LLM provider (Llama3)
- [ChromaDB](https://www.trychroma.com/) – Vector store
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) – Text embeddings
- [PyPDFLoader](https://docs.langchain.com/docs/modules/data_connection/document_loaders/pdf) – PDF parsing
- [dotenv](https://pypi.org/project/python-dotenv/) – Environment variable management

---

## 📥 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/pdf-chat-assistant.git
cd pdf-chat-assistant

pip install -r requirements.txt

Setup Environment Variables

Create a .env file in the root directory:

HF_TOKEN=your_huggingface_api_token
Also GROQ
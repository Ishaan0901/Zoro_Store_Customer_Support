# 🤖 Zoro Store Customer Support Bot

An AI-powered Customer Support Chatbot built using **Retrieval-Augmented Generation (RAG)** that provides accurate, context-aware responses by retrieving information from product manuals, FAQs, warranty policies, and historical customer support tickets.

The chatbot combines **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, **Mistral AI**, and **Streamlit** to deliver an interactive customer support experience.
---
## 🚀 Features: 

- 💬 Intelligent customer support chatbot
- 📄 Answers questions using product manuals, FAQs, warranty policies, and support documents
- 🔍 Retrieval-Augmented Generation (RAG)
- 🧠 Semantic search using vector embeddings
- 💾 ChromaDB vector database
- 📚 Supports both PDF documents and CSV datasets
- 🗂️ Maximum Marginal Relevance (MMR) retrieval
- 💭 Conversation history support
- ⚡ Fast Streamlit interface with cached models

---

## 🛠️ Tech Stack

- Python
- LangChain
- Streamlit
- ChromaDB
- Hugging Face Embeddings
- Mistral AI
- Sentence Transformers
- Pandas

---

## 📂 Knowledge Base

The chatbot retrieves information from:

- Product Manuals
- FAQ Documents
- Store Policies
- Warranty Information
- Customer Support Guides
- Historical Customer Support Ticket Dataset

---

## 🏗️ Project Workflow

```text
User Query
      │
      ▼
Conversation History
      │
      ▼
Embedding Model
      │
      ▼
ChromaDB Retriever (MMR)
      │
      ▼
Relevant Documents
      │
      ▼
Prompt Template
      │
      ▼
Mistral AI
      │
      ▼
Final Response
```

---

## 📁 Project Structure

```
Zoro_Store_Customer_Support_Bot/
│
├── resources/
│   ├── *.pdf
│   └── cleaned_data.csv
│
├── zoro_store_database/
│
├── phase1.py          # Creates Vector Database
├── app.py             # Streamlit Application
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Zoro_Store_Customer_Support_Bot.git
```

### Navigate

```bash
cd Zoro_Store_Customer_Support_Bot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Environment File

Create a `.env` file and add:

```env
MISTRAL_API_KEY=YOUR_API_KEY
```

### Build Vector Database

```bash
python phase1.py
```

### Launch Application

```bash
streamlit run app.py
```

---

## 📸 Demo

_Add screenshots or a GIF of the chatbot here._

---

## 🎯 Future Improvements

- Source citations for every response
- Multi-turn conversational memory
- Chat history persistence
- Hybrid Search (Keyword + Vector Search)
- Voice-based customer support
- Admin dashboard for analytics
- Multi-language support

---

## 📚 Key AI Concepts Used

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Dense Retrieval
- Maximum Marginal Relevance (MMR)
- Prompt Engineering
- Conversational AI
- Large Language Models (LLMs)

---

## 👨‍💻 Author

**Ishaan Sharma**

- GitHub: https://github.com/Ishaan0901
- LinkedIn: https://www.linkedin.com/in/ishaansharma195/
- Portfolio: https://ishaan0901.github.io/Portfolio/

---

## ⭐ If you found this project useful, consider giving it a star!

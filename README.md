# ⚖️ PakLex — Pakistan Constitution Q&A Chatbot

> AI-powered legal assistant for the Constitution of the Islamic Republic of Pakistan

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-llama3.1:8b-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🧠 What is PakLex?

PakLex is a full-stack AI chatbot that answers questions about the **Constitution of Pakistan** using Retrieval-Augmented Generation (RAG). Ask about fundamental rights, parliamentary procedure, emergency powers, the judiciary, and more — in plain English.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
React Frontend (port 3000)
     │  POST /ask/stream (SSE)
     ▼
FastAPI Backend (port 8000)
     │
     ├─► ChromaDB (vector search)
     │      └─ all-MiniLM-L6-v2 embeddings
     │      └─ Top-4 relevant constitution chunks
     │
     └─► Ollama (llama3.1:8b)
            └─ RAG prompt with retrieved context
            └─ Streaming token response
```

---

## ✨ Features

- 🔍 **RAG Pipeline** — ChromaDB vector search retrieves relevant constitutional text before generation
- ⚡ **Streaming responses** — answers appear word by word via Server-Sent Events
- 🌙 **Dark / Light mode** — persisted to localStorage
- 💡 **Smart suggestions** — related questions after every answer
- 📚 **Source citations** — shows which Articles were referenced
- 📋 **Copy to clipboard** — one-click answer copy
- ♿ **Accessible** — ARIA labels, keyboard navigation, screen reader support
- 📱 **Responsive** — works on mobile and desktop

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B via Ollama |
| Vector DB | ChromaDB + all-MiniLM-L6-v2 |
| Backend | FastAPI + Python 3.11 |
| Frontend | React 18 + custom CSS |
| Embeddings | SentenceTransformers |
| PDF parsing | pdfplumber |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/download) installed
- 6GB+ VRAM GPU (or CPU with patience)

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/paklex.git
cd paklex
```

### 2. Backend setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your paths
```

### 3. Pull the LLM
```bash
ollama pull llama3.1:8b
```

### 4. Ingest the constitution
```bash
# Add your constitution PDF to the project folder
# Update PDF_PATH in .env
python ingest.py
```

### 5. Start the backend
```bash
uvicorn main:app --reload --port 8000
```

### 6. Start the frontend
```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) 🎉

---

## 📁 Project Structure

```
paklex/
├── main.py              # FastAPI backend — RAG + streaming
├── ingest.py            # PDF ingestion → ChromaDB
├── requirements.txt     # Python dependencies
├── .env.example         # Environment config template
├── chroma_db/           # Vector database (auto-generated)
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main React app + streaming
    │   ├── ErrorBoundary.jsx
    │   ├── index.js
    │   └── index.css        # Design system
    └── package.json
```

---

## 🔧 Environment Variables

```env
# Backend
PDF_PATH=./consitution rights.pdf
CHROMA_DIR=./chroma_db
COLLECTION=pakistan_constitution
EMBED_MODEL=all-MiniLM-L6-v2
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
TOP_K=4

# Frontend
REACT_APP_API_URL=http://localhost:8000
```

---

## 📡 API Reference

### `POST /ask/stream`
Streaming endpoint — returns Server-Sent Events.

**Request:**
```json
{
  "question": "What is Article 25?",
  "max_tokens": 512,
  "temperature": 0.3
}
```

**Stream events:**
```
data: {"token": "Article"}
data: {"token": " 25..."}
data: {"sources": ["Article 25"], "suggestions": [...]}
data: [DONE]
```

### `POST /ask`
Non-streaming fallback.

### `GET /health`
Returns Ollama and ChromaDB status.

---

## 🗺️ Roadmap

- [ ] Urdu language support
- [ ] Conversation history / memory
- [ ] Voice input (Web Speech API)
- [ ] Bookmark and export answers
- [ ] Docker deployment
- [ ] GraphRAG for article relationships
- [ ] Bilingual English / Urdu RAG

---

## 👨‍💻 Author

**Asher Farooq**
BSAI Graduate — IQRA University, Islamabad
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

> ⚠️ PakLex is an AI assistant. Always verify legal matters with a qualified lawyer.

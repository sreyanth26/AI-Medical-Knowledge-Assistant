# 🏥 AI Medical Knowledge Assistant

> An open-source **Retrieval-Augmented Generation (RAG)** system for querying medical documents using natural language. Answers are grounded in your uploaded documents — not hallucinated.

---

## 📌 Overview

Upload medical PDFs (WHO guidelines, drug manuals, clinical studies) and ask questions like:

- *"What is the treatment for dengue fever?"*
- *"What are the side effects of paracetamol?"*
- *"What is the recommended dosage of amoxicillin?"*
- *"What are the symptoms of malaria?"*

The system retrieves relevant sections from your documents and uses a **local LLM** (Llama 3 / Mistral via Ollama) to generate accurate, cited answers.

---

## 🧠 How It Works (RAG Pipeline)

```
User Question
     ↓
Embedding Model  (sentence-transformers/all-MiniLM-L6-v2)
     ↓
FAISS Vector Database Search
     ↓
Top-K Relevant Document Chunks Retrieved
     ↓
LLM Generates Context-Based Answer  (Llama 3 via Ollama)
     ↓
Answer + Source Citations Returned
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| RAG Framework | [LangChain](https://langchain.com) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Database | [FAISS](https://github.com/facebookresearch/faiss) |
| Local LLM | [Phi3](https://ollama.ai/library/phi3) / [Mistral](https://ollama.ai/library/mistral) / [Llama3](https://ollama.ai/library/llama3) via [Ollama](https://ollama.ai) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend | [Streamlit](https://streamlit.io) |
| Language | Python 3.10+ |

---

## 📂 Project Structure

```
AI-Medical-Knowledge-Assistant/
│
├── data/
│   ├── medical_guidelines/     # WHO guidelines, clinical protocols
│   ├── drug_database/          # Drug manuals, pharmacopoeias
│   └── research_papers/        # Clinical studies, research
│
├── backend/
│   ├── document_loader.py      # PDF/TXT ingestion & chunking
│   ├── embeddings.py           # HuggingFace embedding model
│   ├── vector_store.py         # FAISS vector store operations
│   ├── rag_pipeline.py         # Core RAG + LLM generation
│   └── api.py                  # FastAPI REST endpoints
│
├── frontend/
│   └── app.py                  # Streamlit web interface
│
├── models/                     # FAISS index saved here
├── .env.example                # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/AI-Medical-Knowledge-Assistant.git
cd AI-Medical-Knowledge-Assistant

pip install -r requirements.txt
```

### 2. Set Up Ollama (Local LLM)

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3      # 8B model, ~4.7 GB
# OR
ollama pull mistral     # 7B model, ~4.1 GB

# Start the Ollama server:
ollama serve
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 4. Start the Backend API

```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 5. Start the Frontend

```bash
streamlit run frontend/app.py
```

Opens at: http://localhost:8501

### 6. Upload Documents & Ask Questions

1. Upload PDFs in the sidebar
2. Click **Index Documents**
3. Type your question
4. Get answers with citations!

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/status` | System status |
| `POST` | `/upload` | Upload a PDF/TXT file |
| `POST` | `/ask` | Ask a question |
| `POST` | `/summarize` | Summarize text |
| `DELETE` | `/index` | Clear the vector index |

### Example API Call

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@dengue_guidelines.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the treatment for dengue fever?", "top_k": 5}'
```

---

## 🔐 Safety & Disclaimers

- Answers are strictly based on retrieved documents
- Each answer includes source citations
- Built-in medical disclaimer in the UI
- **Not a replacement for professional medical advice**

---

## 🚀 Future Extensions

- [ ] Voice-based medical assistant
- [ ] Multi-language support
- [ ] Integration with hospital management systems
- [ ] Medical chatbot for rural healthcare
- [ ] GPU acceleration for faster embeddings
- [ ] Support for DICOM / medical imaging metadata

---

## 📄 Resume Bullet Points

> *Developed an open-source AI Medical Knowledge Assistant using Retrieval-Augmented Generation (RAG). Implemented semantic search with FAISS and HuggingFace embeddings to retrieve medical information from clinical guidelines and drug databases. Integrated a local LLM (Llama 3) to generate context-aware answers with document citations.*

---

## 📜 License

MIT License — free for personal and commercial use.

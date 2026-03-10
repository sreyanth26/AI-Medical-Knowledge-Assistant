# MedRAG — Deployment Guide
> AI Medical Knowledge Assistant v2.0

---

## 📋 Table of Contents
1. [Local Development](#local-development)
2. [Deploy on Render (Free)](#deploy-on-render-free)
3. [Deploy on Railway (Easy)](#deploy-on-railway-easy)
4. [Deploy on AWS EC2](#deploy-on-aws-ec2)
5. [Deploy on Azure Container Apps](#deploy-on-azure-container-apps)
6. [Docker Deployment](#docker-deployment)
7. [Environment Variables](#environment-variables)
8. [Running Tests](#running-tests)

---

## 1. Local Development

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed

### Steps

```powershell
# Clone / navigate to project
cd AI-Medical-Knowledge-Assistant

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows PowerShell
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install pytest httpx     # for testing

# Copy environment config
copy .env.example .env

# Pull and start LLM
ollama pull llama3
ollama serve                 # keep this running

# Terminal 2: Start Backend
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Start Frontend
cd ..
streamlit run frontend/app.py
```

**Access:**
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs
- API ReDoc: http://localhost:8000/redoc

---

## 2. Deploy on Render (Free — Recommended for Demo)

Render offers free hosting with sleep on idle (perfect for portfolio demos).

### Backend (FastAPI)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Configure:
   ```
   Name:          medrag-api
   Root Directory: backend
   Build Command:  pip install -r ../requirements.txt
   Start Command:  uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variables:
   ```
   LLM_MODEL=llama3
   OLLAMA_HOST=https://your-ollama-instance.com  # or use OpenAI API
   VECTOR_INDEX_PATH=/tmp/faiss_index
   ```

> **Note:** Render free tier doesn't support persistent storage.
> Use a cloud vector DB (Pinecone, Weaviate) for production persistence.

### Frontend (Streamlit)

Option A — Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push to GitHub
2. Go to share.streamlit.io → Deploy
3. Set `API_BASE_URL` in secrets:
   ```toml
   # .streamlit/secrets.toml
   API_BASE_URL = "https://medrag-api.onrender.com"
   ```

---

## 3. Deploy on Railway (Easiest)

Railway gives $5 free credit/month and supports persistent volumes.

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy backend
cd backend
railway up

# Set environment variables
railway variables set LLM_MODEL=llama3
railway variables set VECTOR_INDEX_PATH=/app/models/faiss_index
```

---

## 4. Deploy on AWS EC2

### Launch Instance
- **AMI:** Ubuntu 22.04 LTS
- **Type:** t3.medium (2 vCPU, 4 GB RAM) — minimum for Llama 3
- **Storage:** 20 GB minimum (model weights ~5 GB)
- **Security Group:** Open ports 22 (SSH), 8000 (API), 8501 (Streamlit)

### Setup Script

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update and install Python
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3 &

# Clone project
git clone https://github.com/yourname/AI-Medical-Knowledge-Assistant.git
cd AI-Medical-Knowledge-Assistant

# Setup virtualenv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service for backend
sudo tee /etc/systemd/system/medrag.service > /dev/null <<EOF
[Unit]
Description=MedRAG FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/AI-Medical-Knowledge-Assistant/backend
Environment="LLM_MODEL=llama3"
Environment="VECTOR_INDEX_PATH=/home/ubuntu/AI-Medical-Knowledge-Assistant/models/faiss_index"
ExecStart=/home/ubuntu/AI-Medical-Knowledge-Assistant/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable medrag
sudo systemctl start medrag

# Check status
sudo systemctl status medrag
```

### Nginx Reverse Proxy (for HTTPS)

```nginx
# /etc/nginx/sites-available/medrag
server {
    server_name your-domain.com;
    
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
sudo certbot --nginx -d your-domain.com  # free SSL
```

---

## 5. Docker Deployment

### Dockerfile (Backend)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: "3.9"
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_MODEL=llama3
      - OLLAMA_HOST=http://ollama:11434
      - VECTOR_INDEX_PATH=/app/models/faiss_index
    volumes:
      - ./models:/app/models
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  frontend:
    image: python:3.11-slim
    command: streamlit run /app/frontend/app.py --server.port 8501
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://backend:8000
    volumes:
      - ./frontend:/app/frontend
    depends_on:
      - backend

volumes:
  ollama_data:
```

```bash
# Build and run
docker-compose up --build

# Pull model inside Ollama container
docker exec -it <ollama_container_id> ollama pull llama3
```

---

## 6. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `VECTOR_INDEX_PATH` | `../models/faiss_index` | FAISS index storage path |
| `MAX_FILE_SIZE_MB` | `50` | Max upload file size |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API port |
| `API_BASE_URL` | `http://localhost:8000` | Frontend API URL |

---

## 7. Running Tests

```powershell
# Install test dependencies
pip install pytest httpx pytest-asyncio

# Run all tests
pytest tests/test_medrag.py -v

# Run with coverage
pip install pytest-cov
pytest tests/test_medrag.py -v --cov=backend --cov-report=html

# Run specific test class
pytest tests/test_medrag.py::TestAPIEndpoints -v

# Run and stop on first failure
pytest tests/test_medrag.py -x -v
```

### Expected Output
```
tests/test_medrag.py::TestDocumentLoader::test_load_text_file         PASSED
tests/test_medrag.py::TestDocumentLoader::test_chunk_metadata_populated PASSED
tests/test_medrag.py::TestVectorStore::test_add_and_search             PASSED
tests/test_medrag.py::TestRAGPipeline::test_answer_returns_response_object PASSED
tests/test_medrag.py::TestAPIEndpoints::test_upload_txt_file           PASSED
...
========= 35 passed in 12.43s =========
```

---

## 8. Production Checklist

Before sharing your demo link:

- [ ] Set `ALLOWED_ORIGINS` to your frontend domain (not `*`)
- [ ] Use HTTPS (certbot or cloud provider SSL)
- [ ] Set `MAX_FILE_SIZE_MB` appropriately
- [ ] Test all endpoints via `/docs`
- [ ] Upload sample data files from `data/` folder
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify Ollama is running and model is pulled
- [ ] Add your public URL to your resume/portfolio

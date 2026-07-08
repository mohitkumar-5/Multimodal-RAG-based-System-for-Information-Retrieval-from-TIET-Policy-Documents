# 🔍 PolicyLens — Thapar Institute RAG System

**PolicyLens** is an AI-powered academic assistant for Thapar Institute of Engineering & Technology (TIET), built on a Retrieval Augmented Generation (RAG) architecture. Ask questions about fees, hostel rules, course schemes, academic regulations, and more — and get answers grounded in 100+ official TIET documents.

---

## 🚀 Live Features

| Feature | Description |
|---------|-------------|
| 🤖 Multi-LLM | Llama 3.3 70B (Groq) with fallback to Qwen 2.5 & Llama 3.1 8B |
| 📚 100+ Docs | Full TIET policy corpus indexed in Qdrant vector DB |
| 🔍 MMR Search | Maximum Marginal Relevance retrieval + Cross-Encoder reranking |
| 🎙️ Voice Input | Groq Whisper STT transcription |
| 🗣️ Voice Output | gTTS text-to-speech playback |
| 🖼️ Vision | Upload images of documents for OCR-style Q&A |
| 🧠 Memory | Redis-backed session history (falls back to in-memory) |
| 🎨 Animated UI | Three.js WebGL jellyfish background + animated canvas |

---

## 📦 Download the PDF Corpus

> The 100+ TIET policy documents are **not stored in this repo** (too large).
> Download the full zipped corpus from Google Drive:

### ⬇️ [Click here to download tiet_docs.zip](https://drive.google.com/YOUR_LINK_HERE)

After downloading, extract and place the PDFs into the `data/` folder:

```bash
# After downloading tiet_docs.zip:
unzip tiet_docs.zip -d data/
```

**Or run the auto-download script:**

```bash
python download_docs.py
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PolicyLens.git
cd PolicyLens
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Then open `.env` and add your actual API keys:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
REDIS_URL=your_redis_url          # optional
HF_TOKEN=your_huggingface_token   # optional fallback
```

### 5. Run the Server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **http://127.0.0.1:8000**

---

## 🏗️ Project Structure

```
PolicyLens/
├── app/
│   ├── main.py              # FastAPI routes & server
│   ├── rag.py               # Core RAG pipeline (LangChain)
│   └── database.py          # Feedback DB (SQLite)
├── frontend/
│   ├── index.html           # 3-page UI (Home, Docs, Chat)
│   ├── style.css            # Design system + animations
│   ├── app.js               # Three.js background + chat logic
│   └── vercel.json          # Vercel deployment config
├── tests/
│   └── test_rag.py          # Unit tests
├── data/                    # ← Put your PDF corpus here
├── PolicyLens_Experiment.ipynb  # Jupyter notebook for experiments
├── download_docs.py         # Auto-download script for PDF corpus
├── Dockerfile               # Docker deployment
├── requirements.txt
├── .env.example             # Template for environment variables
└── README.md
```

---

## 🔑 API Keys You Need

| Key | Where to Get |
|-----|-------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — Free tier available |
| `QDRANT_URL` + `QDRANT_API_KEY` | [cloud.qdrant.io](https://cloud.qdrant.io) — Free tier (1GB) |
| `REDIS_URL` | [upstash.com](https://upstash.com) — Free tier available |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — Optional |

---

## 🐳 Docker Deployment

```bash
docker build -t policylens .
docker run -p 8000:8000 --env-file .env policylens
```

---

## 📓 Experiment Notebook

The `PolicyLens_Experiment.ipynb` notebook lets you test the full RAG pipeline interactively — embeddings, retrieval, LLM answering, and source inspection — without running the web server.

---

## ⚠️ Security Note

- **Never commit your `.env` file** — it is gitignored by default
- All API keys must be set as environment variables only
- The `.env.example` file shows the required variable names without actual values

---

## 🏛️ Built With

- [LangChain](https://langchain.com) — RAG pipeline & chains
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Qdrant](https://qdrant.tech) — Vector database
- [FastAPI](https://fastapi.tiangolo.com) — Backend API
- [Three.js](https://threejs.org) — WebGL background animation
- [HuggingFace](https://huggingface.co) — BGE embedding model

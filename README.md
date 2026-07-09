# 🪼 PolicyLens — Multimodal TIET Policy RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/Qdrant-DF3A1A?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant">
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/Groq_API-F55A42?style=for-the-badge&logo=groq&logoColor=white" alt="Groq">
</p>

<p align="center">
  PolicyLens is an AI-powered conversational agent designed to retrieve accurate information from Thapar Institute of Engineering and Technology (TIET) academic policies, syllabus schemes, and official circulars using Retrieval-Augmented Generation (RAG).
</p>

---

## 🚀 Live Website
*   **Link:** *[To be added later]*

## 🎥 Project Demo Video
*   **Video Link:** *[To be added later]*

---

## 💡 Why I Built This Project
Thapar Institute publishes all of its academic policies, credit lists, fee structures, and campus regulations across dozens of separate, complex PDF documents. For students, parents, and faculty, finding a specific rule—such as hostel allotment criteria, grading policies, or course schemes—requires opening and reading through hundreds of pages. 

I built PolicyLens to simplify this process. By creating a unified search interface, users can ask questions in plain English, speak their queries, or upload screenshots. The system instantly scans the official documents, extracts the relevant guidelines, and lists the answers clearly with direct source citations to save time and reduce confusion.

---

## ✨ Features

*   📁 **Supports 100+ TIET documents:** Indexes prospectus schedules, course catalogs, and academic rulebooks.
*   🎙️ **Voice Query Input:** Capture speech queries directly through the microphone with automatic transcription.
*   🖼️ **Visual OCR/Image Input:** Process screenshots of tables, schedules, or circulars to ask questions about them.
*   🗣️ **Text-to-Speech (TTS):** Play voice output of the generated answers.
*   🔍 **Semantic Search:** Grounded in Qdrant Cloud Vector Database and BGE dense vector embeddings.
*   ⚡ **Hybrid MMR Retrieval:** Utilizes Maximum Marginal Relevance to retrieve diverse, non-redundant contexts.
*   🧠 **Conversational Memory:** Remembers context across chat turns using Upstash Redis.
*   🎨 **Interactive Web UI:** Features a custom fluid WebGL jellyfish mouse-interactive background.

---

## 🎨 System Architecture & Workflow

![System Architecture](architecture.png)

The application coordinates data flows across the following modules:

1.  **WebGL Web UI (Frontend):** Renders the user-interactive chat window, WebGL fluid canvas, and handles browser speech-to-text recording and audio playback.
2.  **FastAPI Backend Server:** Acts as the primary router and controller. It parses form data, writes temporary image/voice assets, tracks API rate-limits, and coordinates database updates.
3.  **Multimodal Input Processing:**
    *   **Audio Inputs** are sent to Groq's Whisper-Large-v3 engine to receive high-fidelity transcriptions.
    *   **Image Inputs** are base64-encoded and passed to Llama 4 Scout Vision API to identify topics or text content.
4.  **Embedding & Vector Retrieval:** Query text is translated to a 768-dimension vector using Hugging Face's `bge-base-en-v1.5` model. Qdrant Cloud compares it against document vectors, returning the top matches.
5.  **Chain Assembly & Generation:** LangChain retrieves previous message logs from Upstash Redis, combines them with the Qdrant document contexts, and feeds the formatted prompt to Groq's Llama 3.3 70B model to generate the final response.

---

## 📂 Repository Structure
```text
PolicyLens/
├── app/                     # Backend Source Code
│   ├── database.py          # Redis connections, rate limiting, and feedback aggregates
│   ├── main.py              # FastAPI endpoints, routers, and CORS setup
│   └── rag.py               # Core LangChain RAG pipeline, LLM routers, and prompts
├── frontend/                # Single Page App Static Web UI
│   ├── app.js               # State management, speech transcription api, and WebGL code
│   ├── index.html           # Main Single Page App structure
│   ├── style.css            # Custom CSS style, glassmorphism templates, and animations
│   └── vercel.json          # Vercel deployment configurations
├── notebooks/               # Interactive Playground
│   └── TIET_RAG_Pipeline.ipynb # 16-step complete Jupyter notebook
├── tests/                   # Automated Tests
│   └── test_rag.py          # Unit tests (capping, models, fallbacks)
├── .env.example             # Template for API credentials
├── .gitignore               # Ignored files (e.g. .env, cache files)
├── LICENSE                  # MIT License
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
└── thapar_rag_scraper.py    # Auto-scraper script for TIET documents
```

---

## 🧠 Models Used
*   **Primary Chat LLM:** `Llama 3.3 70B Versatile` (via Groq) — Ultra-fast, highly accurate reasoning.
*   **Vision LLM:** `Llama 4 Scout 17B Instruct` (via Groq) — For document layout and image analysis.
*   **Speech-to-Text (STT):** `Whisper Large v3` (via Groq) — For accurate voice transcriptions.
*   **Text-to-Speech (TTS):** `gTTS` (Google Text-To-Speech) — For playing vocalized answers.
*   **Semantic Embeddings:** `BAAI/bge-base-en-v1.5` — High-performance dense vector model.
*   **Fallback LLM:** `Qwen 2.5 7B` / `Llama 3.1 8B` (via Groq) — Backup chains to avoid API rate limits.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mohitkumar-5/Multimodal-RAG-based-System-for-Information-Retrieval-from-TIET-Policy-Documents.git
cd Multimodal-RAG-based-System-for-Information-Retrieval-from-TIET-Policy-Documents
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your actual API credentials:
```bash
cp .env.example .env
```
Open `.env` and set your credentials:
```env
GROQ_API_KEY=your_groq_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_key
COLLECTION_NAME=tiet_policy_docs
REDIS_URL=rediss://default:...@upstash.io:6379
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
HF_TOKEN=your_huggingface_token
```

### 5. Scrape policy documents (Optional)
If you do not have the document corpus locally, run the scraper script to retrieve the TIET PDFs:
```bash
python thapar_rag_scraper.py
```
This will download the documents and place them inside the `data/` directory.

### 6. Run the Application
Start the FastAPI server:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **127.0.0.1:8000** or **localhost:8000** in your web browser.

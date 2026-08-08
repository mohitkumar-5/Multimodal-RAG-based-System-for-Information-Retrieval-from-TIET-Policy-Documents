# 🪼 PolicyLens — Multimodal RAG Based System for Information Retrieval from TIET Policy Documents

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/Qdrant-DF3A1A?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant">
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/Groq_API-F55A42?style=for-the-badge&logo=groq&logoColor=white" alt="Groq">
</p>

<p align="center">
  PolicyLens is an advanced intelligent search and conversational system developed to index, organize, and query official academic regulations, course schemes, fee sheets, and hostel policies of the Thapar Institute of Engineering and Technology (TIET).
</p>

---



---

## 💡 Why I Built This Project
Every year, students, parents, and faculty spend hours digging through dozens of different TIET PDF guides, circulars, and handbooks to find specific details—such as tuition fee details, course credits, grading requirements, or hostel eligibility guidelines. These rules are scattered across hundreds of pages, making manual search slow and confusing.

I built PolicyLens to consolidate all of these scattered official documents into a single, cohesive, smart chatbot interface. By utilizing Retrieval-Augmented Generation, users can ask questions in plain English, speak their queries, or upload screenshots of circulars, and instantly receive exact point-wise answers verified directly against official sources.

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

Here is the architectural overview of how PolicyLens processes data and handles queries:

![System Architecture](architecture.png)

### 1. Offline Document Ingestion Pipeline (One-Time Ingestion)
Before users can query the system, the document library is indexed using a structured ingestion pipeline:
*   **TIET Policy Documents (PDFs):** Official prospectuses, fee sheets, and circulars are placed in the ingestion queue.
*   **Text & Table Extraction:** A parser extracts structured text, tabular data, and images from the documents.
*   **Semantic Chunking:** The extracted content is broken down into small, semantically meaningful text blocks (chunks) with overlap to preserve context across boundaries.
*   **Embedding Generation:** Each text chunk is converted into a 768-dimension vector embedding using the Hugging Face Serverless Inference API (`bge-base-en-v1.5` model).
*   **Vector Storage:** The generated vector embeddings, along with original text and file metadata (filename, page numbers), are stored in the Qdrant Cloud vector database.

### 2. Online Query Retrieval & Generation Pipeline (Real-Time Execution)
When a user interacts with the application:
1.  **User Input:** The user submits a query via **Text**, **Voice** (recording), or **Image** (screenshot) through the Web UI.
2.  **Input Processing:** The FastAPI Backend receives the request:
    *   *Voice recordings* are transcribed to text using Groq's Whisper Large v3 (STT) model.
    *   *Images* are parsed and analyzed using Groq's Llama 4 Scout Vision model to extract questions or tabular data.
3.  **Embedding Generation:** The resulting search text query is converted into a vector representation using the same serverless Inference API.
4.  **Vector Search:** The query vector is matched against the Qdrant Cloud database. The system retrieves the top relevant document chunks using MMR (Maximum Marginal Relevance) to ensure relevance and diversity of information.
5.  **Context & Prompt Assembly:** The backend retrieves the session's chat history from Upstash Redis and assembles a structured prompt containing the retrieved document chunks, historical conversation logs, and the active query.
6.  **AI Response Generation:** The prompt is sent to the Groq LLM (Llama 3.3 70B), which generates a precise, point-wise answer grounded strictly in the retrieved official documents, complete with page citations.
7.  **Answer Display:** The Web UI displays the formatted text answer and plays a vocalized audio response using Text-to-Speech (TTS).

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
*   **Semantic Embeddings:** `BAAI/bge-base-en-v1.5` (via Hugging Face Serverless Inference API) — High-performance dense vector model.
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

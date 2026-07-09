# Multimodal RAG Based System for Information Retrieval from TIET Policy Documents (PolicyLens)

PolicyLens is a state-of-the-art, highly accurate, and secure multimodal information retrieval system designed to index and extract knowledge from Thapar Institute of Engineering and Technology (TIET) official academic policies, regulations, syllabus schemes, and student guidelines.

---

## 🚀 Live Demo / Website
> 🔗 **Website Link:** *[To be added later]*

---

## 🎥 Project Demo Video
> 📹 **Demo Video:** *[To be added later]*

---

## 💡 Why I Built This Project
Navigating through university guidelines, fee structures, credit requirements, and hostel rules across dozens of different PDFs is slow and frustrating for students and parents. Furthermore, commercial, general-purpose LLMs (like standard ChatGPT) suffer from **hallucinations** and lack access to the private, specific documents of TIET.

I built this project to act as a **zero-hallucination cognitive layer** for TIET. Grounded strictly in verified official source documents via Retrieval-Augmented Generation (RAG), PolicyLens provides point-wise, cited answers instantly, making academic rules and schemes easily accessible to everyone.

---

## 🛠️ Tech Stack & Libraries
*   **Backend:** ![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54) Python 3.10+, ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi) FastAPI, ![Uvicorn](https://img.shields.io/badge/Uvicorn-000000?style=flat&logo=gunicorn&logoColor=white) Uvicorn
*   **Frontend:** ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=flat&logo=html5&logoColor=white) HTML5, ![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=flat&logo=css3&logoColor=white) CSS3 (Vanilla), ![JavaScript](https://img.shields.io/badge/javascript-%23F7DF1E.svg?style=flat&logo=javascript&logoColor=black) JavaScript (Vanilla), ![Three.js](https://img.shields.io/badge/three.js-000000?style=flat&logo=three.js&logoColor=white) Three.js (WebGL Jellyfish Liquid Blob Mouse follower background)
*   **RAG Framework:** ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) LangChain (LCEL)
*   **Vector Database:** ![Qdrant](https://img.shields.io/badge/Qdrant-DF3A1A?style=flat&logo=qdrant&logoColor=white) Qdrant Cloud (Vector Database)
*   **Cache & Database:** ![Upstash Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white) Upstash Redis (Memory & Rate Limiting)
*   **Inference API:** ![Groq](https://img.shields.io/badge/Groq-f55a42?style=flat&logo=groq&logoColor=white) Groq API

---

## 📊 About the Data
The system is built to search and understand **100+ official TIET documents** (PDF format), including:
*   UG & PG Prospectus and Schemes
*   Tuition Fee Structures and Scholarship Handbooks
*   Academic Course Catalogs & Credit Requirements
*   Hostel Accommodation Rules & General Guidelines

> 🔒 **Data Privacy & Ingestion:** The actual PDF files are not included in this GitHub repository for storage and privacy reasons. However, you can automatically download and collect them using the provided `thapar_rag_scraper.py` script.

---

## 🎨 System Architecture & Workflow

![System Architecture](architecture.png)

1.  **User Input:** The user provides query input in the form of **Text**, **Voice**, or **Image** (e.g. screenshot of a fee table or syllabus scheme) through the Frontend Web UI.
2.  **FastAPI Backend processing:**
    *   **Voice queries** are transcribed to text using Groq's Whisper Large v3 (STT).
    *   **Image queries** are processed by Groq's Llama 4 Scout Vision model to extract text details.
3.  **Embedding Generation:** The search text is converted into dense vector embeddings using the `BAAI/bge-base-en-v1.5` model.
4.  **Vector Retrieval:** The query vector is searched against **Qdrant Cloud** using MMR (Maximum Marginal Relevance) to retrieve the top-K relevant document chunks while avoiding redundant info.
5.  **Prompt & Context Construction:** The retrieved document contents and context source details are formatted along with historical chat memory loaded from **Upstash Redis**.
6.  **LLM Generation:** The prompt is sent to **Groq Llama 3.3 70B** to generate a point-wise response, strictly grounded in the document context.
7.  **Final Response:** The answer text is sent to the frontend, where it is displayed and optionally played as voice output via Text-to-Speech (gTTS).

---

## 📂 Repository Structure
```text
PolicyLens/
├── app/                     # Backend Source Code
│   ├── database.py          # Upstash Redis connections, rate limiting, and feedback DB
│   ├── main.py              # FastAPI endpoints, routing, and CORS setup
│   └── rag.py               # Core LangChain RAG pipeline, LLM routers, and prompts
├── frontend/                # SPA Static Web UI
│   ├── app.js               # Chat state, speech recognition, and WebGL Three.js canvas
│   ├── index.html           # Main Single Page App (Home, Docs, Chat Workspace)
│   ├── style.css            # Custom CSS theme, glassmorphism, and animations
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
*   **Fallback LLM:** `Qwen 2.5 7B` (via Hugging Face API) / `Llama 3.1 8B` (via Groq) — As automated high-availability backup chains.

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
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

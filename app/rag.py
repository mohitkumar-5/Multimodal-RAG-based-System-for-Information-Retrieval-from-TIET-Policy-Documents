import os
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import time
import base64
import numpy as np
import torch
from dotenv import load_dotenv

# LangChain and Groq imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Client/utility imports
from qdrant_client import QdrantClient
from groq import Groq
from gtts import gTTS

load_dotenv()

# --------------------------------------------------
# CONFIGURATION & KEY LOADS
# --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# Global Redis connection (single client) – use URL directly for LangChain history
redis_connection = None
if REDIS_URL:
    try:
        from langchain_community.chat_message_histories import RedisChatMessageHistory
        # Validate connection early
        test_history = RedisChatMessageHistory(session_id="test_conn", url=REDIS_URL, key_prefix="tiet_chat:")
        redis_connection = REDIS_URL  # Store URL string for later use
        print("[RAG] Redis connection URL validated successfully.")
    except Exception as e:
        print(f"[RAG] Failed to validate Redis URL: {e}. Falling back to in‑memory history.")
        redis_connection = None

QDRANT_URL = os.getenv("QDRANT_URL", "https://8d88d793-5447-4f29-b169-ebb8a17a1137.eu-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tiet_policy_docs")

VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
WHISPER_MODEL = "whisper-large-v3"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.82

# Check Hugging Face DNS reachability once at startup (prevents 12s hang on campus firewalls)
import socket
def check_hf_dns():
    try:
        socket.gethostbyname("router.huggingface.co")
        return True
    except Exception:
        return False
HF_AVAILABLE = check_hf_dns()
print(f"[RAG] Hugging Face Serverless API reachability status: {HF_AVAILABLE}")

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------
# Embedding Model setup (BAAI/bge-base-en-v1.5)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[RAG] Using device for HuggingFaceEmbeddings: {device}")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# Qdrant Vector Store
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
)

# Groq Client & Chat LLM
groq_client = Groq(api_key=GROQ_API_KEY)

# 1. Primary Groq LLM
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
    api_key=GROQ_API_KEY
)

# 2. Fallback HuggingFace Setup
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 3. Combined Runnable Router with Fallback (Triple Fail-Safe Router)
class ModelRouter:
    """Encapsulates primary and fallback LLM routing with exponential back‑off."""
    def __init__(self, primary_llm, hf_token: str | None, hf_available: bool):
        self.primary = primary_llm
        self.hf_token = hf_token
        self.hf_available = hf_available
        self.max_retries = 2

    def _call_hf(self, prompt_str: str):
        if not (self.hf_token and self.hf_available):
            return None
        hf_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
        data = {"model": hf_model, "messages": [{"role": "user", "content": prompt_str}], "temperature": 0.1, "max_tokens": 1024}
        try:
            import requests
            response = requests.post(url, headers=headers, json=data, timeout=30.0)
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                from langchain_core.messages import AIMessage
                return AIMessage(content=answer)
            else:
                print(f"[RAG] HF fallback API error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[RAG] HF fallback error: {e}")
        return None

    def _call_groq_8b(self, prompt):
        try:
            groq_8b = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=1024, api_key=GROQ_API_KEY)
            res = groq_8b.invoke(prompt)
            print("[RAG] Groq 8B fallback succeeded.")
            return res
        except Exception as e:
            print(f"[RAG] Groq 8B fallback failed: {e}")
            return None

    def invoke(self, prompt):
        # Primary 70B
        attempt = 0
        while attempt <= self.max_retries:
            try:
                return self.primary.invoke(prompt)
            except Exception as e:
                print(f"[RAG] Primary 70B failed (attempt {attempt+1}): {e}")
                # If rate‑limit or network, break to fallback
                break
        # HF fallback
        prompt_str = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        hf_res = self._call_hf(prompt_str)
        if hf_res:
            return hf_res
        # Groq 8B fallback
        groq_res = self._call_groq_8b(prompt)
        if groq_res:
            return groq_res
        raise RuntimeError("All LLM backends failed.")

# Instantiate router
model_router = ModelRouter(primary_llm=groq_llm, hf_token=hf_token, hf_available=HF_AVAILABLE)

llm = RunnableLambda(model_router.invoke)

# Vision Model
vision_llm = ChatGroq(
    model=VISION_MODEL,
    temperature=0.1,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

# --------------------------------------------------
# CHAT MEMORY PERSISTENT ROUTING (Redis / In-Memory)
# --------------------------------------------------
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """In‑memory fallback for chat session histories when Redis is unavailable."""
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

in_memory_histories = {}

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Return a RedisChatMessageHistory if a valid REDIS_URL is set, otherwise fallback to in‑memory."""
    if redis_connection:
        try:
            return RedisChatMessageHistory(
                session_id=session_id,
                url=redis_connection,
                key_prefix="tiet_chat:"
            )
        except Exception as e:
            print(f"[RAG] Redis history init failed: {e}. Falling back to in‑memory.")

    # Fallback in‑memory history (singleton per session)
    if session_id not in in_memory_histories:
        in_memory_histories[session_id] = InMemoryChatMessageHistory()
    return in_memory_histories[session_id]

# --------------------------------------------------
# RETRIEVAL PIPELINE – simplified with MMR retriever and optional re‑ranking
# --------------------------------------------------
from langchain_core.documents import Document

# Optional cross‑encoder reranker (downloaded lazily)
def _load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    except Exception as e:
        print(f"[RAG] Cross‑encoder not available: {e}")
        return None

CROSS_ENCODER = _load_cross_encoder()

def retrieve_documents(query: str, top_k: int = 10) -> list[Document]:
    """Retrieve relevant documents using MMR and optional cross‑encoder re‑ranking.
    Returns a list of Document objects ready for context formatting.
    """
    # Use MMR (Maximum Marginal Relevance) to increase diversity
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "lambda_mult": 0.5})
    initial_docs = retriever.invoke(query)

    # If a cross‑encoder is available, perform re‑ranking based on relevance to the query
    if CROSS_ENCODER:
        try:
            pairs = [(query, doc.page_content) for doc in initial_docs]
            scores = CROSS_ENCODER.predict(pairs)
            # Sort docs by descending score
            ranked = [doc for _, doc in sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)]
            return ranked[:top_k]
        except Exception as e:
            print(f"[RAG] Re‑ranking failed: {e}")

    return initial_docs[:top_k]

# Helper to embed query with BGE prefix (kept for consistency)
def embed_query_with_prefix(query: str):
    prefixed = "Represent this sentence for searching relevant passages: " + query
    return embedding_model.embed_query(prefixed)

# Old stitching function retained for backward compatibility (disabled by default)
def retrieve_with_stitching(query: str):
    print("[RAG] retrieve_with_stitching is deprecated – using retrieve_documents instead.")
    return retrieve_documents(query)

def format_context(docs) -> str:
    """Formats retrieved chunks with citations. Enforces a global token/character budget."""
    blocks = []
    current_char_count = 0
    # Global context limit (shared across the app)
    MAX_CONTEXT_CHARS = 16000
    for doc in docs:
        source_tag = f"[Source File: {doc.metadata.get('filename')}, Page: {doc.metadata.get('page')}]"
        block = f"{source_tag}\n{doc.page_content}"
        if current_char_count + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - current_char_count
            if remaining > 100:
                blocks.append(block[:remaining] + "\n... [Context truncated] ...")
            break
        blocks.append(block)
        current_char_count += len(block)
    return "\n\n---\n\n".join(blocks)

# --------------------------------------------------
# LLM CHAINS SETUP
# --------------------------------------------------
SYSTEM_PROMPT = """
You are PolicyLens, a reliable AI assistant for Thapar Institute of Engineering and Technology (TIET), Patiala.

Your primary purpose is to answer questions related to TIET academic schemes, policies, regulations, courses, fees, procedures, and official documents.

Follow these rules strictly:

1. DOCUMENT-FIRST ANSWERING:
- Always check the provided retrieved context first.
- Use information from the retrieved TIET documents as the primary source.
- Never modify, assume, or invent information from the documents.
- When using document information, include citations in this format at the end of the sentence or block:
  (Source: filename, Page X)
- Never use "Document X" or "Source File X" in your citations. Always reference the actual filename, e.g. (Source: policy_doc.pdf, Page 3).
- **Split Multi-Source Citations**: If you cite multiple different files for a fact, write them as separate parenthetical blocks, for example:
  Use "(Source: file1.pdf, Page 2) (Source: file2.pdf, Page 3)" instead of combining them into a single parenthesis.

2. SPECIFICITY, ACCURACY & CORRECTNESS:
- **Do not generalize program criteria**: If the retrieved documents contain information about specific programs or categories (e.g. specifically for PhD, MCA, MSc, or Bachelor), always state the specific program name in your answer. Do not present specific program rules as general TIET rules. If a general query is asked, list the criteria for all programs present in the context separately and clearly.
- Your primary goal is accuracy. Repeat all specific numbers (fees, dates, percentages, credits) exactly. Do not calculate or estimate values.
- For course schemes, list the courses, codes, credits, and semester details exactly as shown in the text.

3. WHEN INFORMATION IS FOUND:
- Give a clear, structured, and point-wise list answer based only on the retrieved context.
- Include relevant document citations.
- For tables, dates, credits, fees, eligibility criteria, and numbers, reproduce exactly as written.

4. WHEN INFORMATION IS NOT FOUND:
If the answer cannot be found in the retrieved TIET documents:
- Clearly state:
  "I could not find this information in the available TIET documents."
- Do not pretend that the answer came from official documents.
- Then, if appropriate, provide a general answer using your broader knowledge.
- Clearly label it:
  "Note: The following information is based on general knowledge and may not exactly match TIET's current rules. Please verify with official TIET sources."

5. PARTIAL INFORMATION:
- If only part of the answer is available in the documents:
  - Answer the supported part with citations.
  - Clearly mention which part was not available.

6. CONFLICTING DOCUMENTS:
- If different documents contain different information:
  - Mention the conflict.
  - Provide the sources where the differences appear.
  - Do not choose one answer without explanation.

7. TABLE AND STRUCTURED DATA:
- Preserve tables, lists, course codes, credits, semester details, and numerical information accurately.
- Do not summarize away important details.

8. HALLUCINATION CONTROL:
- Never create fake policies, rules, dates, fees, regulations, course structures, or official decisions.
- If uncertain, say so.

9. SCOPE:
- If the question is unrelated to TIET academics or policies:
  politely say:
  "I am designed to assist with TIET academic and policy-related queries."

10. RESPONSE STYLE & FORMATTING (STRICT POINT-WISE):
- Always present answers in a clean, concise, point-wise list format using bullet points or numbered lists.
- Avoid writing long, dense blocks of text or paragraphs.
- Keep every point short, punchy, and directly informative.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rag_chain = (
    {
        "context": lambda x: format_context(retrieve_documents(x["question"])) ,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

rewrite_prompt = ChatPromptTemplate.from_template(
    """Given the previous question and answer, and a new follow-up question,
rewrite the new question as a standalone question that includes any necessary
context from the previous exchange. If the new question is already standalone
and doesn't need the previous context, just return it unchanged.

Previous Question: {prev_question}
Previous Answer: {prev_answer}

New Question: {new_question}

Standalone Question:"""
)

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# --------------------------------------------------
# CORE CONVERSATIONAL INTERFACES
# --------------------------------------------------
def cosine_similarity(vec1, vec2) -> float:
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def is_followup_question(new_question: str, prev_question: str) -> bool:
    """Checks if new_question is topically related to prev_question via embedding similarity."""
    try:
        new_vec = embedding_model.embed_query(new_question)
        prev_vec = embedding_model.embed_query(prev_question)
        similarity = cosine_similarity(new_vec, prev_vec)
        return similarity >= RELEVANCE_THRESHOLD
    except Exception as e:
        print(f"[RAG] Error checking follow-up: {e}")
        return False

def ask(question: str, session_id: str = "default") -> str:
    """
    Main entry point for asking a question with history/memory.
    - Connects to session history in Redis/in-memory.
    - Rewrites query if related to previous turn.
    - Feeds recent conversation turns directly to the prompt template.
    - Stores outcome in history.
    """
    # Safe connection fallback: errors in Redis will not crash the QA route
    history = None
    messages = []
    try:
        history = get_chat_history(session_id)
        messages = history.messages
    except Exception as history_err:
        print(f"[Session {session_id}] History load error: {history_err}. Running without persistent memory.")

    prev_human = None
    prev_ai = None

    # Retrieve last conversation turn (Human + AI)
    for msg in reversed(messages):
        if msg.type == "ai" and prev_ai is None:
            prev_ai = msg.content
        elif msg.type == "human" and prev_human is None and prev_ai is not None:
            prev_human = msg.content
            break

    final_question = question

    if prev_human and prev_ai:
        if is_followup_question(question, prev_human):
            try:
                final_question = rewrite_chain.invoke({
                    "prev_question": prev_human,
                    "prev_answer": prev_ai,
                    "new_question": question,
                })
                print(f"[Session {session_id}] Linked to previous context. Rewritten: {final_question}")
            except Exception as e:
                print(f"[Session {session_id}] Error in query rewriting: {e}")
        else:
            print(f"[Session {session_id}] New topic detected.")

    # Limit history memory in prompt to last 6 messages (3 turns) to keep context tokens small
    chat_history_messages = messages[-6:] if len(messages) > 6 else messages

    # Call RAG with history context
    answer = rag_chain.invoke({
        "question": final_question,
        "chat_history": chat_history_messages
    })

    # Save to history safely
    if history:
        try:
            history.add_user_message(question)
            history.add_ai_message(answer)
            # Memory capping for in-memory mode
            if not redis_connection and len(history.messages) > 40:
                history.messages = history.messages[-40:]
        except Exception as save_err:
            print(f"[Session {session_id}] History save failed: {save_err}")

    return answer

# --------------------------------------------------
# MULTIMODAL FEATURES (VISION & VOICE)
# --------------------------------------------------
def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def extract_topic_from_image_bytes(image_bytes: bytes) -> str:
    """Uses Groq Vision model to describe the topic or extract the question in the image."""
    base64_image = encode_image(image_bytes)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Look at this image. If it contains an explicit question, "
                    "extract that question exactly. Otherwise, if it shows a "
                    "topic, policy excerpt, or piece of text/content, summarize "
                    "what topic or subject this is about in one or two clear "
                    "sentences, so it can be used as a search query. "
                    "Respond with ONLY the question or topic summary, nothing else."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    response = vision_llm.invoke([message])
    return response.content.strip()

def ask_with_image(image_bytes: bytes, session_id: str = "default") -> dict:
    """Extracts topic from image, runs RAG ask, returns question and answer."""
    extracted_query = extract_topic_from_image_bytes(image_bytes)
    print(f"[Session {session_id}] Image Query Extracted: {extracted_query}")
    answer = ask(extracted_query, session_id=session_id)
    return {
        "extracted_query": extracted_query,
        "answer": answer,
    }

def speech_to_text(audio_file_path: str) -> str:
    """Transcribes audio file to text using Groq Whisper. Cleans up file afterwards."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    with open(audio_file_path, "rb") as audio_file:
        transcription = groq_client.audio.transcriptions.create(
            file=(os.path.basename(audio_file_path), audio_file.read()),
            model=WHISPER_MODEL,
            response_format="text"
        )

    # Clean up temp file
    try:
        os.remove(audio_file_path)
    except OSError:
        pass

    return transcription.strip()

def text_to_speech_bytes(text: str) -> bytes:
    """Generates MP3 speech audio from text entirely in memory."""
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()

def ask_with_voice(audio_file_path: str, session_id: str = "default") -> dict:
    """Full voice loop: STT -> RAG ask -> TTS. Deletes audio upload automatically."""
    transcribed_question = speech_to_text(audio_file_path)
    print(f"[Session {session_id}] Transcribed Question: {transcribed_question}")
    
    answer_text = ask(transcribed_question, session_id=session_id)
    answer_audio_bytes = text_to_speech_bytes(answer_text)

    return {
        "transcribed_question": transcribed_question,
        "answer_text": answer_text,
        "answer_audio_bytes": answer_audio_bytes,
    }

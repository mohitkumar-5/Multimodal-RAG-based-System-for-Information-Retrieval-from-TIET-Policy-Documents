import os
import io
import sys
import time
import base64
from typing import List
from dotenv import load_dotenv

# Ensure utf-8 output streams
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# LangChain and Groq imports
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Client/utility imports
from qdrant_client import QdrantClient
from groq import Groq
from gtts import gTTS
import requests

load_dotenv()

# --------------------------------------------------
# CONFIGURATION & KEY LOADS
# --------------------------------------------------
def get_env_safe(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    if val:
        return val.strip().strip("'\"")
    return default

GROQ_API_KEY = get_env_safe("GROQ_API_KEY")
REDIS_URL = get_env_safe("REDIS_URL") or None

# Global Redis connection validation
redis_connection = None
if REDIS_URL:
    try:
        # Validate connection early
        test_history = RedisChatMessageHistory(session_id="test_conn", url=REDIS_URL, key_prefix="tiet_chat:")
        redis_connection = REDIS_URL
        print("[RAG] Redis connection URL validated successfully.")
    except Exception as e:
        print(f"[RAG] Failed to validate Redis URL: {e}. Falling back to in‑memory history.")
        redis_connection = None

QDRANT_URL = get_env_safe("QDRANT_URL", "https://8d88d793-5447-4f29-b169-ebb8a17a1137.eu-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = get_env_safe("QDRANT_API_KEY")
COLLECTION_NAME = get_env_safe("COLLECTION_NAME", "tiet_policy_docs")

# Model configuration
PRIMARY_TEXT_MODEL = get_env_safe("PRIMARY_TEXT_MODEL", "openai/gpt-oss-120b")
FALLBACK_TEXT_MODEL = get_env_safe("FALLBACK_TEXT_MODEL", "qwen/qwen3-32b")
PRIMARY_VISION_MODEL = get_env_safe("PRIMARY_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
FALLBACK_VISION_MODEL = get_env_safe("FALLBACK_VISION_MODEL", "qwen/qwen3.6-27b")
WHISPER_MODEL = "whisper-large-v3"

TOP_K = 5

# Check Hugging Face DNS reachability
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
# EMBEDDINGS (Memory optimized via HF Serverless API)
# --------------------------------------------------
hf_token = get_env_safe("HF_TOKEN") or get_env_safe("HUGGINGFACEHUB_API_TOKEN") or None

class HuggingFaceAPIEmbeddings(Embeddings):
    """Custom API-based embeddings wrapper to avoid loading PyTorch / BGE locally (saving 1.5GB RAM)."""
    def __init__(self, model_name: str, token: str | None = None):
        self.model_name = model_name
        self.token = token
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _call_api(self, inputs):
        for attempt in range(5):
            try:
                response = requests.post(self.api_url, headers=self.headers, json={"inputs": inputs}, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code in (503, 429, 500, 502, 504):
                    # Exponential backoff for temporary errors and rate limits
                    wait_time = 2 ** attempt
                    if response.status_code == 503:
                        try:
                            estimated_time = response.json().get("estimated_time", 5)
                            wait_time = min(estimated_time, 10)
                        except Exception:
                            pass
                    print(f"[RAG] HF API transient status {response.status_code}. Waiting {wait_time}s (Attempt {attempt+1}/5)...")
                    time.sleep(wait_time)
                else:
                    print(f"[RAG] HF API Permanent Error status {response.status_code}: {response.text}")
                    break
            except Exception as e:
                print(f"[RAG] HF API connection error: {e}")
                time.sleep(2)
        return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self._call_api(texts)
        if res is not None:
            return res
        return [[0.0] * 768 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        prefixed = "Represent this sentence for searching relevant passages: " + text
        res = self._call_api([prefixed])
        if res is not None:
            return res[0]
        return [0.0] * 768

print("[RAG] Initializing Serverless Hugging Face API Embeddings...")
embedding_model = HuggingFaceAPIEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    token=hf_token
)

GROQ_API_KEY_2 = get_env_safe("GROQ_API_KEY_2") or get_env_safe("GROQ_API_KEY_FALLBACK") or GROQ_API_KEY

# Qdrant Vector Store setup with Cloud + Local Fallback
local_db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qdrant_db")

client = None
vector_store = None

if QDRANT_URL and QDRANT_API_KEY:
    try:
        remote_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5)
        remote_client.get_collection(COLLECTION_NAME)
        client = remote_client
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model,
        )
        print(f"[RAG] Successfully connected to Qdrant Cloud cluster: {QDRANT_URL}")
    except Exception as cloud_err:
        print(f"[RAG] Qdrant Cloud unavailable ({cloud_err}). Switching to local Qdrant embedded DB...")

if vector_store is None:
    print(f"[RAG] Initializing local Qdrant embedded database from: {local_db_dir}")
    client = QdrantClient(path=local_db_dir)
    try:
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            print(f"[RAG] Creating local collection '{COLLECTION_NAME}'...")
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
    except Exception as create_err:
        print(f"[RAG] Error checking/creating collection: {create_err}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

# Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------
# LLM ROUTING
# --------------------------------------------------
class ModelRouter:
    """Encapsulates primary and fallback LLM routing with failover logic and multi-key support."""
    def __init__(self, primary_model: str, fallback_model: str, api_key: str, fallback_api_key: str = None):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        fb_key = fallback_api_key or api_key
        
        self.primary_llm = ChatGroq(
            model=primary_model,
            temperature=0.1,
            max_tokens=1024,
            api_key=api_key
        )
        self.fallback_llm = ChatGroq(
            model=fallback_model,
            temperature=0.1,
            max_tokens=1024,
            api_key=fb_key
        )

    def invoke(self, prompt):
        try:
            return self.primary_llm.invoke(prompt)
        except Exception as e:
            print(f"[RAG] Primary LLM ({self.primary_model}) failed: {e}. Falling back to ({self.fallback_model}).")
            try:
                return self.fallback_llm.invoke(prompt)
            except Exception as fallback_err:
                print(f"[RAG] Fallback LLM ({self.fallback_model}) failed: {fallback_err}")
                raise fallback_err

# Instantiate model router for text/reasoning tasks
model_router = ModelRouter(
    primary_model=PRIMARY_TEXT_MODEL,
    fallback_model=FALLBACK_TEXT_MODEL,
    api_key=GROQ_API_KEY,
    fallback_api_key=GROQ_API_KEY_2
)

llm = RunnableLambda(model_router.invoke)

# Primary & Fallback Vision Models
primary_vision_llm = ChatGroq(
    model=PRIMARY_VISION_MODEL,
    temperature=0.1,
    max_tokens=512,
    api_key=GROQ_API_KEY
)
fallback_vision_llm = ChatGroq(
    model=FALLBACK_VISION_MODEL,
    temperature=0.1,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

# --------------------------------------------------
# CHAT MEMORY ROUTING (Redis / In-Memory)
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

    if session_id not in in_memory_histories:
        in_memory_histories[session_id] = InMemoryChatMessageHistory()
    return in_memory_histories[session_id]

# --------------------------------------------------
# RETRIEVAL PIPELINE (Memory Optimized)
# --------------------------------------------------
from langchain_core.documents import Document

def retrieve_documents(query: str, top_k: int = TOP_K) -> list[Document]:
    """Retrieve relevant documents using MMR. Uses 0MB local RAM."""
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "lambda_mult": 0.5})
    return retriever.invoke(query)

def format_context(docs) -> str:
    """Formats retrieved chunks with citations. Enforces a global character budget."""
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
        "context": lambda x: format_context(retrieve_documents(x["question"])),
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
def ask(question: str, session_id: str = "default") -> str:
    """
    Main entry point for asking a question with history/memory.
    - Connects to session history in Redis/in-memory.
    - Rewrites query if related to previous turn using the LLM directly.
    - Feeds recent conversation turns directly to the prompt template.
    - Stores outcome in history.
    """
    history = None
    messages = []
    try:
        history = get_chat_history(session_id)
        messages = history.messages
    except Exception as history_err:
        print(f"[Session {session_id}] History load error: {history_err}. Running without persistent memory.")

    final_question = question

    # If chat history exists, attempt to rewrite the query contextually
    if messages:
        prev_human = None
        prev_ai = None
        # Retrieve the last exchange (Human + AI)
        for msg in reversed(messages):
            if msg.type == "ai" and prev_ai is None:
                prev_ai = msg.content
            elif msg.type == "human" and prev_human is None and prev_ai is not None:
                prev_human = msg.content
                break

        if prev_human and prev_ai:
            try:
                # LLM-based standalone rewriting: no expensive embeddings or threshold limits
                rewritten = rewrite_chain.invoke({
                    "prev_question": prev_human,
                    "prev_answer": prev_ai,
                    "new_question": question,
                }).strip()
                
                # Strip quotes if the LLM output is wrapped
                if rewritten.startswith('"') and rewritten.endswith('"'):
                    rewritten = rewritten[1:-1]
                
                if rewritten:
                    final_question = rewritten
                    print(f"[Session {session_id}] Standalone rewritten query: {final_question}")
            except Exception as e:
                print(f"[Session {session_id}] Error in query rewriting: {e}")

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
    """Uses Groq Vision model to describe the topic or extract the question in the image with fallback."""
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

    try:
        response = primary_vision_llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        print(f"[RAG] Primary vision model ({PRIMARY_VISION_MODEL}) failed: {e}. Trying fallback...")
        try:
            response = fallback_vision_llm.invoke([message])
            return response.content.strip()
        except Exception as fallback_err:
            print(f"[RAG] Fallback vision model ({FALLBACK_VISION_MODEL}) failed: {fallback_err}")
            raise fallback_err

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

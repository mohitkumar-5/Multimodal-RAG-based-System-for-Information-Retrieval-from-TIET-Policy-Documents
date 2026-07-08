import os
import uuid
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64

# Import internal modules
from app.database import check_rate_limit, save_feedback, update_feedback, get_feedback_summary
from app.rag import ask, ask_with_image, ask_with_voice

app = FastAPI(
    title="TIET Policy MultiModal RAG API",
    description="Backend API for TIET Academic Policy QA with Text, Image, and Voice processing.",
    version="1.0.0"
)

# Enable CORS for frontend deployment (e.g. on Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to allow specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# SCHEMAS
# --------------------------------------------------
class TextQueryRequest(BaseModel):
    question: str
    session_id: str = "default"

class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: str  # 'up' or 'down'

# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "TIET Policy RAG API is healthy."}


@app.post("/api/chat/text")
async def chat_text(payload: TextQueryRequest):
    session_id = payload.session_id or "default"
    
    # 1. Rate limiting check
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 10 requests per minute."
        )
        
    try:
        # 2. Call RAG Ask
        answer = ask(payload.question, session_id=session_id)
        
        # 3. Create feedback placeholder
        feedback_id = str(uuid.uuid4())
        save_feedback(feedback_id, session_id, payload.question, answer)
        
        return {
            "success": True,
            "answer": answer,
            "feedback_id": feedback_id
        }
    except Exception as e:
        print(f"[API] Error in text QA: {e}")
        return {
            "success": False,
            "error": "Failed to process your question. Please try again later.",
            "details": str(e)
        }


@app.post("/api/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    # 1. Rate limiting check
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 10 requests per minute."
        )
        
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image."
        )
        
    try:
        image_bytes = await file.read()
        
        # 2. Call Vision RAG
        result = ask_with_image(image_bytes, session_id=session_id)
        
        # 3. Save feedback placeholder
        feedback_id = str(uuid.uuid4())
        save_feedback(
            feedback_id, 
            session_id, 
            f"[Uploaded Image: Topic '{result['extracted_query']}']", 
            result['answer']
        )
        
        return {
            "success": True,
            "extracted_query": result["extracted_query"],
            "answer": result["answer"],
            "feedback_id": feedback_id
        }
    except Exception as e:
        print(f"[API] Error in image QA: {e}")
        return {
            "success": False,
            "error": "Failed to extract or analyze image. Please try again.",
            "details": str(e)
        }


@app.post("/api/chat/voice")
async def chat_voice(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    # 1. Rate limiting check
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 10 requests per minute."
        )
        
    # Validate file type (typically audio/*)
    if not file.content_type.startswith("audio/") and not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.webm', '.ogg')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid audio format."
        )
        
    # Write uploaded stream to temporary file on disk for Whisper library
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"tiet_voice_{uuid.uuid4()}{suffix}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Call Voice RAG Pipeline
        result = ask_with_voice(temp_path, session_id=session_id)
        
        # 3. Save feedback placeholder
        feedback_id = str(uuid.uuid4())
        save_feedback(
            feedback_id, 
            session_id, 
            f"[Spoken Audio: '{result['transcribed_question']}']", 
            result['answer_text']
        )
        
        # Base64 encode the generated MP3 TTS audio response
        audio_base64 = base64.b64encode(result["answer_audio_bytes"]).decode("utf-8")
        
        return {
            "success": True,
            "transcribed_question": result["transcribed_question"],
            "answer": result["answer_text"],
            "audio_base64": audio_base64,
            "feedback_id": feedback_id
        }
    except Exception as e:
        print(f"[API] Error in voice QA: {e}")
        # Make sure to clean up file if it still exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return {
            "success": False,
            "error": "Failed to process voice recording. Please speak clearly and try again.",
            "details": str(e)
        }


@app.post("/api/feedback")
def submit_feedback_api(payload: FeedbackRequest):
    success = update_feedback(payload.feedback_id, payload.rating)
    if success:
        return {"success": True, "message": "Feedback submitted successfully."}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback session ID not found."
        )


@app.get("/api/feedback/summary")
def get_feedback_summary_api():
    return get_feedback_summary()

# --------------------------------------------------
# MOUNT STATIC FRONTEND
# --------------------------------------------------
# Mount frontend folder to serve the static app locally (after endpoint routing)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
    print(f"[API] Static frontend mounted from: {frontend_dir}")
else:
    print(f"[API] Warning: Static frontend directory not found at: {frontend_dir}")

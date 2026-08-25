import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

# Test ModelRouter fallback mechanism
@patch('app.rag.ChatGroq')
def test_model_router_fallback(mock_chat_groq):
    # Two calls to ChatGroq in __init__ (primary and fallback)
    mock_primary = MagicMock()
    mock_fallback = MagicMock()
    mock_chat_groq.side_effect = [mock_primary, mock_fallback]
    
    # Mock primary LLM to fail, and fallback LLM to succeed
    mock_primary.invoke.side_effect = Exception("Groq Rate Limit")
    mock_fallback.invoke.return_value = AIMessage(content="Answer from fallback Qwen3")
    
    from app.rag import ModelRouter
    router = ModelRouter(primary_model="openai/gpt-oss-120b", fallback_model="qwen/qwen3-32b", api_key="dummy")
    
    res = router.invoke("Test prompt")
    assert res.content == "Answer from fallback Qwen3"
    
    mock_primary.invoke.assert_called_once_with("Test prompt")
    mock_fallback.invoke.assert_called_once_with("Test prompt")

# Test ModelRouter raising exception if both fail
@patch('app.rag.ChatGroq')
def test_model_router_both_fail(mock_chat_groq):
    mock_primary = MagicMock()
    mock_fallback = MagicMock()
    mock_chat_groq.side_effect = [mock_primary, mock_fallback]
    
    mock_primary.invoke.side_effect = Exception("Primary failed")
    mock_fallback.invoke.side_effect = Exception("Fallback failed")
    
    from app.rag import ModelRouter
    router = ModelRouter(primary_model="openai/gpt-oss-120b", fallback_model="qwen/qwen3-32b", api_key="dummy")
    
    with pytest.raises(Exception) as exc_info:
        router.invoke("Test prompt")
    assert "Fallback failed" in str(exc_info.value)

# Test context formatting and character budget capping
def test_format_context():
    from app.rag import format_context
    
    docs = [
        Document(page_content="A" * 10000, metadata={"filename": "doc1.pdf", "page": 1}),
        Document(page_content="B" * 10000, metadata={"filename": "doc2.pdf", "page": 2}),
    ]
    
    formatted = format_context(docs)
    # The limit is 16000 chars. So doc1 (approx 10050 chars) fits, but doc2 (approx 10050 chars) will exceed it
    # Therefore doc2 should be truncated or skipped.
    assert len(formatted) <= 16500
    assert "doc1.pdf" in formatted
    assert "[Context truncated]" in formatted

# Test API health check endpoint
def test_health_check():
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

# Test ModelRouter fallback mechanism
def test_model_router_fallback():
    from app.rag import ModelRouter
    
    # Mock primary LLM that fails
    primary_mock = MagicMock()
    primary_mock.invoke.side_effect = Exception("Groq 70B Rate Limit")
    
    # Instantiate ModelRouter with hf_token and hf_available as True
    router = ModelRouter(primary_llm=primary_mock, hf_token="test_token", hf_available=True)
    
    # Mock _call_hf to succeed
    hf_response = AIMessage(content="Answer from Hugging Face")
    with patch.object(router, '_call_hf', return_value=hf_response) as mock_hf:
        res = router.invoke("Test prompt")
        assert res.content == "Answer from Hugging Face"
        mock_hf.assert_called_once()

# Test ModelRouter fallback to Groq 8B if HF also fails
def test_model_router_fallback_to_groq_8b():
    from app.rag import ModelRouter
    
    primary_mock = MagicMock()
    primary_mock.invoke.side_effect = Exception("Groq 70B Rate Limit")
    
    router = ModelRouter(primary_llm=primary_mock, hf_token="test_token", hf_available=True)
    
    # Mock _call_hf to fail/return None and _call_groq_8b to succeed
    groq_8b_response = AIMessage(content="Answer from Groq 8B")
    with patch.object(router, '_call_hf', return_value=None), \
         patch.object(router, '_call_groq_8b', return_value=groq_8b_response) as mock_8b:
        res = router.invoke("Test prompt")
        assert res.content == "Answer from Groq 8B"
        mock_8b.assert_called_once()

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

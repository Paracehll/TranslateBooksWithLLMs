import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.core.translator import refine_chunks
from src.core.epub.xhtml_translator import _refine_epub_chunks
from src.core.subtitle_translator import refine_subtitle_translations
from src.core.llm.base import LLMResponse

@pytest.mark.asyncio
async def test_refine_chunks_zero_retries_fails_immediately():
    """Test max_refinement_retries=0: when refinement returns None, fail after initial attempt."""
    mock_client = AsyncMock()
    mock_client.make_request.return_value = None  # fails
    mock_client.context_window = 2048

    translated_chunks = ["Draft chunk text"]
    original_chunks = [{"context_before": "", "context_after": ""}]

    logs = []
    def log_callback(event, msg, data=None):
        logs.append((event, msg))

    with patch("src.core.translator.create_llm_client", return_value=mock_client):
        result = await refine_chunks(
            translated_chunks=translated_chunks,
            original_chunks=original_chunks,
            target_language="English",
            model_name="test-model",
            api_endpoint="http://localhost:11434",
            log_callback=log_callback,
            llm_provider="ollama",
            max_refinement_retries=0,
        )

    # Output should fallback to original draft text
    assert result == ["Draft chunk text"]
    # make_request called exactly 1 time
    assert mock_client.make_request.call_count == 1


@pytest.mark.asyncio
async def test_refine_chunks_n_retries_recovers():
    """Test max_refinement_retries=2: fails on attempt 1, succeeds on attempt 2."""
    mock_client = AsyncMock()

    resp_fail = None
    resp_success = LLMResponse(
        content="Refined chunk text",
        prompt_tokens=10,
        completion_tokens=10,
        context_used=20,
        context_limit=4096,
    )
    mock_client.make_request.side_effect = [resp_fail, resp_success]
    mock_client.extract_translation = MagicMock(side_effect=lambda raw: raw)
    mock_client.context_window = 2048

    logs = []
    def log_callback(event, msg, data=None):
        logs.append((event, msg))

    with patch("src.core.translator.create_llm_client", return_value=mock_client):
        result = await refine_chunks(
            translated_chunks=["Draft chunk text"],
            original_chunks=[{"context_before": "", "context_after": ""}],
            target_language="English",
            model_name="test-model",
            api_endpoint="http://localhost:11434",
            log_callback=log_callback,
            llm_provider="ollama",
            max_refinement_retries=2,
        )

    assert result == ["Refined chunk text"]
    assert mock_client.make_request.call_count == 2
    retry_logs = [m for event, m in logs if event == "refinement_retry" or "Retrying" in m]
    assert len(retry_logs) == 1


@pytest.mark.asyncio
async def test_refine_chunks_infinite_retries_eventually_succeeds():
    """Test max_refinement_retries=-1: retries until success."""
    mock_client = AsyncMock()

    resp_success = LLMResponse(
        content="Refined chunk text",
        prompt_tokens=10,
        completion_tokens=10,
        context_used=20,
        context_limit=4096,
    )
    # Fails 3 times, succeeds on 4th call
    mock_client.make_request.side_effect = [None, None, None, resp_success]
    mock_client.extract_translation = MagicMock(side_effect=lambda raw: raw)
    mock_client.context_window = 2048

    logs = []
    def log_callback(event, msg, data=None):
        logs.append((event, msg))

    with patch("src.core.translator.create_llm_client", return_value=mock_client):
        result = await refine_chunks(
            translated_chunks=["Draft chunk text"],
            original_chunks=[{"context_before": "", "context_after": ""}],
            target_language="English",
            model_name="test-model",
            api_endpoint="http://localhost:11434",
            log_callback=log_callback,
            llm_provider="ollama",
            max_refinement_retries=-1,
        )

    assert result == ["Refined chunk text"]
    assert mock_client.make_request.call_count == 4


@pytest.mark.asyncio
async def test_epub_refinement_retries_and_fallback():
    """Test EPUB chunk refinement retries."""
    mock_client = AsyncMock()
    # Fails twice, succeeds 3rd
    resp_success = LLMResponse(
        content="Refined EPUB chunk",
        prompt_tokens=10,
        completion_tokens=10,
        context_used=20,
        context_limit=4096,
    )
    mock_client.make_request.side_effect = [None, None, resp_success]
    mock_client.extract_translation = MagicMock(side_effect=lambda raw: raw)
    mock_client.context_window = 2048

    logs = []
    def log_callback(event, msg, data=None):
        logs.append((event, msg))

    translated_chunks = ["Draft EPUB chunk"]
    chunks = [{"global_indices": [], "local_tag_map": {}}]

    result = await _refine_epub_chunks(
        translated_chunks=translated_chunks,
        chunks=chunks,
        target_language="English",
        model_name="test-model",
        llm_client=mock_client,
        context_manager=None,
        placeholder_format=("{{", "}}"),
        log_callback=log_callback,
        prompt_options={},
        max_refinement_retries=2
    )

    assert result == ["Refined EPUB chunk"]
    assert mock_client.make_request.call_count == 3


@pytest.mark.asyncio
async def test_srt_refinement_infinite_retries():
    """Test SRT refinement with infinite retries (-1)."""
    mock_client = AsyncMock()
    resp_success = LLMResponse(
        content="[0]Refined SRT line",
        prompt_tokens=10,
        completion_tokens=10,
        context_used=20,
        context_limit=4096,
    )
    # Fails twice (returns None), succeeds 3rd time
    mock_client.make_request.side_effect = [None, None, resp_success]
    mock_client.extract_translation = MagicMock(side_effect=lambda raw: raw)

    translations = {0: "Draft SRT line"}

    result = await refine_subtitle_translations(
        translations=translations,
        target_language="English",
        model_name="test-model",
        llm_client=mock_client,
        max_refinement_retries=-1
    )

    assert result == {0: "Refined SRT line"}
    assert mock_client.make_request.call_count == 3

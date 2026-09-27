"""
Unit tests for API key sanitization and endpoint normalization (issue handling UnicodeEncodeError).
"""
import pytest
from src.core.llm.base import sanitize_api_key, normalize_api_keys
from src.api.api_keys import resolve_api_key
from src.core.llm.providers.openai import OpenAICompatibleProvider


class TestApiKeySanitization:
    def test_sanitize_unicode_dashes(self):
        # Test various unicode dashes (en-dash \u2013, em-dash \u2014, minus sign \u2212)
        raw_key = "sk-or-v1-1234\u20135678\u20149012\u22123456"
        expected = "sk-or-v1-1234-5678-9012-3456"
        assert sanitize_api_key(raw_key) == expected

    def test_sanitize_smart_quotes_and_zero_width_spaces(self):
        # Test smart quotes, zero-width spaces (\u200b), BOM (\ufeff), non-breaking spaces (\u00a0)
        raw_key = "\ufeff\u200bsk-or-v1-'test'\u2018key\u2019-\u201cquoted\u201d\u00a0"
        expected = "sk-or-v1-'test''key'-\"quoted\""
        assert sanitize_api_key(raw_key) == expected

    def test_normalize_api_keys_sanitizes(self):
        raw_csv = "  sk-or-v1-key1\u2013abc  ,  \ufeffsk-or-v1-key2\u2014def  "
        normalized = normalize_api_keys(raw_csv)
        assert normalized == ["sk-or-v1-key1-abc", "sk-or-v1-key2-def"]

    def test_resolve_api_key_sanitizes(self):
        raw_key = "sk-or-v1-test\u2013key"
        resolved = resolve_api_key(raw_key, "OPENROUTER_API_KEY", "")
        assert resolved == "sk-or-v1-test-key"

    def test_endpoint_normalization_sanitizes_unicode_dashes(self):
        endpoint = "http://localhost:11434/v1\u2013test/v1"
        normalized = OpenAICompatibleProvider._normalize_endpoint(endpoint)
        assert normalized == "http://localhost:11434/v1-test/v1/chat/completions"

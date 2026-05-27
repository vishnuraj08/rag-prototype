# =============================================================================
# tests/test_generator.py — Tests for Generator (with mocked Anthropic API)
# =============================================================================
# Run with:  pytest tests/test_generator.py -v
#
# KEY CONCEPT — Mocking:
#   We do NOT make real API calls in tests. Real calls cost money, are slow,
#   need a real API key, and can fail due to network issues — all bad for tests.
#
#   Instead we use unittest.mock.patch to replace the real API call with a
#   fake one that we control. The fake always returns instantly with whatever
#   we tell it to return.
#
#   Think of it like this: we're testing YOUR code (the error handling, the
#   prompt building, the empty query guard) — not Anthropic's servers.
# =============================================================================

import pytest
from unittest.mock import patch, MagicMock
import anthropic

from components.generator import Generator


# Helper: build a fake API response that looks like what Anthropic returns
def make_fake_response(text="This is the answer."):
    """
    Create a mock response object that mimics anthropic.types.Message.
    The real response has: response.content[0].text and response.usage.
    """
    mock_content_block = MagicMock()
    mock_content_block.text = text

    mock_usage = MagicMock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 50

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    mock_response.usage = mock_usage
    return mock_response


class TestGeneratorGenerate:
    """Tests for the generate() method."""

    def test_returns_string(self):
        """generate() should always return a string."""
        with patch("components.generator.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = make_fake_response("Answer here.")
            gen = Generator()
            result = gen.generate("What is Python?", "Python is a language.")
            assert isinstance(result, str)

    def test_returns_correct_answer(self):
        """The text from the API response should be returned."""
        expected = "Python was created by Guido van Rossum."
        with patch("components.generator.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = make_fake_response(expected)
            gen = Generator()
            result = gen.generate("Who created Python?", "Context about Python.")
            assert result == expected

    def test_empty_query_returns_error_message(self):
        """
        An empty query string should be caught BEFORE hitting the API.
        No API call should be made — we check by asserting create() was never called.
        """
        with patch("components.generator.Anthropic") as MockAnthropic:
            gen = Generator()
            result = gen.generate("", "Some context.")
            assert "non-empty" in result.lower() or "please" in result.lower()
            MockAnthropic.return_value.messages.create.assert_not_called()

    def test_whitespace_only_query_rejected(self):
        """A query that is only spaces/tabs should also be rejected."""
        with patch("components.generator.Anthropic") as MockAnthropic:
            gen = Generator()
            result = gen.generate("   ", "Some context.")
            MockAnthropic.return_value.messages.create.assert_not_called()
            assert isinstance(result, str)

    def test_authentication_error_handled(self):
        """
        A bad API key should return a readable error string, not a traceback.
        """
        with patch("components.generator.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = (
                anthropic.AuthenticationError(
                    message="Invalid API key",
                    response=MagicMock(status_code=401),
                    body={}
                )
            )
            gen = Generator()
            result = gen.generate("What is Python?", "Context.")
            assert "api key" in result.lower() or "error" in result.lower()
            assert isinstance(result, str)   # must not raise

    def test_rate_limit_error_handled(self):
        """Rate limit errors should return a friendly string."""
        with patch("components.generator.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = (
                anthropic.RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body={}
                )
            )
            gen = Generator()
            result = gen.generate("What is Python?", "Context.")
            assert "rate limit" in result.lower() or "error" in result.lower()

    def test_empty_content_response_handled(self):
        """
        If the API returns an empty content list (rare but possible),
        we should get a readable message, not an IndexError.
        """
        with patch("components.generator.Anthropic") as MockAnthropic:
            mock_response = MagicMock()
            mock_response.content = []   # empty content list
            MockAnthropic.return_value.messages.create.return_value = mock_response
            gen = Generator()
            result = gen.generate("What is Python?", "Context.")
            assert isinstance(result, str)
            assert result != ""   # should return some message, not empty string


class TestBuildPrompt:
    """Tests for the _build_prompt() helper."""

    def test_prompt_contains_question(self):
        """The built prompt should include the question."""
        with patch("components.generator.Anthropic"):
            gen = Generator()
            prompt = gen._build_prompt("What is Python?", "Python is a language.")
            assert "What is Python?" in prompt

    def test_prompt_contains_context(self):
        """The built prompt should include the context."""
        with patch("components.generator.Anthropic"):
            gen = Generator()
            context = "Python was created in 1991."
            prompt = gen._build_prompt("When was Python created?", context)
            assert context in prompt

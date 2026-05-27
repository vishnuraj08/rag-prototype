# =============================================================================
# components/generator.py — STEP 6: GENERATOR
# =============================================================================
# PURPOSE: Take the user's query + retrieved context → generate a grounded answer.
#
# WHY THIS STEP EXISTS:
# We have the relevant context chunks. Now we need an LLM to READ them and
# ANSWER the user's question based only on that context.
#
# KEY CONCEPT — "Grounded" Generation:
# Without RAG: LLM answers from training data (can hallucinate, outdated)
# With RAG:    LLM answers from YOUR documents (grounded, up-to-date, auditable)
#
# SUPPORTED PROVIDERS (controlled by LLM_PROVIDER in config.py):
#
#   "anthropic" → calls Claude via Anthropic's API
#                 Requires: ANTHROPIC_API_KEY environment variable
#                 Models: claude-sonnet-4-6, claude-haiku-4-5-20251001, etc.
#
#   "ollama"    → calls a local model running on your machine via Ollama
#                 Requires: Ollama installed + `ollama serve` running
#                 No API key, no internet, completely private
#                 Models: llama3, mistral, phi3, gemma, etc.
#
# HOW TO SWITCH PROVIDERS:
#   Open config.py and change:
#     LLM_PROVIDER = "anthropic"   →   LLM_PROVIDER = "ollama"
#     LLM_MODEL = "claude-sonnet-4-6"   →   LLM_MODEL = "llama3"
#   That's it. Nothing else changes.
#
# DESIGN PATTERN — Strategy Pattern:
# The Generator class picks the right "strategy" (Anthropic or Ollama) at
# startup and stores it. The generate() method always looks the same to callers
# regardless of which provider is active underneath.
# =============================================================================

import logging
from config import LLM_PROVIDER, LLM_MODEL, MAX_TOKENS, SYSTEM_PROMPT, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class Generator:
    """
    Generates answers by calling an LLM with the user's question + retrieved context.

    Supports multiple backends (Anthropic, Ollama) — controlled by config.py.
    The public interface (generate() method) is identical regardless of backend.
    """

    def __init__(self):
        """
        Initialize the correct LLM client based on LLM_PROVIDER in config.py.

        This is the only place where provider-specific setup happens.
        Everything else in the class is provider-agnostic.
        """
        self.provider = LLM_PROVIDER
        logger.info(f"Generator initializing with provider='{LLM_PROVIDER}', model='{LLM_MODEL}'")

        if self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
                f"Valid options are: 'anthropic', 'ollama'"
            )

    # -------------------------------------------------------------------------
    # Provider initialisation — one method per provider
    # -------------------------------------------------------------------------

    def _init_anthropic(self):
        """Set up the Anthropic client. Reads ANTHROPIC_API_KEY from environment."""
        try:
            import anthropic
            self._anthropic = anthropic
            self.client = anthropic.Anthropic()
            # Test that the key exists (constructor doesn't validate it yet)
            logger.info("Anthropic client initialised successfully.")
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is not installed. "
                "Run: pip install anthropic"
            )

    def _init_ollama(self):
        """
        Set up the Ollama client.

        Ollama must be running on your machine:
          1. Install Ollama from https://ollama.com
          2. Run: ollama serve            (starts the local server)
          3. Run: ollama pull llama3      (downloads the model, one time)

        The Ollama API is OpenAI-compatible, so we use the openai package
        with a custom base_url pointing to localhost instead of OpenAI's servers.
        This avoids adding a separate 'ollama' package dependency.
        """
        try:
            from openai import OpenAI
            # Point the OpenAI client at your local Ollama server instead of api.openai.com
            # Ollama exposes an OpenAI-compatible REST API at /v1
            self.client = OpenAI(
                base_url=f"{OLLAMA_BASE_URL}/v1",
                api_key="ollama",     # Ollama doesn't need a real key — this is a placeholder
            )
            logger.info(f"Ollama client initialised. Server: {OLLAMA_BASE_URL}")
        except ImportError:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Run: pip install openai"
            )

    # -------------------------------------------------------------------------
    # Public interface — identical regardless of provider
    # -------------------------------------------------------------------------

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer to the query using the provided context.

        This method looks exactly the same to any caller (pipeline.py, tests, etc.)
        regardless of whether Anthropic or Ollama is being used underneath.

        Args:
            query   : The user's question (plain string)
            context : Formatted context string from the Retriever

        Returns:
            The LLM's response as a plain string
        """
        # Guard: refuse to call the API with an empty question
        if not query or not query.strip():
            logger.warning("generate() called with an empty query — skipping API call")
            return "Please provide a non-empty question."

        user_message = self._build_prompt(query, context)
        logger.info(f"Calling LLM ({self.provider} / {LLM_MODEL})...")

        if self.provider == "anthropic":
            return self._generate_anthropic(user_message)
        elif self.provider == "ollama":
            return self._generate_ollama(user_message)

    # -------------------------------------------------------------------------
    # Provider-specific generation — one method per provider
    # -------------------------------------------------------------------------

    def _generate_anthropic(self, user_message: str) -> str:
        """Call the Anthropic Messages API and return the answer text."""
        try:
            response = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
        except self._anthropic.AuthenticationError:
            logger.error("Invalid Anthropic API key.")
            return "Error: Invalid API key. Check your ANTHROPIC_API_KEY and try again."
        except self._anthropic.RateLimitError:
            logger.error("Anthropic rate limit hit.")
            return "Error: Rate limit reached. Please wait a moment and try again."
        except self._anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: API request failed — {str(e)}"

        if not response.content:
            return "I was unable to generate an answer. Please try again."

        answer = response.content[0].text
        logger.info(
            f"Response received. "
            f"Input tokens: {response.usage.input_tokens}, "
            f"Output tokens: {response.usage.output_tokens}"
        )
        return answer

    def _generate_ollama(self, user_message: str) -> str:
        """
        Call a local Ollama model via the OpenAI-compatible API.

        Ollama's API looks identical to OpenAI's — same JSON format, same fields.
        We just pointed the client at localhost instead of api.openai.com.
        """
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=MAX_TOKENS,
            )
        except Exception as e:
            # Common cause: Ollama server isn't running
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                return (
                    "Error: Cannot connect to Ollama. "
                    "Make sure Ollama is running: open a terminal and run `ollama serve`"
                )
            logger.error(f"Ollama error: {e}")
            return f"Error: Ollama request failed — {str(e)}"

        answer = response.choices[0].message.content
        logger.info(f"Ollama response received. Model: {LLM_MODEL}")
        return answer

    # -------------------------------------------------------------------------
    # Prompt building — shared by all providers
    # -------------------------------------------------------------------------

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Construct the prompt that combines context + question.

        This is the same regardless of which LLM you're talking to.
        All LLMs understand this format — context first, then question.
        """
        prompt = f"""Here is the relevant context from the documents:

{context}

Based ONLY on the context above, please answer the following question:

Question: {query}

If the answer cannot be found in the context, say so clearly."""

        return prompt

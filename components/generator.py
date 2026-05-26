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
# THE PROMPT STRUCTURE:
# We pass Claude a carefully crafted prompt:
#   System: "You are a helpful assistant. Only use the provided context."
#   User:   "CONTEXT: [retrieved chunks]
#            QUESTION: [user's question]"
#
# The system prompt sets the behavior.
# The context provides the knowledge.
# The question tells Claude what to answer.
#
# HOW TO EXTEND:
# - Add conversation history for multi-turn chat
# - Add citation extraction (ask Claude to cite which chunk it used)
# - Swap Claude for any other LLM by changing the API call
# =============================================================================

import logging
import anthropic
from anthropic import Anthropic
from config import CLAUDE_MODEL, MAX_TOKENS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Generator:
    """
    Calls Claude via the Anthropic API to generate answers grounded in context.
    """

    def __init__(self):
        """
        Initialize the Anthropic client.
        
        The Anthropic() client automatically reads ANTHROPIC_API_KEY
        from your environment variables. 
        Set it with: export ANTHROPIC_API_KEY="sk-ant-..."
        """
        # Anthropic() constructor looks for ANTHROPIC_API_KEY env variable
        # No need to hardcode the key — that would be a security risk!
        self.client = Anthropic()
        logger.info(f"Generator initialized with model: {CLAUDE_MODEL}")

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer to the query using the provided context.
        
        Args:
            query   : The user's question (plain string)
            context : Formatted context string from the Retriever
                      (contains the relevant chunks)
        
        Returns:
            Claude's response as a plain string
        """
        # Guard: refuse to call the API with an empty question
        if not query or not query.strip():
            logger.warning("generate() called with an empty query — skipping API call")
            return "Please provide a non-empty question."

        # Build the user message that combines context + question
        # This structure is commonly called a "RAG prompt template"
        user_message = self._build_prompt(query, context)

        logger.info("Calling Claude API...")

        # Wrap the API call in try/except so network or auth errors don't
        # crash the whole pipeline with an unhelpful Python traceback.
        try:
            # Call the Anthropic Messages API
            # This is a synchronous (blocking) call — it waits for the full response
            response = self.client.messages.create(
                model=CLAUDE_MODEL,         # Which Claude model to use (from config)
                max_tokens=MAX_TOKENS,      # Max length of the response
                system=SYSTEM_PROMPT,       # Sets Claude's behavior/role
                messages=[
                    {
                        "role": "user",     # "user" = the human's turn
                        "content": user_message
                    }
                ]
            )
        except anthropic.AuthenticationError:
            logger.error("Invalid API key — check your ANTHROPIC_API_KEY environment variable")
            return "Error: Invalid API key. Set ANTHROPIC_API_KEY and try again."
        except anthropic.RateLimitError:
            logger.error("Rate limit hit — too many requests in a short window")
            return "Error: Rate limit reached. Please wait a moment and try again."
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: API request failed — {str(e)}"

        # Guard: the API can technically return an empty content list (rare but possible)
        # Without this check, response.content[0] would raise IndexError
        if not response.content:
            logger.warning("Claude returned an empty response — no content blocks")
            return "I was unable to generate an answer. Please try again."

        # response.content is a list of content blocks
        # For text responses, we want the first block's text
        answer = response.content[0].text

        # Log token usage — useful for monitoring costs
        logger.info(
            f"Response received. "
            f"Input tokens: {response.usage.input_tokens}, "
            f"Output tokens: {response.usage.output_tokens}"
        )

        return answer

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Construct the prompt that combines context + question.
        
        This is where "prompt engineering" happens for RAG.
        The structure matters — Claude needs to know:
        1. What the context is
        2. What the question is
        3. That it should use the context to answer
        
        Args:
            query   : User's question
            context : Retrieved context chunks (already formatted)
            
        Returns:
            Complete prompt string for the LLM
        """
        # Triple-quoted strings in Python preserve newlines and formatting
        # This gives Claude a clean, structured prompt to work with
        prompt = f"""Here is the relevant context from the documents:

{context}

Based ONLY on the context above, please answer the following question:

Question: {query}

If the answer cannot be found in the context, say so clearly."""

        return prompt

"""
query_rewriter.py
-----------------
Query expansion / rewriting using Groq.

Improves retrieval by:
  1. Expanding abbreviations and jargon
  2. Adding related technical terms
  3. Generating sub-questions for complex queries
  4. Translating natural language to code-search terms

Example:
  Input:  "why is login broken?"
  Output: "authentication login function bug error exception user credentials"
"""

from typing import List

from groq import Groq
from loguru import logger


REWRITE_PROMPT = """You are a code search query optimizer. 
Given a user question about a codebase, rewrite it into a better search query.

Rules:
- Add relevant technical synonyms and related terms
- Expand abbreviations (auth → authentication, db → database, etc.)
- Add likely function names, class names, or variable names
- Keep the rewritten query concise (under 30 words)
- Output ONLY the rewritten query, nothing else

Examples:
  User: "why is login broken?"
  Rewritten: "authentication login function user credentials validation error exception bug"

  User: "how do I add a new user?"
  Rewritten: "create user registration signup function database insert new account"

  User: "what does the cache do?"
  Rewritten: "cache caching Redis memory store get set expire TTL invalidation"
"""


class QueryRewriter:
    """Uses an LLM to expand and improve retrieval queries."""

    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def rewrite(self, query: str) -> str:
        """Return an expanded version of the query for better retrieval."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=100,
                temperature=0.3,
            )
            rewritten = response.choices[0].message.content.strip()
            logger.debug(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return query

    def generate_subquestions(self, query: str, n: int = 3) -> List[str]:
        """
        Decompose a complex query into simpler sub-questions.
        Useful for multi-hop RAG.
        """
        prompt = (
            f"Break this complex question into {n} simpler sub-questions "
            f"that together answer the original.\n"
            f"Output one sub-question per line, no numbering.\n\n"
            f"Question: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4,
            )
            lines = response.choices[0].message.content.strip().splitlines()
            subqs = [l.strip("- ").strip() for l in lines if l.strip()][:n]
            logger.debug(f"Sub-questions for '{query}': {subqs}")
            return subqs
        except Exception as e:
            logger.warning(f"Sub-question generation failed: {e}")
            return [query]

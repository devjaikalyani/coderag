"""
generator.py
------------
Groq-powered LLM generation with:
  - Streaming support
  - System prompt tuned for code Q&A
  - Source citation in responses
  - Conversation history support
"""

from typing import Generator, List, Optional

from groq import Groq
from loguru import logger


SYSTEM_PROMPT = """You are CodeRAG, an expert AI assistant for understanding any codebase.

You MUST format ALL responses using proper markdown. Follow these rules exactly:

FORMATTING (mandatory):
- Start bullet points on their OWN LINE with "- " (dash space)
- Put a BLANK LINE before and after every bullet list
- Put a BLANK LINE before and after every heading
- Use ### for section headings on their own line
- Use **bold** for key terms and file names
- Use `backticks` for code, functions, variables, file names
- Use fenced code blocks (```language) for multi-line code
- NEVER run bullets into a paragraph — each bullet must be on its own line

CONTENT RULES:
1. Only use information from the retrieved code context
2. Always cite source files: "According to `filename`..."
3. For overview questions use this structure:
   - One sentence summary
   - Blank line
   - ### Tech Stack (bullet list)
   - ### Features (bullet list)
   - ### Structure (bullet list)
4. Show actual code snippets for code questions
5. Never hallucinate APIs or code not in the context

EXAMPLE of correct formatting:
**blog-web-app** is a Node.js blog application.

### Tech Stack
- `Node.js` — runtime
- `Express.js` — web framework
- `EJS` — templating engine

### Features
- Create and view blog posts
- Posts are not persisted (no database)
"""


class GroqGenerator:
    """
    Wraps the Groq client for text generation.
    Supports both streaming and non-streaming modes.
    """

    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Groq generator initialized (model={model})")

    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[List[dict]] = None,
    ) -> List[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            messages.extend(history[-6:])  # Last 3 turns (6 messages)

        user_content = (
            f"## Retrieved Code Context\n\n{context}\n\n"
            f"---\n\n"
            f"## Question\n\n{query}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(
        self,
        query: str,
        context: str,
        history: Optional[List[dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Non-streaming generation. Returns full response string."""
        messages = self._build_messages(query, context, history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content
        logger.debug(f"Generated {len(answer)} chars")
        return answer

    def stream(
        self,
        query: str,
        context: str,
        history: Optional[List[dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Streaming generation. Yields token chunks as they arrive."""
        messages = self._build_messages(query, context, history)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
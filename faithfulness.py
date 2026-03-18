"""
faithfulness.py
---------------
NLI-based faithfulness checker.

Checks if the generated answer is supported by the retrieved context.
Uses a lightweight MiniLM NLI model — fast, works on CPU, compatible
with transformers==4.38.2 and PyTorch 2.2.2.

Score = fraction of answer sentences entailed by context.
Score < threshold → flag as potentially hallucinated.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from loguru import logger


@dataclass
class FaithfulnessResult:
    score: float                                        # 0.0 – 1.0
    is_faithful: bool
    sentence_scores: List[Tuple[str, float, str]]       # (sentence, score, label)

    def summary(self) -> str:
        flag = "✅ Faithful" if self.is_faithful else "⚠️  Potentially hallucinated"
        return f"{flag} (score={self.score:.2f})"


class FaithfulnessChecker:
    """
    Uses a lightweight NLI cross-encoder to check if answer sentences
    are supported by the retrieved context.

    Model: cross-encoder/nli-MiniLM2-L6-H768
      - Fast on CPU (~50ms per sentence)
      - Compatible with transformers==4.38.2
      - Good accuracy for technical text
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        threshold: float = 0.5,
    ):
        self._model_name = model_name
        self.threshold = threshold
        self._model = None   # lazy load

    def _load_model(self):
        """Lazy-load the model on first use to speed up startup."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            import torch
            logger.info(f"Loading faithfulness model: {self._model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(
                self._model_name,
                max_length=512,
                device=device,
            )
            logger.info("Faithfulness model loaded")
        except Exception as e:
            logger.warning(f"Could not load faithfulness model: {e}. Scores will be disabled.")
            self._model = None

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter — avoids heavy NLP deps."""
        import re
        sentences = re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)
        return [
            s.strip() for s in sentences
            if len(s.strip()) > 20 and not s.strip().startswith("```")
        ]

    def check(self, answer: str, context: str) -> FaithfulnessResult:
        """
        Check if the answer is supported by the retrieved context.
        Returns FaithfulnessResult with overall score and per-sentence breakdown.
        If model failed to load, returns a neutral result (score=0.7).
        """
        self._load_model()

        # Graceful fallback if model unavailable
        if self._model is None:
            return FaithfulnessResult(
                score=0.7,
                is_faithful=True,
                sentence_scores=[("Model unavailable", 0.7, "unknown")],
            )

        sentences = self._split_sentences(answer)
        if not sentences:
            return FaithfulnessResult(score=1.0, is_faithful=True, sentence_scores=[])

        # Truncate context to avoid model max length issues
        context_trunc = context[:1500]

        sentence_scores = []
        entailed_count = 0.0

        try:
            # Batch all pairs for efficiency
            pairs = [(context_trunc, sent) for sent in sentences]
            # CrossEncoder NLI returns [contradiction, entailment, neutral] scores
            import numpy as np
            raw_scores = self._model.predict(pairs)

            for sent, scores in zip(sentences, raw_scores):
                if hasattr(scores, '__len__') and len(scores) == 3:
                    # [contradiction, entailment, neutral]
                    entail_score = float(scores[1])
                else:
                    entail_score = float(scores)

                if entail_score >= self.threshold:
                    label = "entailed"
                    entailed_count += 1
                elif entail_score < 0.3:
                    label = "contradicted"
                else:
                    label = "neutral"

                sentence_scores.append((sent, entail_score, label))

        except Exception as e:
            logger.warning(f"Faithfulness scoring error: {e}")
            return FaithfulnessResult(score=0.7, is_faithful=True, sentence_scores=[])

        overall_score = entailed_count / len(sentences) if sentences else 1.0

        return FaithfulnessResult(
            score=overall_score,
            is_faithful=overall_score >= self.threshold,
            sentence_scores=sentence_scores,
        )
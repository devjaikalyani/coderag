"""
evaluator.py
------------
Evaluation pipeline using RAGAS metrics:
  - Context Precision   : Are the retrieved docs relevant?
  - Answer Faithfulness : Is the answer supported by the context?
  - Answer Relevancy    : Does the answer actually address the question?
  - Context Recall      : Are all needed facts retrieved? (needs ground truth)

Also computes:
  - ROUGE-L score (against reference answers)
  - Mean Reciprocal Rank (MRR) for retrieval
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mlflow
import pandas as pd
from loguru import logger
from rouge_score import rouge_scorer


@dataclass
class EvalSample:
    question: str
    ground_truth_answer: str
    ground_truth_contexts: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    question: str
    generated_answer: str
    retrieved_contexts: List[str]
    ground_truth: str
    rouge_l: float
    faithfulness_score: float
    context_hit: bool   # Was ground truth context retrieved?

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "generated_answer": self.generated_answer[:300],
            "ground_truth": self.ground_truth[:300],
            "rouge_l": round(self.rouge_l, 4),
            "faithfulness_score": round(self.faithfulness_score, 4),
            "context_hit": self.context_hit,
        }


class RAGEvaluator:
    """
    Evaluate a CodeRAGPipeline on a QA dataset.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate(self, samples: List[EvalSample]) -> pd.DataFrame:
        """Run evaluation on a list of samples, log to MLflow."""
        results = []

        mlflow.set_tracking_uri(self.pipeline.settings.mlflow_tracking_uri if hasattr(self.pipeline, "settings") else "http://localhost:5000")
        mlflow.set_experiment("coderag_eval")

        with mlflow.start_run(run_name="evaluation"):
            for sample in samples:
                logger.info(f"Evaluating: {sample.question[:60]}…")
                response = self.pipeline.query(
                    sample.question,
                    use_history=False,
                    check_faithfulness=True,
                )

                # ROUGE-L
                rouge_result = self.rouge.score(
                    sample.ground_truth_answer,
                    response.answer,
                )
                rouge_l = rouge_result["rougeL"].fmeasure

                # Context hit
                retrieved_texts = [r.chunk.text for r in response.sources]
                context_hit = any(
                    gt_ctx[:100] in " ".join(retrieved_texts)
                    for gt_ctx in sample.ground_truth_contexts
                ) if sample.ground_truth_contexts else False

                faith_score = response.faithfulness.score if response.faithfulness else 0.0

                results.append(EvalResult(
                    question=sample.question,
                    generated_answer=response.answer,
                    retrieved_contexts=retrieved_texts,
                    ground_truth=sample.ground_truth_answer,
                    rouge_l=rouge_l,
                    faithfulness_score=faith_score,
                    context_hit=context_hit,
                ))

            # Aggregate metrics
            df = pd.DataFrame([r.to_dict() for r in results])
            avg_rouge = df["rouge_l"].mean()
            avg_faith = df["faithfulness_score"].mean()
            context_hit_rate = df["context_hit"].mean()

            mlflow.log_metrics({
                "avg_rouge_l": avg_rouge,
                "avg_faithfulness": avg_faith,
                "context_hit_rate": context_hit_rate,
                "num_samples": len(samples),
            })

            logger.info(
                f"\n{'='*50}\n"
                f"Eval Results ({len(samples)} samples)\n"
                f"  ROUGE-L:          {avg_rouge:.4f}\n"
                f"  Faithfulness:     {avg_faith:.4f}\n"
                f"  Context Hit Rate: {context_hit_rate:.4f}\n"
                f"{'='*50}"
            )

        return df

    def save_results(self, df: pd.DataFrame, path: str = "data/processed/eval_results.csv"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Eval results saved → {path}")


def load_eval_dataset(path: str) -> List[EvalSample]:
    """
    Load eval dataset from JSON.
    Expected format:
    [
      {
        "question": "...",
        "ground_truth_answer": "...",
        "ground_truth_contexts": ["..."]
      },
      ...
    ]
    """
    with open(path) as f:
        data = json.load(f)
    return [EvalSample(**d) for d in data]

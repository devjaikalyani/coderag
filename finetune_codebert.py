"""
finetune_codebert.py
--------------------
Fine-tune CodeBERT as a bi-encoder for code search using
contrastive learning (MultipleNegativesRankingLoss).

Dataset: CodeSearchNet (Python subset from HuggingFace)
  - Positive pairs: (docstring/comment, function body)
  - Hard negatives: random in-batch negatives

This produces a model better suited for code retrieval
than the raw CodeBERT weights.

Usage:
  python scripts/finetune_codebert.py --output models/codebert-finetuned
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader
import mlflow


def load_codesearchnet(split: str = "train", max_samples: int = 50000):
    """
    Load CodeSearchNet Python split.
    Each sample has: func_documentation_string, whole_func_string
    """
    logger.info(f"Loading CodeSearchNet (split={split}, max={max_samples})…")
    ds = load_dataset("code_search_net", "python", split=split, trust_remote_code=True)
    ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for item in ds:
        doc = item.get("func_documentation_string", "").strip()
        code = item.get("whole_func_string", "").strip()
        if doc and code and len(doc) > 10 and len(code) > 20:
            examples.append(InputExample(texts=[doc, code]))

    logger.info(f"Loaded {len(examples)} training pairs")
    return examples


def train(
    base_model: str = "microsoft/codebert-base",
    output_path: str = "models/codebert-finetuned",
    num_epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 200,
    max_samples: int = 50000,
    learning_rate: float = 2e-5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # Load base model
    model = SentenceTransformer(base_model, device=device)

    # Load data
    train_examples = load_codesearchnet(split="train", max_samples=max_samples)
    val_examples = load_codesearchnet(split="validation", max_samples=2000)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # MultipleNegativesRankingLoss:
    # For each (query, positive) pair, all other positives in the batch
    # act as hard negatives. Very effective for retrieval models.
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator on validation set
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    val_queries = [e.texts[0] for e in val_examples[:500]]
    val_docs = [e.texts[1] for e in val_examples[:500]]
    val_labels = [1.0] * 500   # All pairs are positive
    evaluator = EmbeddingSimilarityEvaluator(val_queries, val_docs, val_labels)

    # MLflow tracking
    mlflow.set_experiment("codebert_finetune")
    with mlflow.start_run():
        mlflow.log_params({
            "base_model": base_model,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "loss": "MultipleNegativesRankingLoss",
        })

        logger.info("Starting fine-tuning…")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True,
            evaluation_steps=500,
        )

        # Log final metrics
        final_score = evaluator(model)
        mlflow.log_metric("final_similarity_score", final_score)
        mlflow.log_artifact(output_path)

    logger.info(f"Fine-tuned model saved → {output_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for code search")
    parser.add_argument("--base-model", default="microsoft/codebert-base")
    parser.add_argument("--output", default="models/codebert-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    train(
        base_model=args.base_model,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()

"""Eval harness — run pipeline on golden QA pairs and compute RAGAS metrics.

Usage:
    python -m src.eval
    python -m src.eval --config benchmarks/M0/config.yaml
    python -m src.eval --config benchmarks/M0/config.yaml --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from dotenv import load_dotenv
from llama_index.core.schema import QueryBundle, TextNode
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import PipelineConfig
from src.pipeline.factory import build_generator, build_index, build_retriever

load_dotenv()


def load_qa_pairs(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return cast(list[dict[str, Any]], json.load(f))


def run_eval(
    config_path: Path,
    output_path: Path | None = None,
    generator: Any = None,
    retriever: Any = None,
) -> dict[str, Any]:
    config = PipelineConfig.from_yaml(config_path)

    if retriever is None or generator is None:
        print("Building index...")
        index = build_index(config)
        retriever = build_retriever(index, config.retrieval.top_k)
        generator = build_generator(config)

    qa_pairs = load_qa_pairs(Path("data/eval/qa.json"))
    print(f"Evaluating {len(qa_pairs)} QA pairs...")

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, pair in enumerate(qa_pairs):
        q = pair["question"]
        gt = pair["reference_answer"]

        print(f"  [{i + 1}/{len(qa_pairs)}] {q[:60]}...")
        nodes = retriever.retrieve(QueryBundle(q))
        ctx = [cast(TextNode, n.node).text for n in nodes]
        answer = generator.generate(q, nodes)

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    print("Computing RAGAS metrics...")
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {k: float(v) for k, v in result.items()}

    per_question = [
        {
            "question": q,
            "reference_answer": gt,
            "generated_answer": a,
            "contexts": ctx,
        }
        for q, gt, a, ctx in zip(questions, ground_truths, answers, contexts, strict=True)
    ]

    return {
        "config_path": str(config_path),
        "config": config.model_dump(),
        "scores": scores,
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--config",
        default="benchmarks/M0/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSON",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    output = run_eval(config_path)

    print("\n=== RAGAS Scores ===")
    for metric, score in sorted(output["scores"].items()):
        print(f"  {metric}: {score:.4f}")

    output_path = Path(args.output) if args.output else config_path.parent / "eval_results.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

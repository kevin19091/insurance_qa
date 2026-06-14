"""Eval harness — run pipeline on golden QA pairs and compute RAGAS metrics.

Usage:
    python -m src.eval
    python -m src.eval --config benchmarks/M0/config.yaml
    python -m src.eval --config benchmarks/M0/config.yaml --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, Sequence, Value
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
from src.pipeline.factory import build_generator, build_index, build_retriever, build_rewriter
from src.pipeline.retriever import retrieve_with_rewriting

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

    rewriter = build_rewriter(config, generator=generator)

    qa_pairs = load_qa_pairs(Path("data/eval/qa.json"))
    print(f"Evaluating {len(qa_pairs)} QA pairs...")

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    query_timings: list[dict[str, float]] = []

    for i, pair in enumerate(qa_pairs):
        q = pair["question"]
        gt = pair["reference_answer"]

        print(f"  [{i + 1}/{len(qa_pairs)}] {q[:60]}...")

        t0 = time.time()
        if config.query_rewrite.enabled:
            nodes = retrieve_with_rewriting(retriever, rewriter, QueryBundle(q))
        else:
            nodes = retriever.retrieve(QueryBundle(q))
        retrieval_ms = (time.time() - t0) * 1000

        t0 = time.time()
        ctx = [cast(TextNode, n.node).text for n in nodes]
        answer = generator.generate(q, nodes)
        generation_ms = (time.time() - t0) * 1000

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)
        query_timings.append(
            {
                "retrieval_latency_ms": round(retrieval_ms, 1),
                "generation_latency_ms": round(generation_ms, 1),
                "total_latency_ms": round(retrieval_ms + generation_ms, 1),
            }
        )

    print("Computing RAGAS metrics...")
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": [list(c) for c in contexts],
            "ground_truth": ground_truths,
        }
    )
    dataset = dataset.cast_column("contexts", Sequence(Value("string")))

    t0 = time.time()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    ragas_seconds = round(time.time() - t0, 2)

    scores = {k: float(v) for k, v in result.items()}

    per_question = [
        {
            "question": q,
            "reference_answer": gt,
            "generated_answer": a,
            "contexts": ctx,
            "latency": timing,
        }
        for q, gt, a, ctx, timing in zip(
            questions, ground_truths, answers, contexts, query_timings, strict=True
        )
    ]

    retrieval_times = [t["retrieval_latency_ms"] for t in query_timings]
    generation_times = [t["generation_latency_ms"] for t in query_timings]

    def stats(values: list[float]) -> dict[str, float]:
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "avg_ms": round(sum(values) / n, 1) if n else 0.0,
            "min_ms": round(sorted_v[0], 1) if n else 0.0,
            "max_ms": round(sorted_v[-1], 1) if n else 0.0,
            "p50_ms": round(sorted_v[n // 2], 1) if n else 0.0,
            "p95_ms": round(sorted_v[int(n * 0.95)], 1) if n else 0.0,
        }

    return {
        "config_path": str(config_path),
        "config": config.model_dump(),
        "scores": scores,
        "per_question": per_question,
        "latency": {
            "retrieval": stats(retrieval_times),
            "generation": stats(generation_times),
            "total": stats([t["total_latency_ms"] for t in query_timings]),
            "ragas_eval_seconds": ragas_seconds,
        },
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

    lat = output["latency"]
    print("\n=== Latency ===")
    for phase in ["retrieval", "generation", "total"]:
        s = lat[phase]
        print(f"  {phase}: avg={s['avg_ms']}ms p50={s['p50_ms']}ms p95={s['p95_ms']}ms")

    output_path = Path(args.output) if args.output else config_path.parent / "eval_results.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
